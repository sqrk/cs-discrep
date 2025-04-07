import time 
import torch, re, evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset, Audio
import numpy as np
import torch, random
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoProcessor, AutoModelForSpeechSeq2Seq, TrainerCallback

dataset_name = '' # provide dataset name/path
cache_dir = '' # provide path to your preferred cache directory
reference_label = '' # depending on the dataset, provide the name used for ground truth (e.g. 'sentence', 'label', 'transcript')
language = ''
# model_name = 'openai/whisper-large-v3'
model_name = 'openai/whisper-medium'
HF_username = '' # specify ur huggingface username
hub_token = '' # provide a hugginface write token to ur account
output_dir = f'{HF_username}/{model_name[7:]}FT-{dataset_name[5:]}-{language or "noLID"}' # if there's a slash in your dataset_name make sure to slice e.g. for dataset_name 'sqrk/dataset', use dataset_name[5:] in your output file name
max_hours = 11 # provide the max hours you can train (job limit - 1 to be safe. For HPC that would be 11, and 23 for CSCC)
resume_from_checkpoint = '' # if resuming from previous job, provide checkpoint path, otherwise leave empty
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch_dtype = torch.float16 if device.type == 'cuda' else torch.float32
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

start_time = time.time()


print(f'DATASET: {dataset_name}\nMODEL: {model_name}\nLANGUAGE: {language or "noLID"}\nCKPT DIR: {output_dir}\nSEED: {seed}')

class StopAfterTimeLimitCallback(TrainerCallback): # ensures that training wraps up before job is killed by cluster/HPC
    def __init__(self, max_hours):
        self.max_seconds = max_hours * 3600  

    def on_step_end(self, args, state, control, **kwargs):
        if time.time() - start_time > self.max_seconds: # check if time limit exceeded
            print(f"Reached the time limit of {max_hours} hours, stopping training.")
            control.should_training_stop = True  
        return control

class SaveBestModelCallback(TrainerCallback): # ensures the best model is saved after each epoch (and isn't overwritten by the last one) in case the training crashes
    def on_evaluate(self, args, state, control, **kwargs):
        if control.should_save and state.best_model_checkpoint:
            print(f"Saving best model to {output_dir}/best_model")
            kwargs["model"].save_pretrained(f"{output_dir}/best_model")
            processor.save_pretrained(f"{output_dir}/best_model")

def prepare_dataset(batch):
    audio = batch['audio']
    batch["input_features"] = processor.feature_extractor(audio['array'], sampling_rate=audio['sampling_rate']).input_features[0] 
    batch["labels"] = processor.tokenizer(batch[reference_label]).input_ids
    return batch

dataset = load_dataset(dataset_name, cache_dir=cache_dir)
dataset = dataset.cast_column('audio', Audio(sampling_rate=16000))
dataset = dataset['train'].train_test_split(test_size=0.1)   # using 10% of train has val
dataset['train'] = dataset['train'].shuffle(seed=seed)

if language:
    processor = AutoProcessor.from_pretrained(model_name, language=language, task="transcribe", cache_dir=cache_dir) 
else:
    processor = AutoProcessor.from_pretrained(model_name, task="transcribe", cache_dir=cache_dir)

dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names['train']) 

model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, cache_dir=cache_dir) 
model.to(device)
# model.config.use_cache = False #incompatible with gradient checkpointing

model.generation_config.task = "transcribe"
if language:
    model.generation_config.language = "arabic" 
model.generation_config.forced_decoder_ids = None


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

metric = evaluate.load("wer", cache_dir=cache_dir)

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id  
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,  
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    num_train_epochs=100,
    # gradient_checkpointing=True, #this is throwing segmentation fault, i'm not sure why so i just removed it for now
    gradient_checkpointing=False,
    fp16=True, 
    save_strategy='epoch',
    evaluation_strategy='epoch',
    logging_strategy='epoch',
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=150,
    logging_steps=100,
    report_to=["wandb"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
    save_total_limit=2,
    hub_token=hub_token,
    logging_dir=''
)


trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
    callbacks=[StopAfterTimeLimitCallback(max_hours=max_hours),
               SaveBestModelCallback] 
)

if resume_from_checkpoint:
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
else:
    trainer.train()

trainer.save_model()
processor.tokenizer.save_pretrained(training_args.output_dir)