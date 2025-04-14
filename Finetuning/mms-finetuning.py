import time 
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer, TrainerCallback
from safetensors.torch import save_file as safe_save_file
from transformers.models.wav2vec2.modeling_wav2vec2 import WAV2VEC2_ADAPTER_SAFE_FILE
from datasets import load_dataset, Audio
import numpy as np
import torch, random, json, re, os, evaluate
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


# MMS REQUIRES LANG ID SINCE IT USES A DIFFERENT ADAPTER PER LANGUAGE. 

dataset_name = '' # provide dataset name/path
cache_dir = '' # provide path to your preferred cache directory
reference_label = '' # depending on the dataset, provide the name used for ground truth (e.g. 'sentence', 'label', 'text')
language = '' # mms requires lang labels under a specific format. Find the ids under https://huggingface.co/facebook/mms-1b-all#supported-languages (you have to click on the toggle to unravel the list)
model_name = 'facebook/mms-1b-all' 
HF_username = '' # specify ur huggingface username
hub_token = '' # provide a hugginface write token to ur account
output_dir = f'{HF_username}/{model_name[9:]}FT-{dataset_name}-{language or "noLID"}' # if there's a slash in your dataset_name make sure to slice e.g. for dataset_name 'sqrk/dataset', use dataset_name[5:] in your output file name
resume_from_checkpoint = '' # if resuming from previous job, provide checkpoint path, otherwise leave empty
max_hours = 23 # provide the max hours you can train (job limit - 1 to be safe. For HPC that would be 11, and 23 for CSCC)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch_dtype = torch.float16 if device.type == 'cuda' else torch.float32
batch_size = 4 # change batch size if needed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


start_time = time.time() 

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
            print(f"New best model. Saving to {output_dir}/best_model")
            kwargs["model"].save_pretrained(f"{output_dir}/best_model")
            processor.save_pretrained(f"{output_dir}/best_model")

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    batch["labels"] = processor(text=batch[reference_label]).input_ids
    return batch

def remove_special_characters(batch):
    batch[reference_label] = re.sub(chars_to_remove_regex, '', batch[reference_label]).lower()
    return batch

def extract_all_chars(batch):
  all_text = " ".join(batch[reference_label])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}

dataset = load_dataset(dataset_name, cache_dir=cache_dir)
dataset = dataset.cast_column('audio', Audio(sampling_rate=16000))
dataset = dataset['train'].train_test_split(test_size=0.1)   # using 10% of train has val
dataset['train'] = dataset['train'].shuffle(seed=seed)


# preparing vocab for mms
chars_to_remove_regex = '[,?.!-;:\"“%‘”�\'،؟<>=!\\t/]'
dataset = dataset.map(remove_special_characters)

vocab_train = dataset['train'].map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=dataset['train'].column_names) 
vocab_test = dataset['test'].map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=dataset['test'].column_names) 

vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
len(vocab_dict)
target_lang = 'ara'
new_vocab_dict = {target_lang: vocab_dict}

os.makedirs(output_dir, exist_ok=True)
with open(f'{output_dir}/vocab.json', 'w') as vocab_file:
    json.dump(new_vocab_dict, vocab_file)

tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(output_dir, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|", target_lang=target_lang, cache_dir=cache_dir)
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True, cache_dir=cache_dir)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
tokenizer.push_to_hub(output_dir, token=hub_token)

dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names['train']) 


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

metric = evaluate.load("wer", cache_dir=cache_dir)

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    wer = metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

model = Wav2Vec2ForCTC.from_pretrained(
    model_name,
    attention_dropout=0.0,
    hidden_dropout=0.0,
    feat_proj_dropout=0.0,
    layerdrop=0.0,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
    ignore_mismatched_sizes=True,
    cache_dir = cache_dir
)

model.init_adapter_layers()
model.freeze_base_model()

adapter_weights = model._get_adapters()
for param in adapter_weights.values():
    param.requires_grad = True


training_args = TrainingArguments(
    deepspeed="ds_config.json",
    output_dir=output_dir,
    group_by_length=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=8,
    # gradient_checkpointing=True, #this is throwing segmentation fault, i'm not sure why so i just removed it for now
    gradient_checkpointing=False,
    save_strategy='epoch',
    evaluation_strategy='epoch',
    logging_strategy='epoch',
    num_train_epochs=100,
    fp16=True,
    logging_steps=100,
    learning_rate=1e-5,
    warmup_steps=500,
    save_total_limit=2,
    push_to_hub=True,
    report_to=["wandb"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    hub_token=hub_token,
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    tokenizer=processor.feature_extractor,
    callbacks=[StopAfterTimeLimitCallback(max_hours=max_hours),
               SaveBestModelCallback] 
)

trainer.train()

adapter_file = WAV2VEC2_ADAPTER_SAFE_FILE.format(target_lang)
adapter_file = os.path.join(training_args.output_dir, adapter_file)

safe_save_file(model._get_adapters(), adapter_file, metadata={"format": "pt"})

trainer.push_to_hub()
