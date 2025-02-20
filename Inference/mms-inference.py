from transformers import pipeline, Wav2Vec2ForCTC, AutoProcessor
from datasets import load_dataset, Audio
import numpy as np
import torch, random

# MMS REQUIRES LANG ID SINCE IT USES A DIFFERENT ADAPTER PER LANGUAGE. 

dataset_name = '' # provide dataset name/path
cache_dir = '' # provide path to your preferred cache directory
reference_label = '' # depending on the dataset, provide the name used for ground truth (e.g. 'sentence', 'label', 'transcript')
language = '' # mms requires lang labels under a specific format. Find the ids under https://huggingface.co/facebook/mms-1b-all#supported-languages (you have to click on the toggle to unravel the list)
test_split = '' # provide the name of the split you're doing inference on (e.g. test)
output_file_path = f'../Predictions/{dataset_name}-mms-{language or "noLID"}.tsv' # if there's a slash in your dataset_name make sure to slice e.g. for dataset_name 'sqrk/dataset', use dataset_name[5:] in your output file name
model_kwargs = {"target_lang": language, "ignore_mismatched_sizes": True} # REQUIRED 
model_name = 'facebook/mms-1b-all' 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch_dtype = torch.float16 if device.type == 'cuda' else torch.float32
batch_size = 16 # change batch size if needed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

print(f'DATASET: {dataset_name}\nMODEL: {model_name}\nLANGUAGE: {language or "noLID"}\nSEED: {seed}')

dataset = load_dataset(dataset_name, cache_dir=cache_dir)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
dataset[test_split] = dataset[test_split].shuffle(seed=seed)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model_name,
    torch_dtype=torch_dtype,
    device=device,
    model_kwargs=model_kwargs
)

pipe.model.load_adapter(language) # REQUIRED


def transcribe(batch):
    file_names = [audio['path'] for audio in batch['audio']]
    audio_arrays = [np.array(audio["array"], dtype=np.float32) for audio in batch["audio"]]
    results = pipe(audio_arrays, batch_size=batch_size)
    return {"prediction": [result["text"] for result in results],
            "file_name": file_names}


dataset[test_split] = dataset[test_split].map(transcribe, batched=True, batch_size=batch_size)

with open(output_file_path, 'w') as fout:
    fout.write('file_name\treference\tprediction\n')
    for item in dataset[test_split]:
        fout.write(f"{item['file_name']}\t{item[reference_label]}\t{item['prediction']}\n")
        
print("Inference complete. Predictions saved.")
