from transformers import pipeline
from datasets import load_dataset, Audio
import numpy as np
import torch, random


dataset_name = '' # provide dataset name/path
cache_dir = '' # provide path to your preferred cache directory
reference_label = '' # depending on the dataset, provide the name used for ground truth (e.g. 'sentence', 'label', 'transcript')
language = '' # leave empty if no LID, otherwise provide language
test_split = '' # provide the name of the split you're doing inference on (e.g. test)
output_file_path = f'../Predictions/{dataset_name}-whisper-{language or "noLID"}.tsv' # if there's a slash in your dataset_name make sure to slice e.g. for dataset_name 'sqrk/dataset', use dataset_name[5:] in your output file name
generate_kwargs = {"language": language} if language else {}
model_name = 'openai/whisper-large-v3' 
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
    device=device
)

def transcribe(batch):
    file_names = [audio['path'] for audio in batch['audio']]
    audio_arrays = [np.array(audio["array"], dtype=np.float32) for audio in batch["audio"]]
    results = pipe(audio_arrays, batch_size=batch_size, chunk_length_s=30.0, generate_kwargs=generate_kwargs)
    return {"prediction": [result["text"] for result in results],
            "file_name": file_names}

dataset[test_split] = dataset[test_split].map(transcribe, batched=True, batch_size=batch_size)

with open(output_file_path, 'w') as fout:
    fout.write('file_name\treference\tprediction\n')
    for item in dataset[test_split]:
        fout.write(f"{item['file_name']}\t{item[reference_label]}\t{item['prediction']}\n")

print("Inference complete. Predictions saved.")
