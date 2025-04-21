from jiwer import wer, cer
import pandas as pd
import regex 

dataset_name = 'mixat-FT-csonly'
preds_dir = '../Predictions/FT/'
whisper_filename = 'mixat-csonly-whisper-large-v3FT-mixat-tri-arabic-noLID.tsv' # provide name of whisper predictions file
mms_filename = 'mixat-csonly-mms-1b-allFT-mixat-tri-ara-ara.tsv'
# sm4t_filename = 'casablanca-sm4t-noLID-src_ary.tsv' 
output_file = f'./{dataset_name}-eval.txt' # provide path to where you want your evaluation

whisper_df = pd.read_csv(preds_dir + whisper_filename, sep='\t')
mms_df = pd.read_csv(preds_dir + mms_filename, sep='\t')
# sm4t_df = pd.read_csv(preds_dir + sm4t_filename, sep='\t')

paths = list(whisper_df['file_name'])
refs = list(whisper_df['reference'])
# refs_sm4t = list(sm4t_df['reference'])
whisper_preds = list(whisper_df['prediction'].fillna('')) # fillna to avoid having a float Na when the model output is blank
mms_preds = list(mms_df['prediction'].fillna(''))
# sm4t_preds = list(sm4t_df['prediction'].fillna(''))

def normalize(line):
    line = line.lower()
    line = regex.sub(r"[^\p{L}\p{N}\s]+", '', line)

    # Arabic norm
    line = regex.sub(r"[٠-٩]",lambda x: str(int(x.group())), line)
    line = regex.sub("[إأٱآا]", "ا", line)
    line = regex.sub("[ةه]", "ه", line)
    line = regex.sub(r'([ىيئ])(?=\s|[.,!?؛:\]،]|$)', 'ى', line) # changes ya to alef maqsura when at the end of the word
    
    line = ' '.join(line.split())
    return line


refs = [normalize(ref) for ref in refs]
whisper_norm_preds = [normalize(pred) for pred in whisper_preds]
mms_norm_preds = [normalize(pred) for pred in mms_preds]
# sm4t_norm_preds = [normalize(pred) for pred in sm4t_preds]


with open(output_file, 'w') as fout:
    fout.write('\t\t\t\tWhis\t\tMMS\t\t\tSM4T\n')
    fout.write('Overall WER:\t{:.2f}'.format(100 * wer(refs, whisper_norm_preds)) + 
               '\t\t{:.2f}'.format(100 * wer(refs, mms_norm_preds)) + 
            #    '\t\t{:.2f}'.format(100 * wer(refs, sm4t_norm_preds)) + '\n') 
            '\t\t--\n') 
    fout.write('Overall CER:\t{:.2f}'.format(100 * cer(refs, whisper_norm_preds)) + 
               '\t\t{:.2f}'.format(100 * cer(refs, mms_norm_preds)) + 
            #    '\t\t{:.2f}'.format(100 * cer(refs, sm4t_norm_preds)) + '\n') 
            '\t\t--\n') 
    
    fout.write('\n')
    
    fout.write('Individual:\n')
    # for path, ref, whis, mms, sm4t in zip(paths, refs, whisper_norm_preds, mms_norm_preds, sm4t_norm_preds):
    for path, ref, whis, mms in zip(paths, refs, whisper_norm_preds, mms_norm_preds):
        fout.write(f'{path}\n')
        fout.write('WER:\t\t\t{:.2f}'.format(100 * wer(ref, whis)) +
                   '\t\t{:.2f}'.format(100 * wer(ref, mms)) +
                #    '\t\t{:.2f}'.format(100 * wer(ref, sm4t)) + '\n') 
                    '\t\t--\n') 
        
        fout.write('CER:\t\t\t{:.2f}'.format(100 * cer(ref, whis)) +
                   '\t\t{:.2f}'.format(100 * cer(ref, mms)) +
                #    '\t\t{:.2f}'.format(100 * cer(ref, sm4t)) + '\n') 
                    '\t\t--\n') 
        fout.write('\n')
        

fout.close()