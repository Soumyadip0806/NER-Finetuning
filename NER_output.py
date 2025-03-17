import torch
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification
import numpy as np




#--------------- Define model and tokenizer paths ---------------#
MODEL_PATH = "/home2/soumyadip.ghosh/Muril-base-cased_Mizo_model/model"  # Update as needed
TOKENIZER_PATH = MODEL_PATH

# Load tokenizer and model
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
model.eval().cuda()
print("Model loaded successfully.")



#--------------- Load label list from CSV ---------------#
train_df = pd.read_csv("/home2/soumyadip.ghosh/Mizo_NER_Headword.csv")
label_list = sorted(set(tag for tags in train_df["Tags"].dropna() for tag in tags.split()))



#--------------- Read input sentences ---------------#
input_file = "/home2/soumyadip.ghosh/hindi_mizo_raw.txt"  # Update as needed
output_file = "/home2/soumyadip.ghosh/muril_output.txt"

with open(input_file, "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f.readlines() if line.strip()]



#--------------- Named Entity Recognition (NER) Prediction Function ---------------#
def predict_ner(sentence, sentence_id, max_len=512):
    tokens = sentence.split()
    chunk_size = max_len - 2  # Account for special tokens like [CLS] and [SEP]
    token_labels = []
    
    for start in range(0, len(tokens), chunk_size):
        sub_tokens = tokens[start:start + chunk_size]
        tokenized_inputs = tokenizer(sub_tokens, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        word_ids = tokenized_inputs.word_ids(batch_index=0)  # Get word alignment

        inputs = {key: val.cuda() for key, val in tokenized_inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs).logits.cpu().numpy()

        predictions = np.argmax(outputs, axis=2)[0]
        previous_word_idx = None

        for word_idx, pred in zip(word_ids, predictions):
            if word_idx is None or word_idx == previous_word_idx:
                continue  # Skip special tokens
            token_labels.append((tokens[start + word_idx], label_list[pred]))
            previous_word_idx = word_idx

    return token_labels



#--------------- Process sentences and save predictions ---------------#
with open(output_file, "w", encoding="utf-8") as f:
    for i, sentence in enumerate(sentences):
        f.write(f"# Sentence_Id = {i+1}\n")
        f.write(f"# Sentence : {sentence}\n\n")
        
        token_labels = predict_ner(sentence, i+1)
        for j, (token, label) in enumerate(token_labels):
            f.write(f"{j+1}\t{token}\t{label}\n")
        
        f.write("\n")

print(f"✅ Predictions saved in {output_file}")
