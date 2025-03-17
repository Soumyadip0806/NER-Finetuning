import os
import gc
import torch
import pandas as pd
import numpy as np
from collections import Counter
import torch.nn as nn

from transformers import (AutoModelForTokenClassification, Trainer, BertTokenizerFast,
                          TrainingArguments)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score


#------------ Function to Clear GPU Cache ------------#

def clear_gpu_cache():
    torch.cuda.empty_cache()
    gc.collect()



#------------ Model Paths ------------#

model_names = [
    "model_1",
    "model_2",
    "model_3"
]


#------------ Fine-tuning Function ------------#

def finetune_model(model_name):
    print(f"\n🚀 Starting fine-tuning for: {model_name} ...")

    # Setup directories
    safe_model_name = model_name.replace("/", "_")
    main_dir = f'./{safe_model_name}'
    model_dir = os.path.join(main_dir, 'model')
    tokenizer_dir = os.path.join(main_dir, 'tokenizer')
    logs_dir = os.path.join(main_dir, 'logs')
    os.makedirs(main_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tokenizer_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)




    #------------ Load Dataset ------------#

    df = pd.read_csv("/home2/soumyadip.ghosh/Mizo_NER.csv") # Dataset path
    train_texts, val_texts, train_tags, val_tags = train_test_split(df["Tokens"].tolist(), df["Tags"].tolist(), test_size=0.2, random_state=42)

    # Convert string tokens into lists
    train_texts = [sentence.split() for sentence in train_texts]
    train_tags = [tags.split() for tags in train_tags]
    val_texts = [sentence.split() for sentence in val_texts]
    val_tags = [tags.split() for tags in val_tags]




    #------------ Label Mapping & Class Weights ------------#

    label_list = sorted(set(tag for doc in train_tags for tag in doc))
    label_map = {label: i for i, label in enumerate(label_list)}
    
    label_counts = Counter([tag for doc in train_tags for tag in doc])
    total_labels = sum(label_counts.values())
    class_weights = {label: total_labels / count for label, count in label_counts.items()}
    weights_tensor = torch.tensor([class_weights[label] for label in label_list], dtype=torch.float32).cuda()




    #------------ Tokenization ------------#

    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    def tokenize_and_align_labels(texts, tags):
        tokenized_inputs = tokenizer(texts, is_split_into_words=True, truncation=True, padding="max_length", max_length=128)

        labels = []
        for i, label in enumerate(tags):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []

            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label_map[label[word_idx]])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    train_encodings = tokenize_and_align_labels(train_texts, train_tags)
    val_encodings = tokenize_and_align_labels(val_texts, val_tags)




    #------------ Create Dataset Class ------------#

    class NERDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings
        def __len__(self):
            return len(self.encodings["input_ids"])
        def __getitem__(self, idx):
            return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    train_dataset = NERDataset(train_encodings)
    val_dataset = NERDataset(val_encodings)




    #------------ Load Model ------------#

    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(label_list))




    #------------ Custom Trainer with Weighted Loss ------------#

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fn = nn.CrossEntropyLoss(weight=weights_tensor, ignore_index=-100)
            loss = loss_fn(logits.view(-1, len(label_list)), labels.view(-1))
            return (loss, outputs) if return_outputs else loss




    #------------ Compute Metrics ------------#

    def compute_metrics(eval_pred, logs_dir="./logs"):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
        true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
        
        flat_predictions = [item for sublist in true_predictions for item in sublist]
        flat_labels = [item for sublist in true_labels for item in sublist]
        
        y_pred_filtered = [pred for pred, true in zip(flat_predictions, flat_labels) if true != 'O']
        y_true_filtered = [true for true in flat_labels if true != 'O']

        if not y_true_filtered:
            return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}

        precision, recall, f1, _ = precision_recall_fscore_support(y_true_filtered, y_pred_filtered, average="micro", zero_division=0)
        accuracy = accuracy_score(y_true_filtered, y_pred_filtered)
        
        report_file = os.path.join(logs_dir, "classification_report.txt")
        with open(report_file, "w") as f:
            f.write(classification_report(y_true_filtered, y_pred_filtered, digits=4, zero_division=0))
        print(f"📄 Classification report saved to: {report_file}")
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}




    #------------ Training Arguments & Execution ------------#

    training_args = TrainingArguments(
        output_dir=main_dir,
        evaluation_strategy="epoch",
        save_strategy="no",
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=4,
        weight_decay=0.01,
        logging_dir=logs_dir,
        logging_steps=100,
        load_best_model_at_end=False,
        warmup_steps=500,
        report_to="none"
    )

    trainer = CustomTrainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset, compute_metrics=lambda eval_pred: compute_metrics(eval_pred, logs_dir=logs_dir))
    trainer.train()
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(tokenizer_dir)
    clear_gpu_cache()
    print(f"✅ Fine-tuning completed for: {model_name}\n")

for model_name in model_names:
    finetune_model(model_name)
