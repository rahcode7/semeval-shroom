#
# This script performs exploratory modeling for Shroom Task 6
# It uses DeBERTa model for sequence classification
# The model is trained on a combined dataset of model-aware and model-agnostic data
# The script also includes data preprocessing and splitting into training and validation sets
# Finally, the trained model is saved and used for inference on test data
#

model_aware = "/content/shroom6/labeled.model-aware.v2.json"
model_agnostic = "/content/shroom6/labeled.model-agnostic.json"
test_data = "/content/shroom6/test_data.csv"

import os
import numpy as np
import pandas as pd
from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer, Trainer, TrainingArguments

from tqdm.notebook import tqdm
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import torch
from sklearn.utils import resample

le = LabelEncoder()

df_model_aware = pd.read_json(model_aware)
df_model_agnostic = pd.read_json(model_agnostic)
df_test = pd.read_csv(test_data)

df = pd.concat([df_model_aware, df_model_agnostic], ignore_index=True)

cond = df['hyp'].isin(df_test['hyp']) 
df_train = df.drop(df[cond].index).reset_index(drop=True)

df_train["label_encoded"] = le.fit_transform(df_train["label"])

df_train.tail()

len(df_train), len(df_test), len(df)

df_train.tail()

df_train["hyp"][0], df_train["src"][0], df_train["tgt"][0]

df_train.value_counts("label_encoded"), df_train.value_counts("label")

df_train.value_counts("task")

df_train["src_hyp_tgt"] = df_train["src"] +" "+ df_train["hyp"] + " " + df_train["tgt"]

df_train

df_train["src_hyp_tgt"][0]

np.unique(df_train['label_encoded'])

df_train.value_counts("label_encoded"), df_train.value_counts("label")

train_df, val_df = train_test_split(df_train, test_size=0.2, stratify=df_train["label"], random_state=42)

train_df.value_counts("label"), val_df.value_counts("label")

len(train_df), len(val_df)

MODEL_NAME = "microsoft/deberta-v3-xsmall"

class SentenceDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = str(self.data.iloc[idx]['src_hyp_tgt'])
        label = int(self.data.iloc[idx]['label_encoded'])

        encoding = self.tokenizer(sentence, return_tensors='pt', truncation=True, padding='max_length', max_length=512)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': label
        }

train_dataset = SentenceDataset(train_df)
val_dataset = SentenceDataset(val_df)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

train_dataset.__getitem__(0)

model = DebertaV2ForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,  
    learning_rate=2e-5,   
    metric_for_best_model='eval_loss',  
    greater_is_better=False,  
    report_to='none'
) 

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

model.save_pretrained("./dberta_finetuned_model")

from transformers import pipeline

df_test["src_hyp_tgt"] = df_test["src"] +" "+ df_test["hyp"] + " " + df_test["tgt"]

df_test.head()

sentence_relevance_classifier = pipeline(
    task="text-classification",
    model="./dberta_finetuned_model",  # Path to the saved model directory
    tokenizer=MODEL_NAME,
    return_all_scores=True,
)
sentence = df_test["src_hyp_tgt"][0]
# Perform inference on new sentences
result = sentence_relevance_classifier(sentence)
