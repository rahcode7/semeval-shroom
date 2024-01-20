
model_aware = "/content/shroom6/labeled.model-aware.v2.json"
model_agnostic = "/content/shroom6/labeled.model-agnostic.json"
test_data = "/content/shroom6/test_data.csv"

import os
import numpy as np
import pandas as pd
from transformers import TFAutoModel, AutoTokenizer
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import torch
from sklearn.utils import resample
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from kaggle_datasets import KaggleDatasets

def regular_encode(texts, tokenizer, maxlen=512):
    enc_di = tokenizer.batch_encode_plus(
         texts, 
         return_token_type_ids=False,
         pad_to_max_length=True,
         #padding=True,
         max_length=maxlen
     )
    return np.array(enc_di['input_ids'])

def build_model(transformer, max_len=512):
    """
    https:/www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras
    """
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(cls_token)
    
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

AUTO = tf.data.experimental.AUTOTUNE

EPOCHS = 5
BATCH_SIZE = 4
MAX_LEN = 512
MODEL = 'jplu/tf-xlm-roberta-large'

tokenizer = AutoTokenizer.from_pretrained(MODEL)

le = LabelEncoder()
df_model_aware = pd.read_json(model_aware)
df_model_agnostic = pd.read_json(model_agnostic)
df_test = pd.read_csv(test_data)

df = pd.concat([df_model_aware, df_model_agnostic], ignore_index=True)

cond = df['hyp'].isin(df_test['hyp']) 
df_train = df.drop(df[cond].index).reset_index(drop=True)

df_train["label_encoded"] = le.fit_transform(df_train["label"])

df_train.value_counts("label_encoded"), df_train.value_counts("label")

df_train["src_hyp_tgt"] = df_train["src"] +" "+ df_train["hyp"] + " " + df_train["tgt"]

df_test["src_hyp_tgt"] = df_test["src"] +" "+ df_test["hyp"] + " " + df_test["tgt"]

df_test = df_test[df_test["tgt"].notna()]

df_test

train_df, val_df = train_test_split(df_train, test_size=0.2, stratify=df_train["label"], random_state=42)

df_majority = train_df[train_df['label_encoded']==1]
df_minority = train_df[train_df['label_encoded']==0]
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # Sample with replacement
                                 n_samples=len(df_majority),    # Match number in majority class
                                 random_state=27) # For reproducible results 

# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])

# Display new class counts
df_upsampled['label_encoded'].value_counts()

train_df = df_upsampled

df_majority = val_df[val_df['label_encoded']==1]
df_minority = val_df[val_df['label_encoded']==0]
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # Sample with replacement
                                 n_samples=len(df_majority),    # Match number in majority class
                                 random_state=27) # For reproducible results 

# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])

# Display new class counts
df_upsampled['label_encoded'].value_counts()

val_df = df_upsampled

x_train = regular_encode(train_df.src_hyp_tgt.values.tolist(), tokenizer, maxlen=MAX_LEN)
x_valid = regular_encode(val_df.src_hyp_tgt.values.tolist(), tokenizer, maxlen=MAX_LEN)
x_test = regular_encode(df_test.src_hyp_tgt.values.tolist(), tokenizer, maxlen=MAX_LEN)

y_train = train_df.label_encoded.values.tolist()
y_valid = val_df.label_encoded.values.tolist()

train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_train, y_train))
    .repeat()
    .shuffle(2048)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_valid, y_valid))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(x_test)
    .batch(BATCH_SIZE)
)

# %%time
# with strategy.scope():
transformer_layer = TFAutoModel.from_pretrained(MODEL)
model = build_model(transformer_layer, max_len=MAX_LEN)
model.summary()

n_steps = x_train.shape[0] / BATCH_SIZE
train_history = model.fit(
    train_dataset,
    steps_per_epoch=n_steps,
    validation_data=valid_dataset,
    epochs=EPOCHS
)

prediction = model.predict(test_dataset, verbose=1)

prediction

df_test.head()

df_test.src_hyp_tgt.values.tolist()[0]

model.predict(regular_encode([df_test.src_hyp_tgt.values.tolist()[0]], tokenizer, maxlen=MAX_LEN))

model.predict(regular_encode([df_test.src_hyp_tgt.values.tolist()[7]], tokenizer, maxlen=MAX_LEN))

model.save("xlm-roberta-large.h5")