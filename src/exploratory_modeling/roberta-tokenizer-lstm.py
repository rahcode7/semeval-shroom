#
# This script performs exploratory modeling for Shroom Task 6
# It uses RoBERTa model for sequence classification
# The model is trained on a combined dataset of model-aware and model-agnostic data
# The script also includes data preprocessing and splitting into training and validation sets
# Finally, the trained model is saved and used for inference on test data
#
model_aware = "/content/shroom6/labeled.model-aware.v2.json"
model_agnostic = "/content/shroom6/labeled.model-agnostic.json"
test_data = "/content/shroom6/test_data.csv"

# Load, explore and plot data
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
get_ipython().run_line_magic('matplotlib', 'inline')
# Train test split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
# Text pre-processing
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
# Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from transformers import TFAutoModel, AutoTokenizer

from tensorflow.keras.layers import LSTM, GRU, Dense, Embedding, Dropout, GlobalAveragePooling1D, Flatten, SpatialDropout1D, Bidirectional

def regular_encode(texts, tokenizer, maxlen=512):
    enc_di = tokenizer.batch_encode_plus(
         texts, 
         return_token_type_ids=False,
         pad_to_max_length=True,
         #padding=True,
         max_length=maxlen
     )
    return np.array(enc_di['input_ids'])

le = LabelEncoder()

df_model_aware = pd.read_json(model_aware)
df_model_agnostic = pd.read_json(model_agnostic)
df_test = pd.read_csv(test_data)

df = pd.concat([df_model_aware, df_model_agnostic], ignore_index=True)

cond = df['hyp'].isin(df_test['hyp']) 
df_train = df.drop(df[cond].index).reset_index(drop=True)

df_train["label_encoded"] = le.fit_transform(df_train["label"])

len(df_train), len(df_test), len(df)

df_train["hyp"][0], df_train["src"][0], df_train["tgt"][0]

df_train.value_counts("label_encoded"), df_train.value_counts("label")

df_train.value_counts("task")

df_train["context"] = np.where(df_train["task"] == "PG", df_train["src"], df_train["tgt"])
df_test["context"] = np.where(df_test["task"] == "PG", df_test["src"], df_test["tgt"])

df_train["input_text"] = df_train["hyp"] + " " + df_train["context"]
df_test["input_text"] = df_test["hyp"] + " " + df_test["context"]

df_train["text_length"] = df_train["input_text"].apply(len)

max(df_train["text_length"].values)

df_test = df_test[df_test["tgt"].notna()]

df_train = df_train[["input_text", "label_encoded"]]
df_test = df_test[["input_text", "label"]]

df_train.groupby('label_encoded').describe().T

x_train, x_test, y_train, y_test = train_test_split(df_train["input_text"], df_train["label_encoded"],test_size=0.2, stratify=df_train["label_encoded"], random_state=42)

max_len = 512 
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'

tokenizer = AutoTokenizer.from_pretrained("jplu/tf-xlm-roberta-large")

training_padded = regular_encode(x_train.values.tolist(), tokenizer, maxlen=512)
testing_padded = regular_encode(x_test.values.tolist(), tokenizer, maxlen=512)

training_padded

tokenizer.vocab_size

# ## Dense Model

# Define parameter
vocab_size = tokenizer.vocab_size 
embedding_dim = 100
drop_value = 0.25
n_dense = 24
# Define Dense Model Architecture
model = Sequential()
model.add(Embedding(vocab_size,
                    embedding_dim,
                    input_length = max_len))
model.add(GlobalAveragePooling1D())
model.add(Dense(24, activation='relu'))
model.add(Dropout(drop_value))
model.add(Dense(1, activation='sigmoid'))

model.summary()

learning_rate = 1e-3  # Adjust the learning rate as needed

optimizer = Adam(learning_rate=learning_rate)

model.compile(loss = 'binary_crossentropy', optimizer = optimizer , metrics = ['accuracy'])

num_epochs = 30
early_stop = EarlyStopping(monitor='val_loss', patience=3)
history = model.fit(training_padded,
                    y_train,
                    epochs=num_epochs, 
                    validation_data=(testing_padded, y_test),
                    callbacks =[early_stop],
                    verbose=2)

model.evaluate(testing_padded, y_test)

train_dense_results = model.evaluate(training_padded, np.asarray(y_train), verbose=2, batch_size=256)
valid_dense_results = model.evaluate(testing_padded, np.asarray(y_test), verbose=2, batch_size=256)
print(f'Train accuracy: {train_dense_results[1]*100:0.2f}')
print(f'Valid accuracy: {valid_dense_results[1]*100:0.2f}')

# ## LSTM

# Define parameter
n_lstm = 128
drop_lstm = 0.25
model1 = Sequential()
model1.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model1.add(SpatialDropout1D(drop_lstm))
model1.add(LSTM(n_lstm, return_sequences=False))
model.add(Dense(256, activation='relu'))
model1.add(Dropout(0.5))
model1.add(Dense(1, activation='sigmoid'))

model1.compile(loss = 'binary_crossentropy',
               optimizer = 'adam',
               metrics = ['accuracy'])

num_epochs = 30
early_stop = EarlyStopping(monitor='val_loss', patience=2)
history = model1.fit(training_padded,
                     y_train,
                     epochs=num_epochs, 
                     validation_data=(testing_padded, y_test),
                     callbacks =[early_stop],
                     verbose=2)

# ## BI-LSTM

model2 = Sequential()
model2.add(Embedding(vocab_size,
                     embedding_dim,
                     input_length = max_len))
model2.add(Bidirectional(LSTM(n_lstm,
                              return_sequences = False)))
model2.add(Dropout(drop_lstm))
model2.add(Dense(1, activation='sigmoid'))

model2.compile(loss = 'binary_crossentropy',
               optimizer = 'adam',
               metrics=['accuracy'])

num_epochs = 30
early_stop = EarlyStopping(monitor = 'val_loss',
                           patience = 2)
history = model2.fit(training_padded,
                     y_train,
                     epochs = num_epochs,
                     validation_data = (testing_padded, y_test),
                     callbacks = [early_stop],
                     verbose = 2)

# ## GRU

model3 = Sequential()
model3.add(Embedding(vocab_size,
                     embedding_dim,
                     input_length = max_len))
model3.add(SpatialDropout1D(0.2))
model3.add(GRU(128, return_sequences = False))
model3.add(Dropout(drop_value))
model3.add(Dense(1, activation = 'sigmoid'))

model3.compile(loss = 'binary_crossentropy',
                       optimizer = 'adam',
                       metrics=['accuracy'])

num_epochs = 30
early_stop = EarlyStopping(monitor='val_loss', patience=2)
history = model3.fit(training_padded,
                     y_train,
                     epochs=num_epochs, 
                     validation_data=(testing_padded, y_test),
                     callbacks =[early_stop],
                     verbose=2)

# Comparing the four different models
# print(f"Dense model loss and accuracy: {model.evaluate(testing_padded, y_test)} " )
# print(f"LSTM model loss and accuracy: {model1.evaluate(testing_padded, y_test)} " )
# print(f"Bi-LSTM model loss and accuracy: {model2.evaluate(testing_padded, y_test)} " )
# print(f"GRU model loss and accuracy: {model3.evaluate(testing_padded, y_test)}")

df_test

predict_msg = [df_test["input_text"][0], df_test["input_text"][8]]
def predict(predict_msg):
    padded = regular_encode([df_test["input_text"][0]], tokenizer, maxlen=512)
    
    return(model1.predict(padded))

predict(predict_msg)