import pandas as pd
df = pd.read_csv('/kaggle/input/sarcasm-dataset/Sarcasm Dataset.csv')
df.head()
import re

def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    #Urls removal
    text = re.sub(r'http\S+','', text)
    #Email removal
    text = re.sub(r'\S*@\S*\s?', '', text)
    text = text.lower()
    text = re.sub(r'\s+',' ', text)
    text = re.sub(r'#\S+','', text)
    return text
df['clean_tweet']=df['tweet'].apply(lambda x: preprocess_text(x))
df['clean_tweet']
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

df['sarcasm'] = encoder.fit_transform(df['sarcasm'])
df.head()

import numpy as np
from imblearn.over_sampling import RandomOverSampler

def balance_df(df,text,target):
    ros = RandomOverSampler()
    train_x, train_y = ros.fit_resample(np.array(df[text]).reshape(-1,1), np.array(df[target]).reshape(-1,1))
    new_df = pd.DataFrame(list(zip([x[0] for x in train_x],train_y)), columns = [text,target])
    return new_df

sarcasm_df = pd.DataFrame()
sarcasm_df = balance_df(df,'clean_tweet','sarcasm')
sarcasm_df['sarcasm'].value_counts()

from sklearn.model_selection import train_test_split
X = df['clean_tweet']
y = df['sarcasm']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer

model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',num_labels = 3)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased',num_labels = 3)

my_max = max(len(text) for text in sarcasm_df['clean_tweet'])
my_max
max_length = 143
train_encodings = tokenizer(X_train.tolist(), padding = True, truncation = True, max_length = max_length)
test_encodings = tokenizer(X_test.tolist(), padding = True, truncation = True, max_length = max_length)
train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), y_test))
BATCH_SIZE = 16
train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE,drop_remainder = True).prefetch(tf.data.AUTOTUNE)
optimizer = tf.keras.optimizers.Adam(learning_rate = 5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
model.compile(optimizer = optimizer, loss=loss, metrics=['accuracy'])

history = model.fit(
    train_dataset,
    epochs = 10,
    batch_size = 8,
    validation_data = test_dataset.batch(16)
)
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('train and val loss')
plt.show()
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np

y_probab = model.predict(test_dataset.batch(16))
y_pred = np.argmax(y_probab.logits, axis = 1)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation = 'nearest', cmap = plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Not Sarcastic', 'Sarcastic'])
plt.yticks(tick_marks, ['Not Sarcastic', 'Sarcastic'])

thresh = cm.max()/ 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha = 'center', va = 'center', color = 'black')

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()
report = classification_report(y_test, y_pred)
print(report)
model.save_pretrained("Wyrm_Crawl_bert")
loaded_model = TFDistilBertForSequenceClassification.from_pretrained('Wyrm_Crawl_bert')
def predictive(text):
    text = preprocess_text(text)
    inputs = tokenizer(text, padding = True, truncation = True, max_length = max_length, return_tensors = 'tf')
    logits = loaded_model(inputs).logits
    probab = tf.nn.softmax(logits, axis = 1).numpy()
    predicted_label = np.argmax(probab, axis = 1)
    return predicted_label
text = "wow,Steve harvey has a really cute face"
predicted_label = predictive(text)
print("Label: ", predicted_label)

if predicted_label[0] == 1:
    print("Sarcastic Statement")
else:
    print("Non Sarcastic Statement")
