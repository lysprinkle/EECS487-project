import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from sklearn.model_selection import StratifiedKFold
from transformers import BertTokenizer
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import gc

should_add_k_for_unseen = True


def get_basic_stats(df):
    avg_len = 0
    std_len = 0
    num_articles = {0: 0, 1: 0, 2: 0}
    
    text_list = df['text'].tolist()
    stance_list = df['label'].tolist()
    
    length_list = []
    
    for i in range(len(text_list)):
        length_list.append(len(word_tokenize(text_list[i])))
        if stance_list[i] == 0:
            num_articles[0] += 1
        elif stance_list[i] == 1:
            num_articles[1] += 1
        else:
            num_articles[2] += 1
    
    avg_len = sum(length_list) / len(length_list)
    std_len = np.std(length_list)
        

    ###################################################################

    print(f"Average number of tokens per headline: {avg_len}")
    print(f"Standard deviation: {std_len}")
    print(f"Number of neutral/against/agree headlines: {num_articles}")


class NaiveBayes:
    """Naive Bayes classifier."""

    def __init__(self):
        self.ngram_count = []
        self.total_count = []
        self.category_prob = []
        self.vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.8)
        self.vocab = []
        self.word_freq_neutral = {}
        self.word_freq_against = {}
        self.word_freq_agree = {}
    
    def fit(self, data):
        text_list = []
        for text in data['text'].tolist():
            text_list.append(text.lower())
        
        stance_list = data['label'].tolist()
        
        text_neutral_list = []
        text_against_list = []
        text_agree_list = []
        
        for i in range(len(stance_list)):
            if stance_list[i] == 0:
                text_neutral_list.append(text_list[i])
            elif stance_list[i] == 1:
                text_against_list.append(text_list[i])
            else:
                text_agree_list.append(text_list[i])
        
        self.category_prob = [sum(1 for stance in stance_list if stance == 0) / len(stance_list), 
                              sum(1 for stance in stance_list if stance == 1) / len(stance_list), 
                              sum(1 for stance in stance_list if stance == 2) / len(stance_list)]
        count_array = self.vectorizer.fit_transform(text_list).toarray()
        self.vocab = self.vectorizer.get_feature_names_out()
        V = len(self.vocab)

        count_array_neutral = [0] * V
        count_array_against = [0] * V
        count_array_agree = [0] * V
        
        count_vocab_neutral = 0
        count_vocab_against = 0
        count_vocab_agree = 0
        
        for i in range(len(count_array)):
            if stance_list[i] == 0:
                count_array_neutral += count_array[i]
                count_vocab_neutral += sum(count_array[i])
            elif stance_list[i] == 1:
                count_array_against += count_array[i]
                count_vocab_against += sum(count_array[i])
            else:
                count_array_agree += count_array[i]
                count_vocab_agree += sum(count_array[i])
        
        self.ngram_count.append(count_array_neutral)
        self.ngram_count.append(count_array_against)
        self.ngram_count.append(count_array_agree)
        self.total_count.append(count_vocab_neutral)
        self.total_count.append(count_vocab_against)
        self.total_count.append(count_vocab_agree)
        
        self.word_freq_neutral = {self.vocab[i]: self.ngram_count[0][i] for i in range(len(self.vocab))}
        self.word_freq_against = {self.vocab[i]: self.ngram_count[1][i] for i in range(len(self.vocab))}
        self.word_freq_agree = {self.vocab[i]: self.ngram_count[2][i] for i in range(len(self.vocab))}
    
    def calculate_prob(self, docs, c_i):
        prob = []

        if c_i == 0:
            for doc in docs:
                doc = word_tokenize(doc.lower())
                length = len(doc)
                for i in range(length-1):
                    doc.append(doc[i] + " " + doc[i+1])
                cur_prob = np.log(self.category_prob[0])
                for cur_vocab in doc:
                    if cur_vocab in self.word_freq_neutral:
                        cur_prob += np.log((self.word_freq_neutral[cur_vocab] + 1) / (self.total_count[0] + len(self.vocab)))

                prob.append(cur_prob)

        elif c_i == 1:
            for doc in docs:
                doc = word_tokenize(doc.lower())
                length = len(doc)
                for i in range(length-1):
                    doc.append(doc[i] + " " + doc[i+1])
                cur_prob = np.log(self.category_prob[1])
                for cur_vocab in doc:
                    if cur_vocab in self.word_freq_against:
                        cur_prob += np.log((self.word_freq_against[cur_vocab] + 1) / (self.total_count[1] + len(self.vocab)))

                prob.append(cur_prob)
        
        else:
            for doc in docs:
                doc = word_tokenize(doc.lower())
                length = len(doc)
                for i in range(length-1):
                    doc.append(doc[i] + " " + doc[i+1])
                cur_prob = np.log(self.category_prob[2])
                for cur_vocab in doc:
                    if cur_vocab in self.word_freq_agree:
                        cur_prob += np.log((self.word_freq_agree[cur_vocab] + 1) / (self.total_count[2] + len(self.vocab)))

                prob.append(cur_prob)

        return prob

    def predict(self, docs):
        prediction = [None] * len(docs)

        prob1 = self.calculate_prob(docs, 0)
        prob2 = self.calculate_prob(docs, 1)
        prob3 = self.calculate_prob(docs, 2)
        for i in range(len(docs)):
            if prob1[i] > prob2[i] and prob1[i] > prob3[i]:
                prediction[i] = 0
            elif prob2[i] > prob1[i] and prob2[i] > prob3[i]:
                prediction[i] = 1
            else:
                prediction[i] = 2

        return prediction


def evaluate(predictions, labels):
    accuracy, mac_f1, mic_f1 = None, None, None

    accuracy = accuracy_score(labels, predictions)

    mac_f1 = f1_score(labels, predictions, average='macro')

    mic_f1 = f1_score(labels, predictions, average='micro')

    return accuracy, mac_f1, mic_f1
    
def bert_tokenizer_and_padding(X, tokenizer, max_seq_length):
    X_encoded = tokenizer(X, max_seq_length=max_seq_length, truncation=True, padding='max_length', return_tensors='tf')
    return X_encoded

def build_model(bert_layer, num_classes):
    input_word_ids = Input(shape=(max_seq_length,), dtype=tf.int32, name="input_word_ids")
    inputs = [input_word_ids]
    pooled_output, sequence_output = bert_layer(inputs)
    output = Dense(num_classes, activation='softmax')(pooled_output)
    model = Model(inputs=inputs, outputs=output)
    return model

def compile_model(model):
    optimizer = Adam(learning_rate=2e-5)
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

import matplotlib.pyplot as plt

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    
def compute_class_weights(y):
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y), y)
    class_weight_dict = dict(enumerate(class_weights))
    return class_weight_dict

def preprocess_for_bert(texts, max_length):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    tokenized_texts = [tokenizer.encode(text, add_special_tokens=True, max_length=max_length, truncation=True) for text in texts]

    padded_texts = np.array([text + [0] * (max_length - len(text)) for text in tokenized_texts])

    attention_masks = np.where(padded_texts != 0, 1, 0)

    segment_ids = np.zeros_like(padded_texts)

    return padded_texts, attention_masks, segment_ids


def fit_label_encoder(train_df):
    le = LabelEncoder()
    le.fit(train_df['Stance'])
    return le


def encode_labels(y, le):
    y = le.transform(y)
    return y

def get_model(path):
    return keras.models.load_model(path, custom_objects={'KerasLayer':hub.KerasLayer})
