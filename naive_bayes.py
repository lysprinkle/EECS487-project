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
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report
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
    
def tokenize_tweet(text, tokenizer, max_length):
    tokens = tokenizer.tokenize(text)[:max_length - 2]
    return ["[CLS]"] + tokens + ["[SEP]"]

def pad_tokens(tokens, max_length):
    padding_length = max_length - len(tokens)
    return tokens + [0] * padding_length, [1] * len(tokens) + [0] * padding_length

def generate_attention_masks(padded_tokens):
    return [1 if token != 0 else 0 for token in padded_tokens]

class ModelTrainingHandler:
    
    def __init__(self, tokenizer, bert_layer_url):
        self.tokenizer = tokenizer
        self.bert_layer_url = bert_layer_url

    def _prepare_data(self, features):
        tokenized_features = [self.tokenizer.encode(text, add_special_tokens=True) for text in features]
        return tokenized_features

    def _build_model(self):
        bert_layer = tf.keras.layers.Lambda(lambda x: x, name="dummy_layer")
        input_layer = Input(shape=(None,), dtype=tf.int32, name="input_layer")
        output_layer = bert_layer(input_layer)
        output = Dense(2, activation='softmax')(output_layer)
        model = Model(inputs=input_layer, outputs=output)
        return model

    def train_model(self, features, labels, batch_size, learning_rate, num_epochs):
        processed_features = self._prepare_data(features)
        encoded_labels = to_categorical(labels, num_classes=2)

        X_train, X_test, y_train, y_test = train_test_split(processed_features, encoded_labels, test_size=0.2, random_state=42)

        model = self._build_model()
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
        
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)

        model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[lr_scheduler])

        y_pred = model.predict(X_test)
        y_pred_classes = y_pred.argmax(axis=1)
        y_test_classes = y_test.argmax(axis=1)
        test_accuracy = accuracy_score(y_test_classes, y_pred_classes)

        return model

    class ModelTrainingFacade:

    def __init__(self, tokenizer_module, bert_module):
        self.text_tokenizer = tokenizer_module
        self.bert_structure = bert_module


    def single_run_training(self, features, labels, size_batch, rate_learning, epochs_num):
        feature_train_prepped = self._prepare_text_data(features, self.text_tokenizer)
        label_train_encoded = to_categorical(labels, dtype="uint8")

        neural_model = self._construct_neural_network()

        neural_model.compile(optimizer=keras.optimizers.Adam(learning_rate=rate_learning), loss='categorical_crossentropy', metrics=['accuracy'])

        class_weighting = class_weight.compute_class_weight('balanced', np.unique(labels), labels)
        dict_class_weights = dict(enumerate(class_weighting))

        stop_early = EarlyStopping(monitor='val_loss', patience=5, mode='min', min_delta=0.001, restore_best_weights=True)

        training_history = neural_model.fit(x=feature_train_prepped, y=label_train_encoded, batch_size=size_batch, class_weight=dict_class_weights, epochs=epochs_num, callbacks=[stop_early])

        return neural_model

def get_early_stop(pat, delta)
    return keras.callbacks.EarlyStopping(monitor = 'loss', patience = 5, mode = 'min', min_delta = 0.01, restore_best_weights = True)

def create_neural_network(bert_component, sequence_length=50, num_classes=5):
    input_ids = Input(shape=(sequence_length,), dtype=tf.int32, name="input_ids")
    mask = Input(shape=(sequence_length,), dtype=tf.int32, name="mask")
    segment_ids = Input(shape=(sequence_length,), dtype=tf.int32, name="segment_ids")

    pooled_output, _ = bert_component([input_ids, mask, segment_ids])
    pooled_flat = pooled_output[:, 0, :] 

    layer_one = Dense(512, activation='relu')(pooled_flat)
    layer_one_dropout = Dropout(0.8)(layer_one)
    layer_two = Dense(128, activation='relu')(layer_one_dropout)
    layer_two_dropout = Dropout(0.8)(layer_two)

    final_output = Dense(num_classes, activation='softmax')(layer_two_dropout) 

    neural_net = Model(inputs=[input_ids, mask, segment_ids], outputs=final_output)
    neural_net.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return neural_net

def evaluate_network(neural_net, test_data, true_labels, label_encoder, data_frame):
    predictions = neural_net.predict(test_data)
    predicted_labels = [np.argmax(prediction) for prediction in predictions]
    
    decoded_predictions = label_encoder.inverse_transform(predicted_labels)
    true_decoded_labels = data_frame['TargetColumn'].values

    print(classification_report(true_decoded_labels, decoded_predictions, target_names=label_encoder.classes_))

def get_encoded_label(df, y)
    le = LabelEncoder()
    le.fit(df['Stance'])
    return le.transform(y)
