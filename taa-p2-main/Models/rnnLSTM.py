import numpy as np
import pandas as pd
import joblib
import nltk
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, roc_curve, auc
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

from nltk.corpus import stopwords

LOAD_MODEL = False

DATASET_PATH = '../Datasets/combined_data.csv'
TEXT_LABEL = 'text'
SPAM_LABEL = 'label'
MAX_NUM_WORDS = 20000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 100

nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('english')).union({"escapenumber"})

# Text dataset processing
def load_data(filename, test_size=0.3):
    df = pd.read_csv(filename)
    print("Loaded data shape: ", df.shape)
    return train_test_split(df[TEXT_LABEL], df[SPAM_LABEL], test_size=test_size)

# Tokenize and pad sequences
def preprocess_text(texts, max_num_words=MAX_NUM_WORDS, max_sequence_length=MAX_SEQUENCE_LENGTH):
    tokenizer = Tokenizer(num_words=max_num_words, lower=True, oov_token="<UNK>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    data = pad_sequences(sequences, maxlen=max_sequence_length)
    return data, tokenizer, word_index

# Create RNN model
def create_rnn_model(embedding_dim=EMBEDDING_DIM, max_sequence_length=MAX_SEQUENCE_LENGTH, max_num_words=MAX_NUM_WORDS, lstm_units=128, dropout_rate=0.2):
    model = Sequential()
    model.add(Embedding(input_dim=max_num_words, output_dim=embedding_dim))
    model.add(LSTM(units=lstm_units, dropout=dropout_rate, recurrent_dropout=dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.build(input_shape=(None, max_sequence_length))
    model.summary()
    return model

def perform_grid_search(X_train_seq, y_train):
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    model = KerasClassifier(model=create_rnn_model, verbose=1, callbacks=[early_stopping])

    param_grid = {
        'model__embedding_dim': [50, 100, 200],
        'model__lstm_units': [128, 256],
        'model__dropout_rate': [0.2, 0.5],
        'batch_size': [32, 64, 128],
        'epochs': [20]
    }


    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', n_jobs=1, cv=3, verbose=3)
    grid_result = grid.fit(X_train_seq, y_train, validation_split=0.1)

    return grid_result

if __name__ == "__main__":
    # Ensure TensorFlow uses GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
    else:
        print("No gpus found.")
    # Preprocess and split dataset
    X_train, X_test, y_train, y_test = load_data(DATASET_PATH, 0.3)
    X_train_seq, tokenizer, word_index = preprocess_text(X_train)
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=MAX_SEQUENCE_LENGTH)

    if not LOAD_MODEL:
        grid_result = perform_grid_search(X_train_seq, y_train)

        # Save the best model and tokenizer
        best_model = grid_result.best_estimator_.model_
        print(f"\n=== Best model's params:\n {grid_result.best_params_}\n")
        best_model.save('rnn_classifier_best.h5')
        joblib.dump(tokenizer, 'tokenizer.pkl')
    else:
        from tensorflow.keras.models import load_model
        best_model = load_model('rnn_classifier_best.h5')
        tokenizer = joblib.load('tokenizer.pkl')

    # Test model
    y_pred_proba = best_model.predict(X_test_seq)
    y_pred = (y_pred_proba > 0.5).astype("int32")
    print(f"Classification report: \n{classification_report(y_test, y_pred)}")

    # Plot ROC curve
    fpr, tpr, threshold = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
