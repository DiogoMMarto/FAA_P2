
from copy import deepcopy
import os
from typing import Callable
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, roc_curve, auc , confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

TEST_SPLIT = 0.2
VAL_SPLIT = 0.1

def gen(param_grid: dict[str, list]):
        param , vals = param_grid.popitem()
        if len(param_grid) == 0:
            return [{param: val} for val in vals]
        p = gen(param_grid)
        r = []
        for val in vals:
            _p = deepcopy(p)
            for d in _p:
                d[param] = val
            r += _p
        return r
        
def model_pass(param_grid: dict[str, list], make_model: Callable ,X_train , y_train , X_test , y_test , i = 0 , train_func = None ,load_dir = None ):
        param_grid['name'] = f"Model{i}"
        model = make_model(**param_grid)
        model.name = param_grid['name']
        if load_dir:
            # walk dir 
            model = tf.keras.models.load_model(f"{load_dir}/{model.name}.keras")
            print(f"Loaded model: {model.name}")
            history = []
        else:
            if train_func:
                model , history = train_func(model, X_train, y_train)
            else:
                model , history = train_model(model, X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test , history)
        return model , metrics
    
def grid_search(param_grid: dict[str, list], make_model, X_train , y_train , X_test , y_test , train_func = None , load_dir = None):
    r = []
    i = 0
    grid = gen(param_grid)
    print(f"Grid size: {len(grid)}")
    for params in grid:
        try:
            model , metrics = model_pass(params, make_model , X_train , y_train , X_test , y_test , i , train_func , load_dir)
            print(f"Model: {model.name}")
            print(f"Parameters: {params}")
            print(f"Metrics: {metrics}")
            r.append((model, params, metrics))
        except ValueError as e:
            pass
        i += 1
    return r

def train_model(model, X_train_seq, y_train, epochs=20):
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(X_train_seq, y_train, epochs=epochs, batch_size=128, validation_split=0.1, verbose=1, callbacks=[early_stopping])
    return model , history

def evaluate_model(model, X_test_seq, y_test,history):
    y_pred = model.predict(X_test_seq)
    y_pred = (y_pred > 0.5).astype(int)
    
    metrics = dict()
    metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)
    metrics['classification_report'] = classification_report(y_test, y_pred,output_dict=True)
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    metrics['roc_auc'] = roc_auc
    metrics['fpr'] = fpr
    metrics['tpr'] = tpr
    metrics['history'] = history
    
    return metrics

def tokenize_text(text, max_num_words , max_sequence_length):
    tokenizer = Tokenizer(num_words=max_num_words)
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    return padded_sequences , tokenizer