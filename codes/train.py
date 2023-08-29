# -*- coding: utf-8 -*-
import numpy as np
import random
random.seed(1)
np.random.seed(17)
import os
import math
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef
import tensorflow as tf
from sklearn.model_selection import KFold
tf.set_random_seed(153)
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Bidirectional, LSTM, Dropout, Dense, Activation, Flatten,Input, Permute
from tensorflow.keras.layers import multiply
from tensorflow.keras.optimizers import Adam
import sys
import extract

TIME_STEPS_LIST = extract.seq_length

INPUT_SIZE = 256

# Determine max time steps
max_time_steps = max(TIME_STEPS_LIST)
max_columns=max_time_steps*INPUT_SIZE 

# ----------------------------------------- load labels and features
print('load labels……')
sys.stdout.flush()
f1 = open('./train_and_test/train.tsv', 'r')
data = f1.readlines()
f1.close()
labels = []
for ff in range(len(data)):
    labels.append(int(data[ff].strip().split('\t')[0]))
onehot_encoder = OneHotEncoder(sparse=False)
labels = np.array(labels).reshape(len(labels), 1)
labels = onehot_encoder.fit_transform(labels)
print('load features……')
sys.stdout.flush()
features = []
with open('./train_and_test/train_features.txt', 'r') as file:
    for line in file:
        line_data = line.strip().split(' ')

        # Fill with 0 or trim the row data
        if len(line_data) < max_columns:
            line_data += [0.0] * (max_columns - len(line_data))
        elif len(line_data) > max_columns:
            line_data = line_data[:max_columns]
        features.append(line_data)

# Convert the list to a NumPy array
features = np.array(features, dtype=float)
print('features:',features.shape)
features = features.reshape(-1,max_time_steps,INPUT_SIZE)
print('train load feature done!', features.shape)
sys.stdout.flush()

SINGLE_ATTENTION_VECTOR = False
def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = inputs
    #a = Permute((2, 1))(inputs)
    #a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(input_dim, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((1, 2), name='attention_vec')(a)

    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul

def attention_model(max_time_steps, INPUT_SIZE):
    inputs = Input(shape=(max_time_steps, INPUT_SIZE))
    # CNN layer
    x = Conv1D(filters=64,kernel_size=1, activation='relu')(inputs)
    x = Dropout(0.3)(x)

    x = Bidirectional(LSTM(units=64, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = attention_3d_block(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(2)(x)
    outputs = Activation('sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

model=attention_model(max_time_steps, INPUT_SIZE)
model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8),loss='binary_crossentropy', metrics=['accuracy'] )
# ------------------------------------------- training and testing
# Define the number of folds
n_folds = 10
accuracies = []
mccs = []
sensitivities = []
specificities = []
precisions = []
fscores = []
all_test_predictions = []
all_test_labels = []
# Perform k-fold cross-validation
kf = KFold(n_splits=n_folds, shuffle=True)
for train_indices, test_index in kf.split(features):
    train_indices = list(train_indices)
    train_features = features[train_indices]
    train_labels = labels[train_indices]
    test_features = features[test_index]
    test_labels = labels[test_index]

    # Train the model
    model.fit(train_features, train_labels, epochs=8, batch_size=16, verbose=0)
    predictions=model.predict(test_features)
    predictions= np.argmax(predictions, axis=1)
    test_labels = np.argmax(test_labels, axis=1)
    all_test_predictions.extend(predictions)
    all_test_labels.extend(test_labels)
    accuracy = accuracy_score(test_labels, predictions)
    mcc = matthews_corrcoef(test_labels, predictions)
    tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = precision_score(test_labels, predictions)
    fscore = f1_score(test_labels, predictions)
    accuracies.append(accuracy)
    mccs.append(mcc)
    sensitivities.append(sensitivity)
    specificities.append(specificity)
    precisions.append(precision)
    fscores.append(fscore)

average_accuracy = np.mean(accuracies)
average_mcc = np.mean(mccs)
average_sensitivity = np.mean(sensitivities)
average_specificity = np.mean(specificities)
average_precision = np.mean(precisions)
average_fscore = np.mean(fscores)
print("Average Accuracy:", average_accuracy)
print("Average MCC:", average_mcc)
print("Average Sensitivity:", average_sensitivity)
print("Average Specificity:", average_specificity)
print("Average Precision:", average_precision)
print("Average F1-Score:", average_fscore)
model.save('./iAMP-Attenpred_model.h5')
print("Model saved as 'iAMP-Attenpred_model.h5'")
roc_auc = roc_auc_score(all_test_labels, all_test_predictions)
print("ROC AUC Score:", roc_auc)
