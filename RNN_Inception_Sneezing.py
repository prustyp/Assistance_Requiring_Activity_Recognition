import json
import os
import sys
from contextlib import redirect_stdout
from datetime import datetime
from os import listdir
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
from keras import optimizers
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from project_utils import EnsurePath


RESULT_FOLDER='/home/pprusty05/Deep_learning/result'
SAVED_MODEL_FOLDER='/home/pprusty05/Deep_learning/saved_model'



list_of_folder_with_extracted_feature = []
## append the full path of the trimmed videos folder
list_of_folder_with_extracted_feature.append("/home/pprusty05/Deep_learning/data/Training/extracted_features_inception_2/sneezing")
list_of_folder_with_extracted_feature.append("/home/pprusty05/Deep_learning/data/Training/extracted_features_inception_2/others2")


X, y = [], []
index = 0
for folders in list_of_folder_with_extracted_feature:
    onlyfiles = [f for f in listdir(folders) if join(folders, f).endswith(".npy")]
    #print(onlyfiles[1])
    for files in onlyfiles:
        #files.shape
        path = os.path.join(folders, files)
        img_data = np.load(path)
        #print(img_data.shape)
        X.append(img_data)
        output_list = []
        for i in range(len(list_of_folder_with_extracted_feature)):
            if(i == index):
                output_list.append(1)
            else:
                output_list.append(0)
        y.append(output_list)

    index = index+1

X = np.array(X)
y = np.array(y)
print(X.shape)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1)


input_shape = (100, 2048)
model = Sequential()
model.add(LSTM(2048,#kernel_regularizer=l1(0.01), recurrent_regularizer=l1(0.01), bias_regularizer=l1(0.01),
               return_sequences=False,
               input_shape=input_shape,
              dropout=0.5))
model.add(BatchNormalization())
#model.add(Dense(512, activation='tanh'))
#model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
#optimizer = optimizers.sgd(lr=1e-5, decay=1e-6)
optimizer = optimizers.adam(lr=1e-5, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                   metrics=['accuracy'])

history=model.fit(
    X_train,
    y_train,
    batch_size=100,
    validation_data=(X_val, y_val),
    verbose=1,
    epochs=100
)



# Plot the Loss Curves
plt.figure(figsize=[8, 6])
plt.plot(history.history['loss'], 'r', linewidth=3.0)
plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Loss Curves', fontsize=16)

# Plot the Accuracy Curves
plt.figure(figsize=[8, 6])
plt.plot(history.history['accuracy'], 'r', linewidth=3.0)
plt.plot(history.history['val_accuracy'], 'b', linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Accuracy Curves', fontsize=16)



##SAVE EVERYTHING
RESULT_FOLDER_TS = os.path.join(RESULT_FOLDER, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
EnsurePath(RESULT_FOLDER_TS)

model_folder_path = os.path.join(RESULT_FOLDER_TS, 'model_dir')
EnsurePath(model_folder_path)
model_json = model.to_json()

json_path = os.path.join(model_folder_path, "model_num.json")
with open(json_path, "w") as json_file:
    json_file.write(model_json)

weight_path = os.path.join(model_folder_path, "model_weight.h5")
model.save_weights(weight_path)
entire_model_path = os.path.join(model_folder_path, "entire_model.h5")
model.save(entire_model_path)

history_file_path = os.path.join(RESULT_FOLDER_TS, 'value_history.json')
with open(history_file_path, 'w') as f:
    json.dump(str(history.history), f)

[test_loss, test_acc] = model.evaluate(X_test, y_test)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))
y_pred = model.predict_classes(X_test)
y_pred = to_categorical(y_pred)
cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

print("confusion_matrix")
print(cm)
print("Accuracy", cm.diagonal() / cm.sum(axis=1))
print("Precision", precision_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average=None))
print("Recall", recall_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average=None))
print("f1_score", f1_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average=None))

test_eval_dict = {}
test_eval_dict['accuracy'] = (cm.diagonal() / cm.sum(axis=1)).tolist()
test_eval_dict['precision'] = (precision_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average=None)).tolist()
test_eval_dict['recall'] = (recall_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average=None)).tolist()
test_eval_dict['f1_score'] = (f1_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average=None)).tolist()
test_eval_dict['description_of_model'] = 'Add some description about the model here'

test_eval_file_path = os.path.join(RESULT_FOLDER_TS, 'test_eval.json')

with open(test_eval_file_path, 'w') as fp:
    json.dump(test_eval_dict, fp)

model_summary_file_path = os.path.join(RESULT_FOLDER_TS, 'model_summary.txt')
with open(model_summary_file_path, 'w') as f:
    with redirect_stdout(f):
        model.summary()
