import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import math 
import sys
from sklearn.model_selection import train_test_split
import re

def show_plots(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

def gen_labels(size):
    labels = [0]*size
    for i in range(len(labels)):
            labels[i] = math.floor(i/200)
            labels[i] = labels[i] - (math.floor(i/2000)*10)
    return labels

def gen_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(16,15,1)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(10))
    return model

def show_img(img):
    im = np.resize(img, (16,15))
    plt.imshow(im, cmap='Greys', interpolation='nearest')
    plt.show()

def read_data(path):
    
    with open(path) as f:
        i = 0
        data = np.zeros((8000,240))
        for line in f:
            digits = re.findall("\d+\.\d+", line)
            for j in range(len(digits)):
                data[i][j] = pd.to_numeric(digits[j])
            i = i + 1 
    data = data.reshape(8000,16,15)
    return(data)

def run_model(model,X_train, Y_train, X_test, Y_test):
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    history = model.fit(X_train, Y_train, epochs=25, 
                    validation_data=(X_test, Y_test))
    show_plots(history)

def main():
    os.chdir(os.path.dirname(sys.argv[0]))
    
    data = read_data('final.txt')
    labels = gen_labels(len(data))
    X_train, X_test,y_train, y_test = train_test_split(data,labels,
                                   random_state=94, 
                                   test_size=0.25, 
                                   shuffle=True)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    model = gen_model()
    run_model(model, X_train, y_train, X_test, y_test)
    return

if __name__ == "__main__":
    main()