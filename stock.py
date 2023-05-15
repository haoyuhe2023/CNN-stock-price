# -*- coding = utf-8 -*-
# @Time: 05/05/2023 19:44
# @Author：HaoYu.HE
# @File:stock.py
# @software:PyCharm

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D
from keras.optimizers import SGD
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

df = pd.read_csv('AAPLpy.csv')
df.head()
# Extract the adjusted close prices and normalize them
x0 = df['Adj Close'].values
# plt.plot(x0[:100])

m = max(x0)
x0 = x0 / m
n = len(x0)
# Set the number of time steps and reshape the data
p = 20
x = np.array([x0[k:k + p] for k in range(n - p + 1)])

y = np.array(x0[p:])
X = x[:-1]
X = X[:, :, np.newaxis]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
# build the CNN model
model = Sequential()
model.add(Conv1D(50, 4, padding='same', activation='relu', input_shape=(p, 1)))
model.add(MaxPooling1D(2))
model.add(Flatten())  # Translate 2-D data into 1-dimensional data

model.add(Dense(20))
model.add(Dropout(0.2))  # in case of over-fit
model.add(Activation('relu'))
model.add(Dense(1))  # output layer
model.add(Activation('sigmoid'))
# Compile the model and train it on the training data
model.compile(loss='mse', optimizer=SGD(lr=0.2))
model.summary()
# Train the model, using 50 epochs
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Data visualization
y_predict = model.predict(X_test)
plt.plot(y_test[:100])  # plot y_test
plt.plot(y_predict[:100], 'r')  # plot y_predict
plt.show()