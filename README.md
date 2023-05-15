# CNN-stock-price
Using a simple 3-layer CNN to train stock price
This code is a Python script that uses Keras and TensorFlow to build and train a convolutional neural network (CNN) for predicting stock prices. The data is extracted from a CSV file that contains historical stock price information for the AAPL (Apple Inc.) stock.

The codes reads in the stock price data from the CSV file using pandas. The adjusted close prices are extracted from the data and normalized by dividing them by the maximum value in the data.

Next, the script sets the number of time steps to 20 and reshapes the data into a 3D array. It then splits the data into training and testing sets using the train_test_split function from scikit-learn.

The script then builds a CNN model using Keras. The model consists of a convolutional layer with 50 filters, a kernel size of 4, and a rectified linear activation function. This is followed by a max-pooling layer with a pool size of 2, a flatten layer, a dense layer with 20 units and a rectified linear activation function, a dropout layer to prevent overfitting, and finally a dense output layer with a sigmoid activation function. The model is compiled using mean squared error as the loss function and stochastic gradient descent as the optimizer.

The model is trained using the training data for 50 epochs with a batch size of 32. After training, the script uses the model to make predictions on the testing data and plots the actual prices versus the predicted prices using matplotlib.

Overall, this script is an example of how to use Keras and TensorFlow to build a CNN model for predicting stock prices. 
