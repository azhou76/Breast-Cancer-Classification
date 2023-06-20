#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
   This file contains the Logistic Regression classifier
'''
import random
import numpy as np


def softmax(x):
    '''
    Apply softmax to an array

    @params:
        x: the original array
    @return:
        an array with softmax applied elementwise.
    '''
    e = np.exp(x - np.max(x))
    return (e + 1e-6) / (np.sum(e) + 1e-6)

class LogisticRegression:
    '''
    Multiclass Logistic Regression that learns weights using 
    stochastic gradient descent.
    '''
    def __init__(self, n_features, n_classes, batch_size, conv_threshold):
        '''
        Initializes a LogisticRegression classifer.

        @attrs:
            n_features: the number of features in the classification problem
            n_classes: the number of classes in the classification problem
            weights: The weights of the Logistic Regression model
            alpha: The learning rate used in stochastic gradient descent
        '''
        self.n_classes = n_classes
        self.n_features = n_features
        self.weights = np.zeros((n_classes, n_features + 1))  # An extra row added for the bias
        self.alpha = 0.03
        self.batch_size = batch_size
        self.conv_threshold = conv_threshold

    def train(self, X, Y):
        '''
        Trains the model using stochastic gradient descent

        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
        @return:
            num_epochs: integer representing the number of epochs taken to reach convergence
        '''
        epoch = 0
        converge = False
        last_loss = float('inf')
        while not converge:
            epoch += 1
            # we create an indices list including each example number so we
            # can shuffle X and Y together
            indices = list(range(np.shape(X)[0]))
            np.random.shuffle(indices)
            for i in range(0, (np.shape(X)[0] // self.batch_size)):
                x_batch = X[indices[i * self.batch_size : (i + 1) * self.batch_size]]
                y_batch = Y[indices[i * self.batch_size : (i + 1) * self.batch_size]]
                loss_grad = np.zeros_like(self.weights)
                for x, y in zip(x_batch, y_batch):
                    for j in range(0, self.n_classes):
                        if y == j:
                            loss_grad[j] += (softmax(np.matmul(self.weights, x))[j] - 1) * x
                        elif y != j:
                            loss_grad[j] += softmax(np.matmul(self.weights, x))[j] * x
                self.weights -= ((self.alpha * loss_grad) / len(x_batch))
            if np.abs(self.loss(X, Y) - last_loss) < self.conv_threshold:
                converge = True
            last_loss = self.loss(X, Y)

        return epoch
        

    def loss(self, X, Y):
        '''
        Returns the total log loss on some dataset (X, Y), divided by the number of examples.
        @params:
            X: 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: 1D Numpy array containing the corresponding labels for each example
        @return:
            A float number which is the average loss of the model on the dataset
        '''
        cross_entropy_loss = 0
        for example, y in zip(X, Y):
            # within each example, iterate over every classification and add the 
            # negative log of the softmax probability at the y label index
            example_softmax = softmax(np.matmul(self.weights, example))
            for j in range(0, np.shape(example_softmax)[0]):
                if y == j:
                    cross_entropy_loss += -np.log(example_softmax[j])
        return cross_entropy_loss / np.shape(X)[0]

    def predict(self, X):
        '''
        Compute predictions based on the learned weigths and examples X

        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
        @return:
            A 1D Numpy array with one element for each row in X containing the predicted class.
        '''
        predicted_class = np.empty(np.shape(X)[0])
        i = 0
        for example in X:
            # get the index of the highest value (probability) in each example's softmax array 
            predicted_class[i] = np.argmax(softmax(np.matmul(self.weights, example)))
            i += 1
        return predicted_class

    def accuracy(self, X, Y):
        '''
        Outputs the accuracy of the trained model on a given testing dataset X and labels Y.

        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
        @return:
            a float number indicating accuracy (between 0 and 1)
        '''
        predictions = self.predict(X)
        accuracy = []
        for x, y in zip(predictions, Y):
            # when our model's prediction matches the example's actual label
            if x == y:
                accuracy.append(1)
            else:
                accuracy.append(0)
        return np.mean(accuracy)
