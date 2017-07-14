# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np

from util.activation_functions import Activation
from model.classifier import Classifier
from model.logistic_layer import LogisticLayer
from data.data_set import DataSet

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class LogisticRegression(Classifier):
    """
    A digit-7 recognizer based on logistic regression algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    trainingSet : list
    validationSet : list
    testSet : list
    weight : list
    learningRate : float
    epochs : positive int
    """

    def __init__(self, data, learningRate=0.01, epochs=50, hiddensize=50):

        self.learningRate = learningRate
        self.epochs = epochs
        self.trainingSet = data.trainingSet
        self.validationSet = data.validationSet
        self.testSet = data.testSet
        self.data=data
        self.layer=LogisticLayer(data.trainingSet.input.shape[1],hiddensize,learningRate)

        # Initialize the weight vector with small values
        self.weight = 0.01*np.random.randn(self.layer.size)

    def train(self, verbose=True):
        """Train the Logistic Regression.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """

        from util.loss_functions import DifferentError
        loss = DifferentError()

        learned = False
        iteration = 0

        while not learned:

            self.shuffle()

            totalError = 0

            for input, label in zip(self.trainingSet.input,
                                    self.trainingSet.label):
                # feedforward
                inputarray = input.reshape(1,len(input))
                layeroutput = self.layer.forward(inputarray)
                output = self.fire(layeroutput)
                # compute gradient of regression
                delta=label - output
                grad =delta * self.layer.output
                # backpropagation
                self.layer.computeDerivative(delta,self.weight)
                #update all weights
                self.updateWeights(grad)
                self.layer.updateWeights()

            # compute recognizing error, not BCE using validation data
            for input, label in zip(self.validationSet.input,
                                    self.validationSet.label):
                predictedLabel = self.classify(input)
                error = loss.calculateError(label, predictedLabel)
                totalError += error

            totalError = abs(totalError)

            iteration += 1

            if verbose:
                logging.info("Epoch: %i; Error: %i", iteration, totalError)


            if totalError == 0 or iteration >= self.epochs:
                # stop criteria is reached
                learned = True

    def classify(self, testInstance):
        """Classify a single instance.

        Parameters
        ----------
        testInstance : list of floats

        Returns
        -------
        bool :
            True if the testInstance is recognized as a 7, False otherwise.
        """
        # feedforward
        testInstance=testInstance.reshape(1,len(testInstance))
        output = self.fire(self.layer.forward(testInstance))
        return output > 0.5

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just se map for all of the test
        # set.
        return list(map(self.classify, test))

    def updateWeights(self, grad):
        self.weight += (self.learningRate*grad).reshape(self.weight.size)

    def fire(self, input):
        # input (n,1)
        return Activation.sigmoid(np.dot(self.weight,input))

    def shuffle(self):
        self.data.myshuffle()
        self.trainingSet=self.data.trainingSet
        self.validationSet=self.data.validationSet
        self.testSet=self.data.testSet
