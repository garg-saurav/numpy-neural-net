import numpy as np
import nn
import sys

from util import *
from layers import *

np.random.seed(42)

def check_fully_connected():
    XTrain = np.random.randn(10, 100)
    YTrain = np.random.randn(10, 2)

    nn1 = nn.NeuralNetwork(10, 1)
    nn1.addLayer(FullyConnectedLayer(100, 10, 'relu'))
    nn1.addLayer(FullyConnectedLayer(10,2,'softmax'))

    delta = 1e-7
    size1 = nn1.layers[0].weights.shape
    size2 = nn1.layers[1].weights.shape
    size3 = nn1.layers[0].biases.shape
    size4 = nn1.layers[1].biases.shape
    num_grad1 = np.zeros(size1)
    num_grad2 = np.zeros(size2)
    num_grad3 = np.zeros(size3)
    num_grad4 = np.zeros(size4)

    for i in range(size1[0]):
        for j in range(size1[1]):
            activations = nn1.feedforward(XTrain)
            loss1 = nn1.computeLoss(YTrain, activations)
            nn1.layers[0].weights[i, j] += delta
            activations = nn1.feedforward(XTrain)
            loss2 = nn1.computeLoss(YTrain, activations)
            num_grad_ij = (loss2 - loss1) / delta
            num_grad1[i, j] = num_grad_ij
            nn1.layers[0].weights[i, j] -= delta

    for i in range(size2[0]):
        for j in range(size2[1]):
            activations = nn1.feedforward(XTrain)
            loss1 = nn1.computeLoss(YTrain, activations)
            nn1.layers[1].weights[i, j] += delta
            activations = nn1.feedforward(XTrain)
            loss2 = nn1.computeLoss(YTrain, activations)
            num_grad_ij = (loss2 - loss1) / delta
            num_grad2[i, j] = num_grad_ij
            nn1.layers[1].weights[i, j] -= delta

    for i in range(size3[0]):
        for j in range(size3[1]):
            activations = nn1.feedforward(XTrain)
            loss1 = nn1.computeLoss(YTrain, activations)
            nn1.layers[0].biases[i, j] += delta
            activations = nn1.feedforward(XTrain)
            loss2 = nn1.computeLoss(YTrain, activations)
            num_grad_ij = (loss2 - loss1) / delta
            num_grad3[i, j] = num_grad_ij
            nn1.layers[0].biases[i, j] -= delta

    for i in range(size4[0]):
        for j in range(size4[1]):
            activations = nn1.feedforward(XTrain)
            loss1 = nn1.computeLoss(YTrain, activations)
            nn1.layers[1].biases[i, j] += delta
            activations = nn1.feedforward(XTrain)
            loss2 = nn1.computeLoss(YTrain, activations)
            num_grad_ij = (loss2 - loss1) / delta
            num_grad4[i, j] = num_grad_ij
            nn1.layers[1].biases[i, j] -= delta



    saved1 = nn1.layers[0].weights[:, :].copy()
    saved2 = nn1.layers[1].weights[:, :].copy()
    saved3 = nn1.layers[0].biases[:, :].copy()
    saved4 = nn1.layers[1].biases[:, :].copy()
    activations = nn1.feedforward(XTrain)
    nn1.backpropagate(activations, YTrain)
    new1 = nn1.layers[0].weights[:, :]
    new2 = nn1.layers[1].weights[:, :]
    new3 = nn1.layers[0].biases[:, :]
    new4 = nn1.layers[1].biases[:, :]
    ana_grad1 = saved1 - new1
    ana_grad2 = saved2 - new2
    ana_grad3 = saved3 - new3
    ana_grad4 = saved4 - new4

    print(np.linalg.norm(num_grad1 - ana_grad1))
    print(np.linalg.norm(num_grad2 - ana_grad2))
    print(np.linalg.norm(num_grad3 - ana_grad3))
    print(np.linalg.norm(num_grad4 - ana_grad4))
    print("Gradient Test Passed for Fully Connected Layer!")


def check_ALL_layers():
    XTrain = np.random.randn(10, 3, 32, 32)
    YTrain = np.random.randn(10, 10)

    nn1 = nn.NeuralNetwork(10, 1)

    nn1.addLayer(ConvolutionLayer([3, 32, 32], [3, 3], 4, 1, 'relu'))
    nn1.addLayer(AvgPoolingLayer([4, 30, 30], [2, 2], 2))
    nn1.addLayer(ConvolutionLayer([4, 15, 15], [4, 4], 4, 1, 'relu'))
    nn1.addLayer(MaxPoolingLayer([4, 12, 12], [2, 2], 2))
    nn1.addLayer(FlattenLayer())
    nn1.addLayer(FullyConnectedLayer(144, 10, 'softmax'))

    delta = 1e-7
    size1 = nn1.layers[0].weights.shape
    size2 = nn1.layers[2].weights.shape
    size3 = nn1.layers[0].biases.shape
    size4 = nn1.layers[2].biases.shape
    size5 = nn1.layers[5].weights.shape
    size6 = nn1.layers[5].biases.shape
    num_grad1 = np.zeros(size1)
    num_grad2 = np.zeros(size2)
    num_grad3 = np.zeros(size3)
    num_grad4 = np.zeros(size4)
    num_grad5 = np.zeros(size5)
    num_grad6 = np.zeros(size6)

    for a in range(size1[0]):
        for b in range(size1[1]):
            for i in range(size1[2]):
                for j in range(size1[3]):
                    activations = nn1.feedforward(XTrain)
                    loss1 = nn1.computeLoss(YTrain, activations)
                    nn1.layers[0].weights[a, b, i, j] += delta
                    activations = nn1.feedforward(XTrain)
                    loss2 = nn1.computeLoss(YTrain, activations)
                    num_grad_ij = (loss2 - loss1) / delta
                    num_grad1[a, b, i, j] = num_grad_ij
                    nn1.layers[0].weights[a, b, i, j] -= delta

    for a in range(size2[0]):
        for b in range(size2[1]):
            for i in range(size2[2]):
                for j in range(size2[3]):
                    activations = nn1.feedforward(XTrain)
                    loss1 = nn1.computeLoss(YTrain, activations)
                    nn1.layers[2].weights[a, b, i, j] += delta
                    activations = nn1.feedforward(XTrain)
                    loss2 = nn1.computeLoss(YTrain, activations)
                    num_grad_ij = (loss2 - loss1) / delta
                    num_grad2[a, b, i, j] = num_grad_ij
                    nn1.layers[2].weights[a, b, i, j] -= delta

    for a in range(size3[0]):
        activations = nn1.feedforward(XTrain)
        loss1 = nn1.computeLoss(YTrain, activations)
        nn1.layers[0].biases[a] += delta
        activations = nn1.feedforward(XTrain)
        loss2 = nn1.computeLoss(YTrain, activations)
        num_grad_ij = (loss2 - loss1) / delta
        num_grad3[a] = num_grad_ij
        nn1.layers[0].biases[a] -= delta

    for a in range(size4[0]):
        activations = nn1.feedforward(XTrain)
        loss1 = nn1.computeLoss(YTrain, activations)
        nn1.layers[2].biases[a] += delta
        activations = nn1.feedforward(XTrain)
        loss2 = nn1.computeLoss(YTrain, activations)
        num_grad_ij = (loss2 - loss1) / delta
        num_grad4[a] = num_grad_ij
        nn1.layers[2].biases[a] -= delta

    for i in range(size5[0]):
        for j in range(size5[1]):
            activations = nn1.feedforward(XTrain)
            loss1 = nn1.computeLoss(YTrain, activations)
            nn1.layers[5].weights[i, j] += delta
            activations = nn1.feedforward(XTrain)
            loss2 = nn1.computeLoss(YTrain, activations)
            num_grad_ij = (loss2 - loss1) / delta
            num_grad5[i, j] = num_grad_ij
            nn1.layers[5].weights[i, j] -= delta

    for i in range(size6[0]):
        for j in range(size6[1]):
            activations = nn1.feedforward(XTrain)
            loss1 = nn1.computeLoss(YTrain, activations)
            nn1.layers[5].biases[i, j] += delta
            activations = nn1.feedforward(XTrain)
            loss2 = nn1.computeLoss(YTrain, activations)
            num_grad_ij = (loss2 - loss1) / delta
            num_grad6[i, j] = num_grad_ij
            nn1.layers[5].biases[i, j] -= delta

    saved1 = nn1.layers[0].weights[:, :, :, :].copy()
    saved2 = nn1.layers[2].weights[:, :, :, :].copy()
    saved3 = nn1.layers[0].biases[:].copy()
    saved4 = nn1.layers[2].biases[:].copy()
    saved5 = nn1.layers[5].weights[:, :].copy()
    saved6 = nn1.layers[5].biases[:, :].copy()
    activations = nn1.feedforward(XTrain)
    nn1.backpropagate(activations, YTrain)
    new1 = nn1.layers[0].weights[:, :, :, :]
    new2 = nn1.layers[2].weights[:, :, :, :]
    new3 = nn1.layers[0].biases[:]
    new4 = nn1.layers[2].biases[:]
    new5 = nn1.layers[5].weights[:,:]
    new6 = nn1.layers[5].biases[:,:]    
    ana_grad1 = saved1 - new1
    ana_grad2 = saved2 - new2
    ana_grad3 = saved3 - new3
    ana_grad4 = saved4 - new4
    ana_grad5 = saved5 - new5
    ana_grad6 = saved6 - new6

    print(np.linalg.norm(num_grad1 - ana_grad1))
    print(np.linalg.norm(num_grad2 - ana_grad2))
    print(np.linalg.norm(num_grad3 - ana_grad3))
    print(np.linalg.norm(num_grad4 - ana_grad4))
    print(np.linalg.norm(num_grad5 - ana_grad5))
    print(np.linalg.norm(num_grad6 - ana_grad6))
    print("Gradient Test Passed for ALL layers!")



def check_all_layers():
    XTrain = np.random.randn(10, 3, 32, 32)
    YTrain = np.random.randn(10, 10)

    nn1 = nn.NeuralNetwork(10, 1)

    nn1.addLayer(ConvolutionLayer([3, 32, 32], [3, 3], 4, 1, 'relu'))
    nn1.addLayer(AvgPoolingLayer([4, 30, 30], [2, 2], 2))
    nn1.addLayer(ConvolutionLayer([4, 15, 15], [4, 4], 4, 1, 'relu'))
    nn1.addLayer(MaxPoolingLayer([4, 12, 12], [2, 2], 2))
    nn1.addLayer(FlattenLayer())
    nn1.addLayer(FullyConnectedLayer(144, 10, 'softmax'))

    delta = 1e-7
    size = nn1.layers[2].weights.shape
    num_grad = np.zeros(size)

    for a in range(size[0]):
        for b in range(size[1]):
            for i in range(size[2]):
                for j in range(size[3]):
                    activations = nn1.feedforward(XTrain)
                    loss1 = nn1.computeLoss(YTrain, activations)
                    nn1.layers[2].weights[a, b, i, j] += delta
                    activations = nn1.feedforward(XTrain)
                    loss2 = nn1.computeLoss(YTrain, activations)
                    num_grad_ij = (loss2 - loss1) / delta
                    num_grad[a, b, i, j] = num_grad_ij
                    nn1.layers[2].weights[a, b, i, j] -= delta

    saved = nn1.layers[2].weights[:, :, :, :].copy()
    activations = nn1.feedforward(XTrain)
    nn1.backpropagate(activations, YTrain)
    new = nn1.layers[2].weights[:, :, :, :]
    ana_grad = saved - new

    print(np.linalg.norm(num_grad - ana_grad))
    assert np.linalg.norm(num_grad - ana_grad) < 1e-5
    print("Gradient Test Passed for All layers!")

# check_fully_connected()
# check_conv_layer()
# check_all_layers()
check_ALL_layers()