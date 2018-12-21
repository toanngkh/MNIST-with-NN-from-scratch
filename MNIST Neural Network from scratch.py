
# coding: utf-8

# In[15]:

import numpy as np
import scipy.special as ss
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')


# In[23]:

class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        # set weights random range (-0.5, + 0.5)
        self.wih = np.random.rand(self.hnodes, self.inodes)-0.5
        self.who = np.random.rand(self.onodes, self.hnodes)-0.5
        # set weights random but have normal distribution range (-0.5, + 0.5)
        self.nwih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.nwho = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        # sigmoid function (function expit() of module scipy.special)
        self.activation_function = lambda x: ss.expit(x)
        
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d arrays
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        
        # calculate signal into hidden layer
        hidden_inputs = np.dot(self.nwih, inputs)
        # calculate output of hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signal into output layer
        final_inputs = np.dot(self.nwho, hidden_outputs)
        # calculate output of output layer
        final_outputs = self.activation_function(final_inputs)
        
        # output layer errors
        output_errors = targets - final_outputs
        # hidden layer errors
        hidden_errors = np.dot(self.nwho.T, output_errors)
        
        # update weights for the links between hidden layer and output layer
        self.nwho += self.lr * np.dot((output_errors * final_outputs * (1 - final_outputs)), np.transpose(hidden_outputs))
        
        # update weights for the links between input layer and hidden layer
        self.nwih += self.lr * np.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)), np.transpose(inputs))
    
    def query(self, inputs_list):
        # convert input list to 2d array (transpose to become vector)
        inputs = np.array(inputs_list, ndmin=2).T
        # calculate signal into hidden layer X hidden = W input-hidden * I input
        hidden_inputs = np.dot(self.nwih, inputs)
        # calculate output of hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signal into output layer
        final_inputs = np.dot(self.nwho, hidden_outputs)
        # calculate output of output layer
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
        


# In[33]:

data_file = open("C:/Users/toan/Documents/Make your own neural network/mnist_train.csv", "r")
data_list = data_file.readlines()
data_file.close()


# In[62]:

input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)


# In[63]:

# train
epochs = 2
for e in range(epochs):
    for record in data_list:
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:]) / 255 * 0.99) + 0.01
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)


# In[40]:

test_file = open("C:/Users/toan/Documents/Make your own neural network/mnist_test.csv", "r")
test_list = test_file.readlines()
test_file.close()


# In[64]:

score_card = []


# In[65]:

# test
for record in test_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    #print(correct_label, "correct label")
    inputs = (np.asfarray(all_values[1:]) / 255 * 0.99) + 0.01
    outputs = n.query(inputs)
    label = np.argmax(outputs)
    #print(label, "network's answer")
    if (label == correct_label):
        score_card.append(1)
    else:
        score_card.append(0)


# In[66]:

scorecard_array = np.asarray(score_card)
print("performance:", scorecard_array.sum() / scorecard_array.size)


# In[ ]:



