# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 00:25:33 2020

@author: HP
"""
def initialize_weights():
    W = []
    for i in range(no_of_hidden_layers + 1):
        l = no_of_nodes_in_layers[i]
        l_next = no_of_nodes_in_layers[i + 1]
        
        w = np.random.randn( l ,l_next )*np.sqrt( 1/l )
        w = np.clip(w,-1,1)        
        W.append(w)
    return W

def normalize(data):
    MAX = np.amax(data)
    MIN = np.amin(data)
    data = (2*data - (MAX + MIN) )/ (MAX - MIN)
    return data

def phi(x):
    return np.tanh(x)   

def is_multiple(x,y):
    if x/y==0 and x>0:
        return True
    else:
        return False

def phi_dash(v):
    return 1 - ( np.square( np.tanh(v) ) )

def shape(x):
    return x.reshape(len(x),1)

def phi_logistic(x,w):
    wt = np.transpose(w)
    siginp = np.matmul(wt,x)
    return  1/(1 + np.exp(-siginp)) 
    
########################################################################
import numpy as np
import random
import matplotlib.pyplot as plt

toy_training1 = np.random.uniform(low= np.pi, high=2*np.pi, size=250)
toy_training2 = np.random.uniform(low= 0, high=np.pi, size=250)
toy_training3 = np.random.uniform(low=-1*np.pi, high=0, size=250)
toy_training4 = np.random.uniform(low=-2*np.pi, high=-1*np.pi, size=250)


toy_training = np.concatenate((toy_training1,toy_training2,toy_training3,toy_training4))
toy_training.sort()
toy_training_y = np.sin(toy_training)
toy_test = np.random.uniform(low=-2*np.pi, high=2*np.pi, size=300)
toy_test.sort()
toy_test_y = np.sin(toy_test)

eta = 0.01
no_of_nodes_in_layers = [1,64,128,1]
no_of_hidden_layers = len(no_of_nodes_in_layers) - 2
training_size = len(toy_training)
test_size = len(toy_test)
W = initialize_weights()

e = np.zeros(training_size)
#toy_training = normalize(toy_training)*0.9
#toy_test = normalize(toy_test)*0.9
toy_training_y = toy_training_y
toy_test_y = toy_test_y
hl = no_of_hidden_layers + 3
v = np.zeros(hl).tolist()
y = np.zeros(hl).tolist()
v_pred = np.zeros(hl).tolist()
y_pred = np.zeros(hl).tolist()
delta = np.zeros(hl).tolist()
mini_batch_sizes = [1000]
Y_pred = []
W_check = []
bias = [0,0]
for b in range(2,hl):
    bias.append(random.uniform(-0.2, 0.2))
epochs = int(input('Enter the number of epochs:'))
for k in range(epochs):
    for i in range(int(training_size)):
        for layer in range(2,no_of_hidden_layers + 3):
            if layer==2:
                operand2 = np.array( [toy_training[i]] ) 
            else:
                operand2 = np.array(y[layer - 1])
            v[layer] = np.matmul(np.transpose(W[layer - 2]),operand2) + bias[layer]    
            y[layer] = phi(v[layer])
        e[i] =  y[layer] - toy_training_y[i]
        
            
    #        avg_e = sum(e)/len(e)
        
        delta[4] = e[i]*phi_dash(v[4])
        
        delta[3] = np.matmul(W[2],shape(delta[4]))*shape(phi_dash(v[3]))
        
        delta[2] = np.matmul(W[1],shape(delta[3]))*shape(phi_dash(v[2]))
        #print('Before',W[2])
        W[2] = W[2] - eta*np.matmul(shape(y[3]), np.transpose(shape(delta[4])))
        #print('Ater',W[2])
        W[1] = W[1] - eta*np.matmul(shape(y[2]), np.transpose(shape(delta[3])))
        
        W[0] = W[0] - eta*np.matmul(np.array([toy_training[i]]), np.transpose(shape(delta[2])))
    
        e = e*0
           
Y_pred = []
for j in range(test_size):     
    for layer in range(2,no_of_hidden_layers + 3):
         if layer==2:  
            v_pred[layer] = np.matmul(np.transpose(W[layer - 2]),np.array( [toy_test[j] ])) + bias[layer]
            y_pred[layer] = phi(v_pred[layer])
         else:
            v_pred[layer] = np.matmul(np.transpose(W[layer - 2]),np.array(y_pred[layer - 1])) + bias[layer]
            y_pred[layer] = phi(v_pred[layer])
    Y_pred.append(y_pred[layer])

plt.plot(toy_test,Y_pred)
plt.plot(toy_test,toy_test_y)    
    
        
