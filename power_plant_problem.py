# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 20:19:53 2020

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
    if len(x.shape)!=1:
        return x
    return x.reshape(len(x),1)

def phi_logistic(x,w):
    wt = np.transpose(w)
    siginp = np.matmul(wt,x)
    return  1/(1 + np.exp(-siginp)) 

def relu(x):
   return np.maximum(0,x)

def denormalize(data,y_max,y_min):
    return (data*(y_max - y_min) + y_max + y_min)/2
    

def MAPE(y,d):
    return abs((y-d)/(d))

def find_r2(f,y):
    avg_y = np.sum(y)/test_size
    nr = np.sum(np.square(f-y))
    dr = np.sum(np.square(y - avg_y))
    return 1 - nr/dr

    
########################################################################
import numpy as np
import random
import matplotlib.pyplot as plt

import xlrd
 
loc = ("Folds5x2_pp.xlsx")
 
wb = xlrd.open_workbook(loc)
sheet = wb.sheet_by_index(4)

training = []
training_y = []
validation = []
validation_y = []
test = []
test_y = []
total_samples = sheet.nrows
training_size = int(0.72 * sheet.nrows)
validation_size = int(0.18* sheet.nrows) 
for i in range(1,training_size):
    training.append(np.array([sheet.cell_value(i, 0),sheet.cell_value(i, 1),sheet.cell_value(i, 2),sheet.cell_value(i, 3)]))
    training_y.append(np.array([sheet.cell_value(i, 4)]))
training = np.array(training)
training_y = np.array(training_y)

for i in range(training_size,training_size + validation_size):
    validation.append(np.array([sheet.cell_value(i, 0),sheet.cell_value(i, 1),sheet.cell_value(i, 2),sheet.cell_value(i, 3)]))
    validation_y.append(np.array([sheet.cell_value(i, 4)]))
validation = np.array(validation)
validation_y = np.array(validation_y)
validation_y_max = np.amax(validation_y)
validation_y_min = np.amin(validation_y)
#training.sort()
for i in range(training_size + validation_size,total_samples):
    test.append(np.array([sheet.cell_value(i, 0),sheet.cell_value(i, 1),sheet.cell_value(i, 2),sheet.cell_value(i, 3)]))
    test_y.append(np.array([sheet.cell_value(i, 4)]))
test = np.array(test)
test_y = np.array(test_y)
test_y_max = np.amax(test_y)
test_y_min = np.amin(test_y)

eta = float(input('Enter the learning rate(typically of the order 10^-4 to 10^-2):'))
no_of_hidden_layers = 2
print('For the 4 layer ANN,')

no_of_nodes_in_layers = []
for i in range(no_of_hidden_layers + 2):
    print('For layer',i+1)
    nodes = int(input('Enter the number of nodes for:'))
    no_of_nodes_in_layers.append(nodes)
#no_of_hidden_layers = len(no_of_nodes_in_layers) - 2
activation_fuction = int(input('Enter,\n1 for tanh activation function \n2 for logistic activation function \n3 for relu activation function '))
test_size = len(test)
W = initialize_weights()

e = np.zeros(training_size).tolist()
training = normalize(training)*0.9
training_y = normalize(training_y)
test = normalize(test)*0.9
test_y = normalize(test_y)
training_y = training_y
test_y = test_y
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
    bias.append(random.uniform(-1, 1))
    
#for iterations in range(0,100,10):
ERROR = []
V_ERROR = []
iterations = []
for k in range(50):
    for i in range(training_size-1):
        print(k,i)
        for layer in range(2,no_of_hidden_layers + 3):
            if layer==2:
                operand2 = np.array( [training[i]] ).T 
            else:
                operand2 = np.array(y[layer - 1])
            v[layer] = np.matmul(np.transpose(W[layer - 2]),operand2) + bias[layer]    
            y[layer] = phi(v[layer])
        e[i] =  y[layer] - training_y[i]

        
            
        
        delta[4] = e[i]*phi_dash(v[4])
        
        delta[3] = np.matmul(W[2],shape(delta[4]))*shape(phi_dash(v[3]))
        
        delta[2] = np.matmul(W[1],shape(delta[3]))*shape(phi_dash(v[2]))
        #print('Before',W[2])
        W[2] = W[2] - eta*np.matmul(shape(y[3]), np.transpose(shape(delta[4])))
        #print('Ater',W[2])
        W[1] = W[1] - eta*np.matmul(shape(y[2]), np.transpose(shape(delta[3])))
        
        W[0] = W[0] - eta*np.matmul(shape(np.array(training[i])), np.transpose(shape(delta[2])))
    
        e = np.zeros(training_size).tolist()
           
    Y_pred = []
    print('Trained')
    for j in range(test_size):     
        for layer in range(2,no_of_hidden_layers + 3):
             if layer==2:  
                v_pred[layer] = np.matmul(np.transpose(W[layer - 2]),np.array( [test[j] ]).T) + bias[layer]
                y_pred[layer] = phi(v_pred[layer])
             else:
                v_pred[layer] = np.matmul(np.transpose(W[layer - 2]),np.array(y_pred[layer - 1])) + bias[layer]
                y_pred[layer] = phi(v_pred[layer])
        Y_pred.append(y_pred[layer][0][0])
    
    error = MAPE(np.array(shape(np.array(Y_pred))), test_y)
    print('k Error:',np.sum(error)/test_size)
    ERROR.append(np.sum(error)/test_size)
    MY = shape(np.array(Y_pred))
    iterations.append(k)
    
    Y_valid = []
    for j in range(validation_size):
        for layer in range(2,no_of_hidden_layers + 3):
            if layer==2:
                v_pred[layer] = np.matmul(np.transpose(W[layer - 2]),np.array( [validation[j] ]).T) + bias[layer]
                y_pred[layer] = phi(v_pred[layer])
            else:
                v_pred[layer] = np.matmul(np.transpose(W[layer - 2]),np.array(y_pred[layer - 1])) + bias[layer]
                y_pred[layer] = phi(v_pred[layer])
        Y_valid.append(y_pred[layer][0][0])
    v_error = MAPE(np.array(shape(np.array(Y_valid))), validation_y)
    print('Valid Error:',np.sum(v_error)/validation_size)
    V_ERROR.append(np.sum(error)/validation_size)
    MY = shape(np.array(Y_valid))
#plt.plot(iterations,V_ERROR)
#plt.plot(iterations,ERROR)
#
#plt.xlabel('Epochs')
#plt.ylabel('MAPE Error')
#plt.title('Error vs Epochs - validation')


Y_pred = denormalize(np.array(Y_pred),test_y_max,test_y_min)
test_y = denormalize(np.array(test_y),test_y_max,test_y_min)
plt.xlim(420, 500)
plt.ylim(420, 500)
plt.scatter(test_y, Y_pred)
m, b = np.polyfit(test_y.T[0], Y_pred, 1)
plt.plot(test_y, m*test_y + b,color='r')
plt.xlabel('Actual values')
plt.ylabel('ANN values')
plt.title('Actual vs ANN')

r2 = find_r2(shape(Y_pred),test_y)
print('R-Squared value:',r2)
#









