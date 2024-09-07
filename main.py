import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import random
from random import randint

#warning, very volatile to change performance
DAYS_TRACKING = 10

def z_score_normalize(data):
    sum = 0
    data = data.astype(np.float64)
    for i in range(0, data.shape[1]):
        sum += data[0][i]
    mean = sum / data.shape[1]
    newData = (data - mean) / mean
    return newData, mean

def unnormalize_data(normalized_data, mean):
    original_data = (normalized_data * mean) + mean
    return original_data
file_path = "portfolio_data.csv" 
df = pd.read_csv(file_path)
data = np.array(df)
dates = pd.to_datetime(df['Date'])
values = df['AMZN']
train_dataX = data[:, :2]
train_dataX = train_dataX[:, 1:]
train_dataX = train_dataX.T
test_dataX = train_dataX[:, :400]
train_dataX = train_dataX[:, 400:]
train_dataX, meanTr = z_score_normalize(train_dataX)
train_dataX = train_dataX.astype(np.float64)
test_dataX, meanTe = z_score_normalize(test_dataX)
test_dataX = test_dataX.astype(np.float64)


def init_params():
    W1 = random.uniform(-0.01, 0.01) 
    W2 = random.uniform(-0.01, 0.01) 
    W3 = random.uniform(-0.01, 0.01) 
    W4 = random.uniform(-0.01, 0.01) 
    W5 = random.uniform(-0.01, 0.01) 
    W6 = random.uniform(-0.01, 0.01) 
    W7 = random.uniform(-0.01, 0.01)
    W8 = random.uniform(-0.01, 0.01)  
    return W1, W2, W3, W4, W5, W6, W7, W8
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))
def sigmoid_deriv(Z):
    fx = 1/(1+np.exp(-Z))
    return fx*(1-fx)
def tanh(Z):
    ez = np.exp(Z)
    enz = np.exp(-Z)
    return (ez - enz)/ (ez + enz)
def tanh_deriv(Z):
    a = tanh(Z)
    dz = 1 - a**2
    return dz
def compute_cost(val, Y): #MSE
    cost = np.sum((val - Y) ** 2)
    return cost

def forward_prop(W1, W2, W3, W4, W5, W6, W7, W8, LTM, STM, LTStore, STStore, INStore, data, time, step, max_step):
    if time == 0:
        LTStore = []
        STStore = []
        INStore = []
        LTM = 0
        STM = 0
    
    while step <= max_step:
        inp = data[0][step]
        INStore.append(inp)
        #forget gate
        LTM_forget = LTM * sigmoid(W1 * STM + W2 * inp)
        #input gate
        Potential_LTM = sigmoid(W3 * STM + W4 * inp) * tanh(W5 * STM + W6 * inp)
        #new LTM
        new_LTM = LTM_forget + Potential_LTM
        LTStore.append(new_LTM)
        #Output gate
        STM_forget = sigmoid(W7 * STM + W8 * inp)
        #new STM
        new_STM = STM_forget * tanh(new_LTM)
        STStore.append(new_STM)
        
        LTM = new_LTM
        STM = new_STM
        step += 1
    
    return data[0][step-1], new_LTM, new_STM, INStore, LTStore, STStore

def backward_prop(W1, W2, W3, W4, W5, W6, W7, W8, new_LTM, new_STM, LTStore, STStore, INStore, data, step):
    weights = [W1, W2, W3, W4, W5, W6, W7, W8]
    new_weights = []
    for i in range(len(weights)):
        updated_weight = calculate_chain(weights[i], W1, W2, W3, W4, W5, W6, W7, W8, new_LTM, new_STM, LTStore, STStore, INStore, data, step)
        new_weights.append(updated_weight)
    return new_weights

def calculate_chain(currentWeight, W1, W2, W3, W4, W5, W6, W7, W8, new_LTM, new_STM, LTStore, STStore, INStore, data, step):
    OUTPUT = data[0][step]
    dCdOutput = 2 * (OUTPUT - new_STM)
    
    if currentWeight == W1:
        init = tanh_deriv(sigmoid_deriv(STStore[-1]))
        sum = 0
        for i in range(0, len(INStore)):
            sum = init
            for j in range(len(INStore) - 1, -1 + i, -1):
                sum = sum * tanh_deriv(sigmoid_deriv(STStore[j]))
            init += sum
        return init
    else: return 1

def get_ratio(val, Y):
    if abs(val / Y) > 1:
        return abs(Y / val)
    else: return abs(val / Y)


def gradient_descent(data, learning_rate, iterations):
    W1, W2, W3, W4, W5, W6, W7, W8 = init_params()
    old_weights = [W1, W2, W3, W4, W5, W6, W7, W8]
    for j in range(iterations):
        for i in range(train_dataX.shape[1] - DAYS_TRACKING - 2):
            prevDay, fin_LTM, fin_STM, INStore, LTStore, STStore = forward_prop(W1, W2, W3, W4, W5, W6, W7, W8, LTM=0, STM=0, INStore=[], LTStore=[], STStore=[], data=data, time=0, step=i, max_step=i+DAYS_TRACKING)
            new_weights = backward_prop(W1, W2, W3, W4, W5, W6, W7, W8, fin_LTM, fin_STM, LTStore, STStore, INStore, data, i + DAYS_TRACKING + 1)
            for i in range(len(old_weights)):
                old_weights[i] -= learning_rate * new_weights[i]
        print("training")
    return old_weights
            
weights = gradient_descent(train_dataX, 0.001, 1000)

def make_predictions_accuracy(weights, data, time, sum, Aarray, inputArray):
    sum = 0
    for i in range (data.shape[1] - DAYS_TRACKING - 2):
       _, _, stm_output, _, _, _ = forward_prop(weights[0], weights[1], weights[2], weights[3], weights[4], weights[5], weights[6], weights[7], LTM=0, STM=0, LTStore=[], STStore=[], INStore=[], data=data, time=0, step=i, max_step=i+DAYS_TRACKING)
       sum += get_ratio(stm_output, data[0][i+DAYS_TRACKING + 1])
    sum = abs(sum / (data.shape[1] - DAYS_TRACKING - 2))
    print("%Off total price Avg. Accuracy: " + str(sum))
    
make_predictions_accuracy(weights, test_dataX, 0, 0, [], [])
make_predictions_accuracy(weights, train_dataX, 0, 0, [], [])

def get_specific_examples(weights, data, time, sum, Aarray, inputArray):
    num = randint(20, 100)
    _, _, stm_output, _, _, _ = forward_prop(weights[0], weights[1], weights[2], weights[3], weights[4], weights[5], weights[6], weights[7], LTM=0, STM=0, LTStore=[], STStore=[], INStore=[], data=data, time=0, step=num, max_step=num+DAYS_TRACKING)
    real_val = data[0][num+DAYS_TRACKING+1]
    stm_output = unnormalize_data(stm_output, meanTe)
    real_val = unnormalize_data(real_val, meanTe)
    print("Expected Stock closing price on day: " + str(num + DAYS_TRACKING + 1) + " with data starting from day: " + str(num) + " value: " + str(stm_output))
    print("Actual Stock closing price on day: " + str(num + DAYS_TRACKING + 1) + " with data starting from day: " + str(num) + " value: " + str(real_val))

print("Examples:")
get_specific_examples(weights, test_dataX, 0, 0, [], [])
get_specific_examples(weights, test_dataX, 0, 0, [], [])
get_specific_examples(weights, test_dataX, 0, 0, [], [])
get_specific_examples(weights, test_dataX, 0, 0, [], [])