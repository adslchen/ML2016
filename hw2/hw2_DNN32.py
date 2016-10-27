import csv
import numpy as np
from math import exp, expm1
import time
import sys

def readData(train_data):
    f = open(train_data,'r')
    temp = []
    for row in csv.reader(f):
        temp.append(row)
    f.close
    data = np.asarray(temp)
    data = np.delete(data,0,1)
    answer = data[:,data.shape[1]-1]
    data = np.delete(data,data.shape[1]-1,1)
    re = [data, answer]
    return re

def train(training_data,answer,learning_rate,epoch):
    start_time = time.time()
    [cols,rows] = training_data.shape
    answer = np.reshape(answer,(cols,1)).astype(np.float)
    w = np.zeros((rows,1))
    count = 0
    adagrad = np.zeros((rows,1))
    while (count < epoch):
        y = 1/(1+np.exp(-np.dot(training_data,w)))
        delta = np.multiply(2.0,np.dot(training_data.transpose(),np.subtract(y,answer)).astype(np.float))/len(w)
        adagrad = adagrad+delta**2
        new_w = np.subtract(w, np.multiply(learning_rate,delta/np.sqrt(adagrad)))
        w = new_w
        count += 1
        new_y = 1/(1+np.exp(np.dot(training_data,w)))
        if (count%1000 == 0):
            #print(new_y)
            cross_entropy = -np.sum((np.multiply(answer,np.log(new_y+1e-30))+np.multiply(np.subtract(1,answer),np.log(np.subtract(1,new_y)+1e-30))))/4001
            error = np.linalg.norm(np.subtract(new_y,answer))**2/len(new_y)
            print("Training iteration ", count, ", error = ", cross_entropy)
    print("--- %s seconds ---" % (time.time() - start_time))
    return w 
def sigmoid(x):
    return 1/(1+np.exp(-x))
def desigmoid(x):
    return x*(1-x)
def normalize(x):
    row_sums = x.sum(axis=1)
    return x / row_sums[:, np.newaxis]


def DNN(training_data,answer,learning_rate,epoch,lamda):
    start_time = time.time()
    nodes = 31
    lamda1 = lamda
    lamda2 = lamda
    w1 = np.random.rand(58,nodes)
    w2 = np.random.rand(nodes,1)
    count = 0
    #adagrad2 = np.zeros((4001,1))+1e-30
    #adagrad1 = np.zeros((4001,31))+1e-30
    m1 = np.zeros((nodes,1))
    v1 = np.zeros((nodes,1))
    m2 = np.zeros((58,nodes))
    v2 = np.zeros((58,nodes))
    Beta1_1 = 0.9
    Beta1_2 = 0.999
    Beta2_1 = 0.9
    Beta2_2 = 0.999
    epil = 1e-8
    while(count < epoch):
        A1 = sigmoid(np.dot(training_data,w1))
        A1.T[0].fill(1)
        y = sigmoid(np.dot(A1,w2))
        #print(answer.shape) 
        #print(y.shape)
        #print(desigmoid(y).shape)
        #print((answer/y).shape)
        #print((answer/y+(1-answer)/(y-1)).shape)
        delta2 = np.multiply(desigmoid(y),-(answer/(y+1e-30)+(1-answer)/(y-1+1e-30)))
       # adagrad2 = adagrad2 + delta2**2
        #print(delta2.shape)
        delta1 = np.multiply(desigmoid(A1),np.dot(delta2,w2.transpose()))
        #adagrad1 = adagrad1 + delta1**2
               

        Gra2 = np.dot(A1.transpose(), delta2) + lamda2 * w2
        Gra1 = np.dot(training_data.transpose(),delta1) +lamda1 *w1
        
        m1 = Beta1_1*m1 + (1-Beta1_1) * Gra2
        v1 = Beta1_2*v1 + (1-Beta1_2) * Gra2**2
        
        m2 = Beta2_1*m2 + (1-Beta2_1) * Gra1
        v2 = Beta2_2*v2 + (1-Beta2_2) * Gra1**2
        #print(Gra2)(1-Beta2_1**count)
        #print(Gra1)(1-Beta1_1**count)
        count += 1
        #lr_t1 = learning_rate * np.sqrt(1-Beta1_2**count)/(1-Beta1_1**count)
        #lr_t2 = learning_rate * np.sqrt(1-Beta2_2**count)/(1-Beta2_1**count)
        w2 = w2 - learning_rate * m1 * (1-Beta2_1**count)/(np.sqrt(v1)+epil)
        w1 = w1 - learning_rate * m2 * (1-Beta1_1**count)/(np.sqrt(v2)+epil)
        #print(w2)
        #print(w1)
        if(count%1000 == 0): 
            cross_entropy = -np.sum(np.multiply(answer,np.log(y+1e-30))+np.multiply(np.subtract(1,answer),np.log(np.subtract(1,y)+1e-30)))/len(y)
            print("Training epoch: ", count, ", error: ", cross_entropy)
    print("--------- %s seconds ----" %(time.time()-start_time))  
    return [w1, w2]
def crossValidation(data,answer):
    data = np.delete(data,data.shape[1]-1,1)
    data = np.concatenate((data,answer),axis=1)
    
    iteration = [1000, 5000, 10000, 20000, 30000]
    lamda = [10 ,1 , 0.5 , 0.1 , 0.01]
    learning_rate = 0.001
    for i in lamda:
        np.random.shuffle(data)
        data1 = data[:2001,:]
        data2 = data[2001:,:]

        answer1 = data1[:,data1.shape[1]-1]
        answer1 = np.reshape(answer1,(answer1.shape[0],1))
        data1 = np.delete(data1,data1.shape[1]-1,1)
        a = np.ones((data1.shape[0],1))
        data1 = np.concatenate((data1,a),axis=1)
        #print(data1.shape)
        #print(answer1.shape)
        answer2 = data2[:,data2.shape[1]-1]
        answer2 = np.reshape(answer2,(answer2.shape[0],1))
        data2 = np.delete(data2,data2.shape[1]-1,1)
        a = np.ones((data2.shape[0],1))
        data2 = np.concatenate((data2,a),axis=1)
        #print(answer2.shape)
        #print(data2.shape)
        print("------------------------lambda = %s ---------------" %(i))
        [w1,w2] = DNN(data1,answer1,learning_rate, 5000,i)

        A1 = sigmoid(np.dot(data2,w1))
        A1.T[0].fill(1)
        y = sigmoid(np.dot(A1,w2))
        cross_entropy = -np.sum(np.multiply(answer2,np.log(y+1e-30))+np.multiply(np.subtract(1,answer2),np.log(np.subtract(1,y)+1e-30)))/len(y)
        print(" lambda",i , ", cross-entropy = ",cross_entropy)

def testData(w1,w2,test_data,prediction):

    f = open(test_data,'r')
    temp = []

    for row in csv.reader(f):
        temp.append(row)
    f.close

    data = np.asarray(temp).astype(np.float)
    data = np.delete(data,0,1)
    a = np.ones((data.shape[0],1))
    data = np.concatenate((data,a),axis=1)

    A1 = sigmoid(np.dot(data,w1))
    A1.T[0].fill(1)
    y = sigmoid(np.dot(A1,w2))

    for i in range(len(y)):
        if(y[i] > 0.5):
            y[i] = 1
        else:
            y[i] = 0

    f = open(prediction,'w')
    for row in range(len(y)+1):
        if (row==0):
            f.write('id,label\n')
        else:
            temp = int(y[row-1])
            f.write(str(row)+','+str(temp)+'\n')
    f.close


if __name__ == "__main__":
    
    if (sys.argv[1] == 'train'):
        train_data = sys.argv[2]
        output_model = sys.argv[3]
        [data, answer] = readData(train_data)
        [cols, rows] = data.shape
        #data = temp[0]
        #answer = temp[1]
        a = np.ones((cols,1))
        data = np.concatenate((data,a),axis=1)
        data = data.astype(np.float)
        answer = np.reshape(answer,(4001,1)).astype(np.float)
        learning_rate = 0.001
        epoch =  5000
    #crossValidation(data,answer)
        [w1,w2] = DNN(data,answer,learning_rate,epoch,0.5)
        np.savez(output_model, w1, w2)
    elif (sys.argv[1] == 'test'):
        model_name = sys.argv[2]
        test_data = sys.argv[3]
        prediction = sys.argv[4]
        model_name = model_name + '.npz'
        npzfile = np.load(model_name)
        w1 = npzfile['arr_0']
        w2 = npzfile['arr_1']
        testData(w1,w2,test_data,prediction)
    #print(data)
    #print(data.shape)
    #print(answer)
    #print(answer.shape)


