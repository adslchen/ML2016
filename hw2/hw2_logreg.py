import csv
import numpy as np
from math import exp, expm1
import time
import sys

def readData(train_datapath):
    f = open(train_datapath,'r')
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
    ##
    ## for logistic regression backpropagation, we can see https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/pdfs/40%20LogisticRegression.pdf
    ##
    
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
        new_y = 1/(1+np.exp(-np.dot(training_data,w)))
        if (count%1000 == 0):
            #print(new_y)
            cross_entropy = -np.sum((np.multiply(answer,np.log(new_y+1e-30))+np.multiply(np.subtract(1,answer),np.log(np.subtract(1,new_y)+1e-30))))/4001
            error = np.linalg.norm(np.subtract(new_y,answer))**2/len(new_y)
            print("Training iteration ", count, ", error = ", cross_entropy)
    print("--- %s seconds ---" % (time.time() - start_time))
    return w 

        
def testData(w,test_datapath,output_prediction):
    f = open(test_datapath,'r')
    temp = [] 
    for row in csv.reader(f):
        temp.append(row)
        #rownumber += 1()
    f.close
    data = np.asarray(temp).astype(np.float)
    data = np.delete(data,0,1)
    a = np.ones((data.shape[0],1))
    data = np.concatenate((data,a),axis=1)
    y = 1/(1+np.exp(-np.dot(data,w)))
    for i in range(len(y)):
        if(y[i] > 0.5):
            y[i] = 1
        else:
            y[i] = 0

    f = open(output_prediction,'w')
    for row in range(len(y)+1):
        if (row==0):
            f.write('id,label\n')
        else:
            temp = int(y[row-1])
            f.write(str(row)+','+str(temp)+'\n')
    f.close


if __name__ == "__main__":

    mode = sys.argv[1]
    #data = temp[0]
    #answer = temp[1]
    if(mode == 'train'):
        train_datapath = sys.argv[2]
        output_model = sys.argv[3]
        [data, answer] = readData(train_datapath)
        [cols, rows] = data.shape
        a = np.ones((cols,1))
        data = np.concatenate((data,a),axis=1)
        data = data.astype(np.float)
        learning_rate = 0.3
        epoch =  100000
        w = train(data,answer,learning_rate,epoch)
        np.save(output_model,w)
    elif (mode == 'test'):
        model_name = sys.argv[2]
        testing_data = sys.argv[3]
        output_prediction = sys.argv[4]
        name = model_name + '.npy'
        w = np.load(name)
        testData(w,testing_data,output_prediction)
    #print(data)
    #print(data.shape)
    #print(answer)
    #print(answer.shape)


