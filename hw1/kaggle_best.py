import csv 
import numpy as np

# Read in data to list
f = open('train.csv','r')
rownumber = 0
data = []
for row in csv.reader(f):
    if rownumber is not 0:
        del row[1]
        if row[1] == 'RAINFALL':
            for x in range(len(row)):
                if(row[x] == 'NR'):
                    row[x] = 0
        data.append(row)
    rownumber += 1
f.close
#print(len(data))

# transfer list to numpy array 
data2 = np.zeros((18,1))
for i in range(0,len(data),18):
    a = np.asarray(data[i:i+18])
    if(i > 0):
        a = a[:,2:26]
    data2 = np.concatenate((data2,a),axis=1)
    
data2 = np.delete(data2,0,1)
data2 = np.delete(data2,0,1)

#print(data2)
#print(data2.shape)
""" [['AMB_TEMP' '14' '14' ..., '13' '13' '13']
    ['CH4' '1.8' '1.8' ..., '1.8' '1.8' '1.8']
    ['CO' '0.51' '0.41' ..., '0.51' '0.57' '0.56']
    ...,
    ['WIND_DIREC' '35' '79' ..., '118' '100' '105']
    ['WIND_SPEED' '1.4' '1.8' ..., '1.5' '2' '2']
    ['WS_HR' '0.5' '0.9' ..., '1.6' '1.8' '2']]   """

training_data = np.zeros((1,162))
y_head = np.zeros((1,1))
print("Processing data...")
for i in range(1,data2.shape[1]-9):
    #print(data2[:,i:i+9].shape)
    a = np.reshape(data2[:,i:i+9],(1,162))
    b = np.reshape(data2[9,i+9],(1,1))
    training_data = np.concatenate((training_data,a))
    y_head = np.concatenate((y_head,b))
training_data = np.delete(training_data,0,0)
y_head = np.delete(y_head,0,0)
a = np.ones((5751,1))
y_head = y_head.astype(np.float)
#print(a)
training_data = np.concatenate((training_data,a),axis=1)
training_data = training_data.astype(np.float)
#print(training_data.shape)
#np.savetxt('trainingdata.txt',training_data,delimiter=',')


X = training_data
w = np.zeros((163,1))
learning_rate = 0.00000150
count = 0
X_positive = np.linalg.pinv(X)
w_head = np.dot(X_positive,y_head)
while (count < 300000):
    y = np.dot(training_data,w)
    a = np.dot(X.transpose(),np.subtract(y,y_head))
    deltaF = np.multiply(2.0, a.astype(np.float))/5751
    new_w = np.subtract(w,np.multiply(learning_rate,deltaF))
    w = new_w
    count += 1
    #print(w)
    # count error
    c = np.subtract(np.dot(training_data,w),y_head) 
    error = (np.linalg.norm(c)**2)/len(c)
    #c_head = np.subtract(np.dot(training_data,w_head),y_head)
    #w_head_error = (np.linalg.norm(c_head)**2)/len(c_head)
    if (count%1000 == 0): 
        print("1 Training iteration ", count," , error = ",error)
     #   print("2 Loss of w head = ", w_head_error )
    #w_error = np.average(np.subtract(w,w_head))
    #print("2 w error = ", w_error)
#   Read in test file
#f = open('my_w.txt','w')
#for item in w:
#    f.write('w'+"%s\n" % item)
#for item in w_head:
#    f.write('w_head = '+"%s\n" % item)
#f.close
f = open('test_X.csv','r')
rownumber = 0
data = []
for row in csv.reader(f):
    for x in range(len(row)):
        if(row[x] == 'NR'):
                row[x] = 0
    data.append(row)
    rownumber += 1
f.close
test_data = np.asarray(data)
test_data = test_data[:,2:11]
yy = np.zeros((1,162))
for i in range(0,test_data.shape[0]-17,18):
    #print(test_data[i:i+18,:].shape)
    a = np.reshape(test_data[i:i+18,:],(1,162))    
    yy = np.concatenate((yy, a))
#print(yy)
yy = yy[1:241,:]
b = np.ones((240,1))
yy = np.concatenate((yy,b),axis=1)
yy = yy.astype(np.float)
#  compute answer
answer = np.dot(yy,w_head)
my_answer = np.dot(yy,w)
distance = np.linalg.norm(np.subtract(answer,my_answer))**2
print("Answer is ", answer)
print("My Answer is ", my_answer)
print("Distance: ", distance)

f = open('kaggle_best.csv','w')
#s = csv.writer(f)

for row in range(len(answer)+1):
    if row == 0:
        f.write('id,value\n')
    else:
        #temp = 'id_'+str(row-1)+','+str(my_answer[row])
        temp = int(my_answer[row-1])
        f.write('id_'+str(row-1)+','+str(temp)+'\n')

f.close

