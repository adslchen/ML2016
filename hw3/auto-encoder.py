from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.models import load_model
import numpy as np
from sklearn import svm
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
import keras.optimizers
import pickle
from keras.utils import np_utils
import tensorflow as tf
import sys
from readData import read_label, read_unlabel, read_test
# To limit the GPU resources
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.33
set_session(tf.Session(config=config))
tf.python.control_flow_ops = tf
rep_size = 256
use_saved_model = False
# Define the model
mode = sys.argv[1]
data_path = sys.argv[2]
DNN_name = sys.argv[3]
output_name = sys.argv[4]

input_img = Input(shape=(32, 32, 3))

x = Convolution2D(32, 3, 3, activation='relu', border_mode='same',dim_ordering='tf')(input_img)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
encoded = MaxPooling2D((2, 2), border_mode='same')(x)
print(encoded)
# at this point the representation is (16, 8, 8) i.e. 256-dimensional

x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(3, 3, 3, activation='sigmoid', border_mode='same')(x)
print(decoded)
autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)
autoencoder.compile(optimizer='adam', loss='mse')

model = Sequential()
#model.add(Flatten(input_shape=(16,8,8)))
model.add(Dense(64, input_dim=rep_size))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(128))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(10))
model.add(Activation('softmax'))
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

def set_datagen():
    datagen = ImageDataGenerator(
            #featurewise_center=True,
            #featurewise_std_normalization=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            dim_ordering='tf')
# compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)
    return datagen

def train_autoencoder(X_train,Y_train,X_val,Y_val,unlabel,X_test):
    #if (use_saved_model == False):
    X_all = np.concatenate((X_train,unlabel),axis=0)
    X_all = np.concatenate((X_all,X_test),axis=0)
    autoencoder.fit(X_all, X_all,batch_size=128,
                    nb_epoch=30,
                    validation_data=(X_val, X_val))
    #encoder.save('encoder.h5')
        #del encoder
    #print("yeee")
    #print(encoder)
    #else:
        #encoder = load_model('encoder.h5')
    encoder.save('encoder.h5')
    representation = encoder.predict(X_train).reshape((X_train.shape[0],16*4*4))
    #representation = representation/np.linalg.norm(representation,axis=1).transpose()
    val_rep = encoder.predict(X_val).reshape((X_val.shape[0],16*4*4))
    #val_rep = val_rep/np.linalg.norm(val_rep,axis=1).transpose()
    model.fit(representation,Y_train,batch_size=128,nb_epoch=50,validation_data=(val_rep,Y_val))
    model.save(DNN_name+'.h5')
#def train_DNN():
#    model.fit(encoder)

def getData(data_path):
    class_num = 500
    train_num = 450
    val_num = 50

    #X_train = np.load('all_label.npy')
    X_train = read_label(data_path)
    #X_test = np.load('test.npy')
    X_test = read_test(data_path)
    unlabel = np.load('all_unlabel.npy').astype('float32')/ 255.
    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.
    Xt_train = np.empty([4500,32,32,3],dtype=float)
    Xt_val = np.empty([500,32,32,3],dtype=float)
    Yt_train = np.empty([4500,10],dtype=int)
    Yt_val = np.empty([500,10],dtype=int)
    y_train = np.empty([X_train.shape[0],1],dtype=int)
    print(y_train.shape)
    for i in range(10):
        y_train[i*class_num:(i+1)*class_num,0] = i
    print(y_train)
    Y_train = np_utils.to_categorical(y_train, 10)
    # Shuffle the data to create validation set training set :4500 , validation set : 500
    for i in range(0,X_train.shape[0]/class_num):

        temp_train = X_train[i*class_num:(i+1)*class_num]
        temp_label = Y_train[i*class_num:(i+1)*class_num]
        #Shuffle the 500 indice
        indices = np.random.permutation(temp_train.shape[0])
        training_idx, test_idx = indices[:train_num], indices[train_num:]
        temp_train, temp_val = temp_train[training_idx], temp_train[test_idx]
        temp_label,temp_label_val = temp_label[training_idx],temp_label[test_idx]

        Xt_train[i*train_num:(i+1)*train_num] = temp_train
        Xt_val[i*val_num:(i+1)*val_num] = temp_val
        Yt_train[i*train_num:(i+1)*train_num] = temp_label
        Yt_val[i*val_num:(i+1)*val_num] = temp_label_val

        #X_train, X_val = X_train[training_idx], X_train[test_idx]
        #Y_train, Y_val = Y_train[training_idx,:], Y_train[test_idx,:]
    X_train = Xt_train
    X_val = Xt_val
    Y_train = Yt_train
    Y_val =Yt_val
    return X_train, X_val, Y_train, Y_val, unlabel, X_test
def readTest(data_path):
    X_test = read_test(data_path)
    return X_test

def predictTest(data_path):
    X_test = readTest(data_path)
    encoder = load_model('encoder.h5')
    DNN = load_model(DNN_name+'.h5')
    representation = encoder.predict(X_test,batch_size=32,verbose=0).reshape((10000,rep_size))
    prediction = DNN.predict_classes(representation, batch_size=32, verbose=0)
    f = open('auto_predict.csv','w')
    for row in range(prediction.shape[0]+1):
        if row == 0:
            f.write('id,class\n')
        else:
            temp = int(prediction[row-1])
            f.write(str(row-1)+','+str(temp)+'\n')
    f.close
    #os.remove('all_label.npy')
    #os.remove('all_unlabel.npy')
    #os.remove('test.npy')

if __name__ == "__main__":
    if mode == 'train':
        [X_train, X_val, Y_train, Y_val, unlabel, X_test] = getData(data_path)
        train_autoencoder(X_train,Y_train,X_val,Y_val,unlabel, X_test)

    elif mode == 'test':
        predictTest(data_path)
