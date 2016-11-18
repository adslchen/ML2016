from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D,BatchNormalization
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
#from keras.advanced_activations import ELU
from keras.callbacks import Callback
from keras.regularizers import l2, activity_l2
from keras.models import load_model
import keras.optimizers
import pickle
import numpy as np
from keras.utils import np_utils
from readData import read_label, read_unlabel, read_test
import tensorflow as tf
import sys
import os
#from keras.callbacks.CSVLogger
# Parameter setting

mode = sys.argv[1]
data_path = sys.argv[2]
model_name = sys.argv[3]
outFilename = sys.argv[4]
epoches = int(sys.argv[5])
epoches_correct = int(sys.argv[6])
maxself_train = int(sys.argv[7])
data_augmentation = True
validation = True
trainFile = sys.argv[6]
self_train_datagen = False
self_train = True
STOP = False

print(self_train)

# To limit the GPU resources
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.33
set_session(tf.Session(config=config))
tf.python.control_flow_ops = tf


class EarlyStoppingByAccVal(Callback):
    def __init__(self, monitor='val_acc', value=0.80, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current > self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True
            STOP = True
# Definitnion of early stopping
earlyStopping=EarlyStoppingByAccVal(monitor='val_acc',value=0.80,verbose=1)
earlyStopping1=EarlyStoppingByAccVal(monitor='val_acc',value=0.83,verbose=1)
earlyStopping2=EarlyStoppingByAccVal(monitor='val_acc',value=0.86,verbose=1)
# Save the best model
modelCheckpoint = keras.callbacks.ModelCheckpoint('best_model.hdf5', monitor='val_acc', verbose=0,
                                save_best_only=True, save_weights_only=False, mode='auto')
#csv_logger = keras.callbacks.CSVLogger(trainFile+'.log')
#csv_logger2 = keras.callbacks.CSVLogger(trainFile+'_correct.log')
#csv_logger3 = keras.callbacks.CSVLogger(trainFile+'_unlabel.log')
ELU = keras.layers.advanced_activations.ELU()
#  Define VGG like model
model = Sequential()
# input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
# # this applies 32 convolution filters of size 3x3 each.
model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(32,32,3),dim_ordering='tf',init='he_normal'))
model.add(BatchNormalization())
model.add(ELU)
model.add(Convolution2D(64, 3, 3,init='he_normal'))
model.add(BatchNormalization())
model.add(ELU)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
#
model.add(Convolution2D(128, 3, 3, border_mode='same',init='he_normal'))
model.add(BatchNormalization())
model.add(ELU)
model.add(Convolution2D(128, 3, 3,init='he_normal'))
model.add(BatchNormalization())
model.add(ELU)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
#
model.add(Convolution2D(256, 3, 3, border_mode='same',init='he_normal'))
model.add(BatchNormalization())
model.add(ELU)
model.add(Convolution2D(256, 3, 3,init='he_normal'))
model.add(BatchNormalization())
model.add(ELU)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
# Note: Keras does automatic shape inference.
model.add(Dense(512))
model.add(BatchNormalization())
model.add(ELU)
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(BatchNormalization())
model.add(ELU)
model.add(Dropout(0.5))
#model.add(Dense(512))
#model.add(BatchNormalization())
model.add(Dense(10))
model.add(Activation('softmax'))
#
# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
adam = keras.optimizers.Adam(lr=0.000005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

def getUnlabel():

    unlabel = read_unlabel(data_path)/255.
    return unlabel

def getLabel():
    class_num = 500
    train_num = 450
    val_num = 50
    # Load the all_label.npy
    X_train = read_label(data_path).astype('float32')/255.
    # Create the label matrix
    y_train = np.empty([X_train.shape[0],1],dtype=int)
    for i in range(10):
        y_train[i*class_num:(i+1)*class_num,0] = i
    Y_train = np_utils.to_categorical(y_train, 10)

    if validation == True:

        Xt_train = np.empty([4500,32,32,3],dtype=float)
        Xt_val = np.empty([500,32,32,3],dtype=float)
        Yt_train = np.empty([4500,10],dtype=int)
        Yt_val = np.empty([500,10],dtype=int)

        # Shuffle the data to create validation set training set :4500 , validation set : 500
        for i in range(0,X_train.shape[0]/class_num):

            temp_train = X_train[i*class_num:(i+1)*class_num]
            temp_label = Y_train[i*class_num:(i+1)*class_num]
            #print("label:", temp_label.shape)
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
    #    print("Y_train",Y_train[450:900])
    #    print("Y_val",Y_val[:50])
    #    print("X_train",X_train[:450])
    #    print("X_train",X_val[:50])
        return [X_train,X_val,Y_train,Y_val]
    else:
        return [X_train,0,Y_train,0]
def self_train1(X_train,Y_train,X_val,Y_val,unlabel):
    correct_data = X_train
    correct_y = Y_train
    if data_augmentation ==True:
        datagen = ImageDataGenerator(
                #featurewise_center=True,
                #featurewise_std_normalization=True,
                rotation_range=0,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                dim_ordering='tf')
        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        datagen.fit(X_train)

        if validation == True:
            # fits the model on batches with real-time data augmentation:
            model.fit_generator(datagen.flow(X_train, Y_train, batch_size=64),
            samples_per_epoch=len(X_train)*10, nb_epoch=epoches,validation_data=(X_val,Y_val),callbacks=[earlyStopping])
        else:

            model.fit_generator(datagen.flow(X_train, Y_train, batch_size=64),
            samples_per_epoch=len(X_train)*10, nb_epoch=epoches)
                            #callbacks=[earlyStopping])
#    raw_input()
    #data_number = 45000-unlabel.shape[0]+500
    if (self_train == True):
        patience = 0
        count = 1
        global epoches_correct
        while( patience <= 2 and count < maxself_train and STOP == False):
            th = 0.99
            if (count > 3):
                th = 0.9
                epoches_correct = 1
            prediction = model.predict_proba(unlabel)
            unlabel_index = np.argwhere(prediction > th)
            if (unlabel_index.shape[0] < 200):
                patience += 1
                break
            #print("index:",unlabel_index.shape)
            labeled = unlabel[unlabel_index[:,0]]
            #print("label shape", labeled.shape)
            y_labeled = unlabel_index[:,1]
            #print(y_labeled.shape)
            y_labeled = np_utils.to_categorical(y_labeled,10)


            X_train = np.concatenate((X_train,labeled),axis=0)
            Y_train = np.concatenate((Y_train,y_labeled),axis=0)
            # Take out the rest data
            mask = np.ones(unlabel.shape[0], dtype=bool)
            mask[unlabel_index] = False
            unlabel = unlabel[mask]
            #print("unlabel shape:", unlabel.shape)
            #print("Training on ", )
            if validation == True:

                if count > 3 and self_train_datagen == True:
                    datagen.fit(X_train)
                    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32),
                    samples_per_epoch=len(X_train), nb_epoch=10,validation_data=(X_val,Y_val),callbacks=[earlyStopping2])
                    model.save(model_name+'.h5')

                else:
                    datagen.fit(correct_data)

                    # fits the model on batches with real-time data augmentation:
                                       # callbacks=[modelCheckpoint])

                    model.fit(X_train,Y_train , batch_size=32, nb_epoch=1,validation_data=(X_val,Y_val))

                    model.fit_generator(datagen.flow(correct_data, correct_y, batch_size=32),
                    samples_per_epoch=len(correct_data)*10, nb_epoch=epoches_correct,validation_data=(X_val,Y_val),
                                    callbacks=[earlyStopping1])
                    model.save(model_name+'.h5')
            else:
                datagen.fit(correct_data)

                print("Training on the correct data...")
                # fits the model on batches with real-time data augmentation:
                print("Training on the added data...")
                model.fit(X_train,Y_train , batch_size=32, nb_epoch=1)
                model.fit_generator(datagen.flow(correct_data, correct_y, batch_size=128),
                samples_per_epoch=len(correct_data)*10, nb_epoch=epoches_correct)#,callbacks=[modelCheckpoint])#,validation_data=(X_val,Y_val))

            count += 1
            # Predict 500 unlabel data
            #to_be_label = unlabel[:500]
            # Cut off the predicted 500 data
            #unlabel = unlabel[500:]
            #print("predicting the ", data_number, "data....")
            # Add the new label data
            #X_train = np.concatenate((X_train,to_be_label))
            #Y_train = np.concatenate((Y_train,prediction))
            #self_train1(X_train,Y_train,X_val,Y_val,unlabel)
   #print("predict label shape is ",yy.shape)
#def VGG_like(X_train,Y_train):
        #
#    model.fit(X_train,Y_train , batch_size=32, nb_epoch=50,validation_data=(X_train,Y_train))
def readTest():
    X_test = read_test(data_path)/255.
    return X_test

def predictTest():
    model = load_model(model_name+'.h5')
    X_test = readTest()
    prediction = model.predict_classes(X_test, batch_size=32, verbose=0)
    f = open(outFilename,'w')
    for row in range(prediction.shape[0]+1):
        if row == 0:
            f.write('id,class\n')
        else:
            temp = int(prediction[row-1])
            f.write(str(row-1)+','+str(temp)+'\n')
    f.close

    #os.remove('all_label.npy')
    #os.remove('all_unlabel.npy')
    #os.remove('all_test.npy')


if __name__ == "__main__":
    #readData()
    if mode == "train":
        [X_train,X_val,Y_train,Y_val] = getLabel()
        unlabel = getUnlabel()
        unlabel = np.concatenate((unlabel,readTest()),axis=0)
        self_train1(X_train,Y_train,X_val,Y_val,unlabel)
    elif mode == "test":
        predictTest()

