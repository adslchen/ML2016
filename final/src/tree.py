
import sys
import time
import numpy as np
np.set_printoptions(precision = 2, threshold = np.nan)
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import csv
import pandas as pd
import pickle
from util import probSort
start = time.time()

w_proba = 0.2
w_like = 0.9

#train
if(sys.argv[1]=='t'):
    print "Loading training data..."
    #train = np.load('data/X_train.npy') #87141731*8
    #label = np.load('data/Y_train.npy') #87141731
    #label = pd.read_csv('ffm/t_encode_expr3.csv',usecols=['clicked'])
    count = 0
    for train in pd.read_csv('train.csv',chunksize=1000000,dtype={'a':np.float}):
        count += 1
        if count != 0 and count %100 == 0:
            print("count:" ,count)
    #train = pd.read_csv('ffm/t_encode_expr3.csv')
        #print("dropping..")
        label = train['clicked']
        train.drop('clicked',axis=1,inplace=True)
        train.drop('likelihood',axis=1,inplace=True)
        #print("turning to numpy")
        train = train.as_matrix()
        label = label.as_matrix()
        #print "train.shape: ", train.shape
        #print "label.shape: ", label.shape
        if(sys.argv[3]=='dt'):
            #print "Building decision tree..."
            clf = tree.DecisionTreeClassifier()
        elif(sys.argv[3]=='rf'):
            #print "Building random forest..."
            clf = RandomForestClassifier(max_depth = 10, n_estimators=100, max_features=None)
        clf.fit(train, label)###
    print "Saving tree model..."
    pickle.dump(clf, open("model/"+sys.argv[2], 'wb'))

#predict
if(sys.argv[1]=='p'):
    print "Loading testing data..."
    #test = np.load('data/test.npy') #32225162*8
    test = pd.read_csv("test_encode.csv")
    likelihood = test['likelihood']
    test.drop('likelihood',axis=1,inplace=True)
    test = test.as_matrix()
    likelihood = likelihood.as_matrix()
    #likelihood = test[:, 8]###
    clf = pickle.load(open(sys.argv[2], 'rb'))
    print "test.shape: ", test.shape
    print "Predicting..."
    clicked_proba = clf.predict_proba(test)[:, 1]###
    print "clicked_proba.shape ", clicked_proba.shape
    print "clicked_proba: ", clicked_proba[0:50]
    #np.save('data/clicked_proba.npy', clicked_proba)
    df = pd.read_csv("../data/clicks_test.csv",usecols=['display_id','ad_id'])
    #df = pd.DataFrame(test[:, [0,1]].astype(int))
    #df.columns = ['display_id', 'ad_id']
    df['proba'] = clicked_proba*w_proba + likelihood*w_like
    probSort(df,'tree_expr3_result3.csv')
    #df.sort_values(['display_id','proba'], inplace=True, ascending=False)
    #subm = df.groupby('display_id').ad_id.apply(lambda x: " ".join(map(str,x))).reset_index()
    #subm.to_csv(sys.argv[3], index=False)

def validation(data):
    #################################################
    # Input data : a pandas dataframe
    # data contains  'display_id' 'ad_id' 'proba'
    # output is the MAP score
    ################################################
    from ml_metrics import mapk
    valid = pd.read_csv('../data/clicks_train.csv').tail(12433214)
    #valid = pd.read_csv('data/valid.csv').head(10000)
    y = valid[valid.clicked==1].ad_id.values
    y = [[_] for _ in y]

    data = data.sort_values(['display_id','proba'],ascending=False).groupby('display_id').ad_id.apply(list)
    data = data.tolist()
    return (mapk(y,data, k=12))

#validation
if(sys.argv[1]=='v'):
    print "Loading validation data..."
    #valid = np.load('data/X_valid.npy')
    #valid = np.load('data/X_valid.npy')[0:10000]
    valid = pd.read_csv('valid.csv')
    likelihood = valid['likelihood'].as_matrix()###
    valid.drop('likelihood',axis=1,inplace=True)
    valid = valid.as_matrix()
    label = valid[:,0] # get the label
    valid = valid[:,1:] 
    clf = pickle.load(open(sys.argv[2], 'rb'))
    print "valid.shape: ", valid.shape
    print "Predicting..."
    clicked_proba = clf.predict_proba(valid)[:, 1]###
    print "clicked_proba.shape ", clicked_proba.shape
    print "clicked_proba: ", clicked_proba[0:50]
    #df = pd.DataFrame(valid[:, [0,1]])
    df = pd.read_csv('../data/clicks_train.csv',usecols=['display_id','ad_id']).tail(12433214)
    #df.columns = ['display_id', 'ad_id']
    df['proba'] = clicked_proba*w_proba + likelihood*w_like
    print "Scoring...(w_proba,w_like) = ", w_proba, w_like
    print validation(df)
    """
    for i in xrange(11):
        df['proba'] = clicked_proba*(1-0.1*i) + likelihood*0.1*i
        print "Scoring...(w_proba,w_like) = ", (1-0.1*i), 0.1*i
        print validation(df)
    """
end = time.time()
total = end-start
second = total%60
minute = int(total/60%60)
hour = int(total/60/60)
print hour, "hours, ", minute, "minutes, ", second, "seconds."

