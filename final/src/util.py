import numpy as np
import sys

import pandas as pd


def validation(data):
    #################################################
    # Input data : a pandas dataframe
    # data contains  'display_id' 'ad_id' 'proba'
    # output is the MAP score
    ################################################
    from ml_metrics import mapk
    valid = pd.read_csv('valid.csv', usecols = ['display_id', 'ad_id', 'clicked'])
    y = valid[valid.clicked==1].ad_id.values
    y = [[_] for _ in y]

    data = data.sort(['display_id','proba'],ascending=False).groupby('display_id').ad_id.apply(list)
    data = data.tolist()
    return (mapk(y,data, k=12))

def getvalData(datapath):
    ###############################################
    # usage: input the data path that you want to train
    # output : [train val]
    ###############################################
    train = pd.read_csv(datapath)

    ids = train.display_id.unique()
    ids = np.random.choice(ids, size=len(ids)//10, replace=False)

    valid = train[train.display_id.isin(ids)]
    train = train[~train.display_id.isin(ids)]

    print ("val set num: ",valid.shape,"train set num", train.shape)
    return [train, valid]

def getVal(data):

    #train = pd.read_csv(datapath,chuncksize=1000)
    
    ids  = data.display_id.unique()
    ids = np.random.choice(ids, size=len(ids)//10, replace=False)

    valid = data[data.display_id.isin(ids)]
    train = data[~data.display_id.isin(ids)]
    return [train, valid]
    #valid.to_pickle(datapath+'/clicks_valid.p')
    #train.to_pickle(datapath+'/clicks_training.p')


def probSort(result, filename):
    #################################
    # result should contain "display_id" , "ad_id","proba"
    # output is a pd.Dataframe "display_id","ad_id"
    #################################

    result['display_id'] = result['display_id'].astype(int)
    result['ad_id'] = result['ad_id'].astype(int)
    result.sort_values(['display_id','proba'],inplace=True,ascending=False)
    output=result.groupby(['display_id'])['ad_id'].apply(lambda x:' '.join(map(str,x))).reset_index()
    output.to_csv(filename,index=False)

def encode(string):
    string = list(string)
    code = ''
    for char in sting:
        code += str(ord(char)-48)
    return int(code)

#def sort_ffm(filename,output):
    #p = np.loadtxt(filename)
    #test = pd.read_csv('../data/outbrain/clickes_test.csv')
    #test['prob'] = pd.DataFrame(data=p)
    #del p 
    #probSort(test,output)
def replaceProb(x):
    if x[0] == 1.0:
        return 1.0
    else:
        return x[1]

def leak_solution(filename,output):
    leak = pd.read_csv('../data/leakage.csv',usecols=['display_id','ad_id','prob'])
    print("Loading test result...")
    clicked_proba = pd.read_csv(filename)
    leak['predict_proba'] = clicked_proba['prob']
    print("Replacing...")
    leak['new_proba'] = leak[['prob','predict_proba']].apply(lambda x: replaceProb(x),axis=1)
    print("leak : ",leak.head())
    print("Saving leak")
    leak.to_csv("leak_ffm.csv")
    print("Sorting and output...")
    leak = leak[['display_id','ad_id','new_proba']]
    leak.columns = ['display_id','ad_id','proba']
    probSort(leak,output)

def ffm2final(prediction):
    a  = pd.read_csv('../data/clicks_test.csv')
    p = np.loadtxt(prediction)
    a['prob'] = pd.DataFrame(data=p,columns=['prob'])

    a.to_csv("ffm_clicked_proba.csv")
    leak_solution("ffm_clicked_proba.csv","ffm_"+sys.argv[1]+".csv")

    # probSort(a, "ffm_"+sys.argv[1]+".csv")

# p = np.loadtxt('valid_pred5.txt')
# v = pd.read_csv('v.csv', usecols = ['display_id', 'ad_id'])
# v['proba'] = pd.DataFrame(p)


# print validation(v)
ffm2final(sys.argv[1])
# print("Reading data...")
# X_train = pd.read_csv('../data/outbrain/train_ad_cat_topic_likelihood.csv')
# Y_train = pd.read_csv('../data/outbrain/train_ad_clicked.csv',dtype=int)
# Y_train.columns=['clicked']
# print("merging data...")
# train = pd.concat([X_train,Y_train],axis=1)
# print(train.head())
# del X_train
# del Y_train
# train, valid = getVal(train)
# print("Saving data...")
# train.to_csv('../data/outbrain/train_ad_topic_likelihood.csv',index=False)
# valid.to_csv('../data/outbrain/val_ad_topic_likelihood.csv',index=False)

