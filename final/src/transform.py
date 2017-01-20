import pandas as pd
import numpy as np
import time
import hashlib

def hash(input):
    return int(hashlib.md5(input))

def encode(string):
    try:
        string = list(string)

        code = ''
        for char in string:
            code += str(ord(char)-48)
        return int(code)%1000000
    except:
        return 0

start = time.time()

print ("Reading csv...")
train = pd.read_csv('../data/train_final.csv', usecols=['clicked', 'campaign_id', 'advertiser_id', 'uuid', 'platform', 'geo_location', 'likelihood', 'source_id','publisher_id', 'category_id', 'topic_id', 'ad_category_id','ad_topic_id'])

train = train.fillna(0)
train[['source_id', 'publisher_id']] = train[['source_id', 'publisher_id']].astype(int)
train['uuid'] = train['uuid'].apply(lambda x: encode(x))
train['geo_location'] = train['geo_location'].apply(lambda x: encode(x))

train.to_csv('train_encode.csv',index=False)
del train

test = pd.read_csv('../data/test_final.csv', usecols=['campaign_id', 'advertiser_id', 'uuid', 'platform', 'geo_location', 'likelihood','source_id', 'publisher_id', 'category_id', 'topic_id', 'ad_category_id','ad_topic_id'])
test = test.fillna(0)
test[['source_id', 'publisher_id']] = test[['source_id', 'publisher_id']].astype(int)
test['uuid'] = test['uuid'].apply(lambda x : encode(x))
test['geo_location'] = test['geo_location'].apply(lambda x : encode(x))
test.to_csv('test_encode.csv',index=False)


end = time.time()
total = end-start
second = total%60
minute = int(total/60%60)
hour = int(total/60/60)
print(hour ,"hours, ", minute, "minutes, ", second, "seconds.")

