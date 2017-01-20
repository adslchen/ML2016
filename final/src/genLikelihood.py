import pandas as pd
import numpy as np


train = pd.read_csv('../data/clicks_train.csv')
ad_likelihood = train.groupby('ad_id').clicked.agg(['count','sum','mean']).reset_index()
M = train.clicked.mean()
#del train

ad_likelihood['likelihood'] = (ad_likelihood['sum'] + 12*M) / (12 + ad_likelihood['count'])
print(ad_likelihood.head())
ad_likelihood = ad_likelihood[['ad_id','likelihood']]

ad_likelihood.to_csv("../data/ad_likelihood.csv", index=False)
