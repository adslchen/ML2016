import pandas as pd
#after merging: display_id, ad_id, clicked, ad_document_id, campaign_id, advertiser_id
train = pd.read_csv("../data/clicks_train.csv")
ad = pd.read_csv("../data/promoted_content.csv")
train = pd.merge(train, ad, on='ad_id', how='left', sort=False)
train.columns = ['display_id', 'ad_id', 'clicked', 'ad_document_id', 'campaign_id', 'advertiser_id']
del ad
print"train_ad"
print train.head(10)

#after merging: clicked, ad_document_id, campaign_id, advertiser_id, uuid, document_id, timestamp, platform, geo_location, likelihood
events = pd.read_csv("../data/events.csv", usecols=['display_id', 
    'uuid', 'document_id', 'timestamp', 'platform'])
events['platform'] = events['platform'].replace('\\N', 0)
events['platform'] = events['platform'].astype(int)
train = pd.merge(train, events, on='display_id', how='left', sort=False)
del events
print "train_ad_event"
print train.head(10)

geo = pd.read_csv("../data/display_geo.csv", usecols=['display_id', 'geo_location'])
train = pd.merge(train, geo, on='display_id', how='left', sort=False)
del geo
del train['display_id']
print "train_ad_event_geo"
print train.head(10)

likelihood = pd.read_csv("../data/ad_likelihood.csv")
train = pd.merge(train, likelihood, on='ad_id', how='left', sort=False)
del likelihood
del train['ad_id']
print "train_ad_event_geo_likelihood"
print train.head(10)


#after merging: clicked, campaign_id, advertiser_id, uuid, timestamp, platform, geo_location, likelihood, source_id, publisher_id, category_id, topic_id, ad_category_id, ad_topic_id
meta = pd.read_csv("../data/documents_meta.csv")
train = pd.merge(train, meta, on='document_id', how='left', sort=False)

cat_topic = pd.read_csv("../data/doc_category_topic.csv")
train = pd.merge(train, cat_topic, on='document_id', how='left', sort=False)
del train['document_id']
print "train_cat_topic"
print train.head(10)

cat_topic.columns=['ad_document_id', 'ad_category_id', 'ad_topic_id']
train = pd.merge(train, cat_topic, on='ad_document_id', how='left', sort=False)
del cat_topic
del train['ad_document_id']
print "train_ad_cat_topic"
print train.head(10)

train.to_csv('data/train_final.csv', index=False)

