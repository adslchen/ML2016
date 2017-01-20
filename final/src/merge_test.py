import pandas as pd
#after merging: display_id, ad_id, clicked, ad_document_id, campaign_id, advertiser_id
test = pd.read_csv("../data/clicks_test.csv")
ad = pd.read_csv("../data/promoted_content.csv")
test = pd.merge(test, ad, on='ad_id', how='left', sort=False)
test.columns = ['display_id', 'ad_id', 'ad_document_id', 'campaign_id', 'advertiser_id']
del ad
print"test_ad"
print test.head(10)

#after merging: clicked, ad_document_id, campaign_id, advertiser_id, uuid, document_id, timestamp, platform, geo_location, likelihood
events = pd.read_csv("../data/events.csv", usecols=['display_id', 
    'uuid', 'document_id', 'timestamp', 'platform'])
events['platform'] = events['platform'].replace('\\N', 0)
events['platform'] = events['platform'].astype(int)
test = pd.merge(test, events, on='display_id', how='left', sort=False)
del events
print "test_ad_event"
print test.head(10)

geo = pd.read_csv("../data/display_geo.csv", usecols=['display_id', 'geo_location'])
test = pd.merge(test, geo, on='display_id', how='left', sort=False)
del geo
del test['display_id']
print "test_ad_event_geo"
print test.head(10)

likelihood = pd.read_csv("../data/ad_likelihood.csv")
test = pd.merge(test, likelihood, on='ad_id', how='left', sort=False)
del likelihood
del test['ad_id']
print "test_ad_event_geo_likelihood"
print test.head(10)


#after merging: clicked, campaign_id, advertiser_id, uuid, timestamp, platform, geo_location, likelihood, source_id, publisher_id, category_id, topic_id, ad_category_id, ad_topic_id
meta = pd.read_csv("../data/documents_meta.csv")
test = pd.merge(test, meta, on='document_id', how='left', sort=False)

cat_topic = pd.read_csv("../data/doc_category_topic.csv")
test = pd.merge(test, cat_topic, on='document_id', how='left', sort=False)
del test['document_id']
print "test_cat_topic"
print test.head(10)

cat_topic.columns=['ad_document_id', 'ad_category_id', 'ad_topic_id']
test = pd.merge(test, cat_topic, on='ad_document_id', how='left', sort=False)
del cat_topic
del test['ad_document_id']
print "test_ad_cat_topic"
print test.head(10)

test.to_csv('../data/test_final.csv', index=False)

