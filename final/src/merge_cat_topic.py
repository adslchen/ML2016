import pandas as pd
cat = pd.read_csv('../data/documents_categories.csv')
top = pd.read_csv('../data/documents_topics.csv')

idx = cat.groupby('document_id').confidence_level.transform(max)==cat['confidence_level']
cat = cat[idx]
del cat['confidence_level']

idx = top.groupby('document_id').confidence_level.transform(max)==top['confidence_level']
top = top[idx]
del top['confidence_level']

cat = pd.merge(cat, top, on='document_id', how='outer')
cat = cat.sort_values('document_id')
cat = cat.fillna(0)
cat = cat.astype(int)
cat.to_csv("../data/doc_category_topic.csv", index=False)
