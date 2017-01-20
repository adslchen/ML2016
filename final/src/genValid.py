import pandas ad pd

train = pd.read_csv("train_encode.csv")
train.head(74708517).to_csv("train.csv", index=False)
train.tail(12433214).to_csv("valid.csv", index=False)
