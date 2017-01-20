import pandas as pd

event = pd.read_csv("../data/events.csv", usecols=['display_id', 'geo_location'])

for index , row in event.iterrows():
    #print row['geo_location']
    temp = row['geo_location']
    if type(temp) is not float: 
        temp = temp.split('>')[0]
    else:
        continue
    event.set_value(index, 'geo_location', temp)

event.to_csv("../data/display_geo.csv")
