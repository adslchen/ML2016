import numpy as np
import pandas as pd
import sys, argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("-o",type=str)
parser.add_argument("-i",type=str)
parser.add_argument("-chop",type=bool,default=True)
args = parser.parse_args()
start = time.time()
csvfile = args.i


def toformat(x, **kwargs):
    if isScore==1:
        return str(idx)+':'+'1'+':'+str(x)
    elif isScore==2:
        return str(idx)+':' +str(int(x[0]))+':'+str(x[1])
    else:
        if x==0:
            return str(idx)+':'+'1'+':'+str(0)
        return str(idx)+':'+str(int(x))+':'+'1'


chunksize = 10000
i = 0
chunks=[]
chop = args.chop
#chop = bool(sys.argv[3])
# the name shoud output to be score
score_set = ['likelihood']
# the name should be output to a scores
conf_set = []
filenum = 0
for chunk in pd.read_csv(csvfile, chunksize=chunksize,dtype={'clicked':np.int32}):
    i+=1
    cols = chunk.columns.tolist()
    chunk = chunk.fillna(0.0)
    isScore = 0
    if i%100==0 :
        print("Processing %s",i,"chunks")
        #break
    if 'clicked' in cols:     # if click in the data we should place it on first column
        cols.remove("clicked")
        cols = ['clicked'] + cols
        chunk  = chunk[cols]
        cc = iter(range(len(cols))) # click is on first column (0)
        
    else:
        cc = iter(range(1,len(cols)+1))
    cols = iter(chunk.columns.tolist())
    drop_cols = [] 
    for idx, col in zip(cc,cols):
        #print(col)
        isScore=0
        if col in score_set:
            isScore = 1
        elif col in conf_set:
            try: # if the column is the last , except
                two_col = chunk.columns.tolist()[idx:idx+2]
                drop_cols += [two_col[1]]
                isScore = 2
            except:
                pass
            
        else:
            isScore = 0
        if col != "clicked":
            if isScore == 2:
                chunk[col] = chunk[two_col].apply(toformat,idx=idx,isScore=isScore,axis=1)
            
            else:
                chunk[col] = chunk[col].apply(toformat,idx=idx,isScore=isScore)
    # drop the confience columns
    chunk = chunk.drop(drop_cols,axis=1)
    chunks.append(chunk)   
    if i != 0 and i%1000 == 0 and chop == True:
        print("saving frame ",filenum)
        df=pd.DataFrame()
        df = pd.concat(chunks,ignore_index=True)
        data = df.as_matrix()
        del df
        np.savetxt(args.o+str(filenum)+".txt",data,fmt='%-1s')
        del chunks
        chunks = []
        filenum += 1
df=pd.DataFrame()
df = pd.concat(chunks,ignore_index=True)
data = df.as_matrix()
print("data = ", data[:10])
del df
np.savetxt(args.o+str(filenum)+".txt",data,fmt='%-1s')
    # for i in range(data.shape[0]):
end = time.time()
total = end-start
second = total%60
minute = int(total/60%60)
hour = int(total/60/60)
print(hour ,"hours, ", minute, "minutes, ", second, "seconds.")
        
    
    


