import numpy as np
import pandas as pd

def correlation(df, cols_2_ignore=[]):
    return np.abs(df.drop(cols_2_ignore, axis=1).corr())
    
    
def df_derived_by_shift(df,lag=0,ignore_columns=[]):
    df = df.copy()
    if not lag:
        return df
    cols ={}
    for i in range(1,lag+1):
        for x in list(df.columns):
            if x not in ignore_columns:
                if not x in cols:
                    cols[x] = ['{}_{}'.format(x, i)]
                else:
                    cols[x].append('{}_{}'.format(x, i))
    for k,v in cols.items():
        columns = v
        dfn = pd.DataFrame(data=None, columns=columns, index=df.index)    
        i = 1
        for c in columns:
            dfn[c] = df[k].shift(periods=i)
            i+=1
        df = pd.concat([df, dfn], axis=1)#, join_axes=[df.index])
        df.reset_index(inplace=True, drop=True) 

    return df
    
    
class CorrelationStudy(object):
    def __init__(self, corr):
        self.corr = corr
        self.dict_corr = {}
        self.threshold = None
        self.corr_drops=[]
    
    def calc_candidates_to_drop(self, threshold):
        self.threshold=threshold
        self.apply_threshold(threshold)
        self.discard_duplicated()
        self.create_candidate_list()
        print (len(self.corr_drops),"/",len(self.corr.keys()))
        return self.corr_drops

    def apply_threshold(self,threshold):
        self.dict_corr = {}
        for i, row in enumerate(self.corr):
            for i, col in enumerate(self.corr):
                if row!=col and abs(self.corr[row][col])>threshold:
                    #print (row, col, corr[row][col])
                    if not row in self.dict_corr:
                        self.dict_corr[row]=set()
                    self.dict_corr[row].add(col)
        return self.dict_corr

    def discard_duplicated(self):
        values = list(self.dict_corr.keys())
        for key in values:
            for val in self.dict_corr.get(key,[]):
                if val in self.dict_corr:
                    del self.dict_corr[val]
        
    def create_candidate_list(self):
        corr_drops = set()
        for key in self.dict_corr:
            corr_drops.update(self.dict_corr[key])
        self.corr_drops=list(corr_drops)