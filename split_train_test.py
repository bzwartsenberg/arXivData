#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 14:27:21 2018

@author: berend
"""
import numpy as np
from data_preprocessing import loaddata
import json


def split_train_and_test(data, ratio, seed = None):
    """Split train and test set"""
    
    if seed is not None:
        np.random.seed(seed)
        
        
    p = np.random.permutation(len(data))
    
    split_n = int(ratio*p.shape[0])
    
    traindata = [data[i] for i in p[0:split_n]]
    testdata = [data[i] for i in p[split_n:]]
    
        
    return traindata,testdata
    
    
    
def write_data(data,path):
    """write data to path"""
    
    with open(path,'w') as f:
        json.dump(data,f)
        
        
if __name__ == "__main__":
    
    datapath = 'data/'
    
    
    train_out_path = 'train_data/train_data.json'
    test_out_path = 'test_data/test_data.json'
    
    
    
    data = loaddata(datapath)
    
    seed = 0
    ratio = 0.6
    
    traindata,testdata = split_train_and_test(data, ratio, seed)
    
    write_data(traindata,train_out_path)
    write_data(testdata,test_out_path)
    
    