#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 17:02:57 2018

@author: berend
"""



### load data and analyze
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import sys
from nltk.corpus import stopwords
import pickle

import data_preprocessing as dp
from sklearn.feature_extraction.text import CountVectorizer

import palettable



if __name__ == "__main__":
    
    trainpath = 'train_data/train_data.json'
    traindata = dp.loadfile(trainpath)             

    
    inc_categories =    ['cond-mat.mes-hall',
                     'cond-mat.mtrl-sci',
                     'cond-mat.stat-mech',
                     'cond-mat.str-el',
                     'cond-mat.supr-con',
                     'cond-mat.soft',
                     'quant-ph',
                     'cond-mat.dis-nn',
                     'cond-mat',
                     'cond-mat.quant-gas',
                     'cond-mat.other',
                     'hep-th',
                     'math.MP',
                     'math-ph',
                     'physics.optics',
                     'physics.chem-ph',
                     ]
    



    train_X,train_y = dp.generate_Xy_data_categories(traindata, inc_categories, ignore_others = False, 
                                shuffle_seed = 0, ydatatype = 'onehot',
                                clean_x = True, keep_latex_tags = True)



    load = False
    savename = 'countvectorizer.obj'
    
    if load:
        with open(savename,'rb') as f:
            cv,cvfit = pickle.load(f)
    else:
        cv = CountVectorizer(tokenizer = None, stop_words = None)
        cvfit = cv.fit_transform(train_X)
        with open(savename,'wb') as f:
                pickle.dump((cv,cvfit),f)        
    wcounts = np.sum(cvfit, axis = 0).A1    
    words = np.array(cv.get_feature_names())[wcounts.argsort()][::-1]
    wcounts = wcounts[wcounts.argsort()][::-1]


    ## find correlations:
    #nested for loops are ugly, but otherwise I need a 100k by 100k array in memory
    #2nd note: there may be a better way to do this, need to check sklearn (pandas has a function, but don't know if it works with sparse arrays)
    
    correlations = np.zeros((train_y.shape[1],cvfit.shape[1]))
    var_y = (train_y - np.mean(train_y, axis = 0,keepdims = True))/np.std(train_y, axis = 0, keepdims = True)
    #this takes a while:
    import time
    t0 = time.time()
    for j in range(cvfit.shape[1]):
        if j % 100 == 0:
            t1 = time.time()
            print('Time per feature: {:.2f}'.format((t1-t0)/(j+1)))
            print('Time left: {:.2f}'.format((cvfit.shape[1] - j)*((t1-t0)/(j+1))))
        var_x = cvfit[:,j].todense().A1
        var_x = (var_x - np.mean(var_x))/np.std(var_x)
        var_x = var_x.reshape((-1,1))
        correlations[:,j] = np.mean(var_x*var_y, axis = 0)
        
    cor_savename = 'correlations_array.obj'
    with open(cor_savename,'wb') as f:
            pickle.dump((correlations),f)        
    
     