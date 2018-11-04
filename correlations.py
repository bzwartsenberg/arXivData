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




class Tee():
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() 
    def flush(self) :
        for f in self.files:
            f.flush()




if __name__ == "__main__":
    
    ###for logging:
    f = open('correlations_out.txt', 'w')
    original = sys.stdout
    sys.stdout = Tee(sys.stdout, f)  
    
    
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



    load = True
    savename = 'save/countvectorizer.obj'
    
    if load:
        with open(savename,'rb') as f:
            cv,cvfit = pickle.load(f)
    else:
        cv = CountVectorizer(tokenizer = None, stop_words = None)
        cvfit = cv.fit_transform(train_X)
        with open(savename,'wb') as f:
                pickle.dump((cv,cvfit),f)  
    wcounts = np.sum(cvfit, axis = 0).A1    
    words = np.array(cv.get_feature_names())




        
    cor_savename = 'save/correlations_array.obj'
    if load:
        with open(cor_savename,'rb') as f:
                correlations = pickle.load(f)    
    else:
        ## find correlations:
        #for loop is ugly, but otherwise it uses too much memory
        #also this takes around 2 hours to run on my macbook
        #there may be a better way to do this, need to check sklearn 
        #(pandas has a function, but I don't want to calculate just the square matrix, that would take forever)
        
        #this takes around 2 hours to run on a macbook
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
        
        with open(cor_savename,'wb') as f:
                pickle.dump((correlations),f)        
    
                
    ##analyze correlations:
    for i,cat in enumerate(inc_categories):
        
        print('\n\nFor category ', cat)
        highcor = np.argsort(correlations[i])[-1:-11:-1]
        lowcor = np.argsort(correlations[i])[0:10]
        print('Highest correlations are: ')
        for j in range(10):
            print('Correlation with word {} is {:.2f}, word occurs {} times'.format(words[highcor[j]],correlations[i,highcor[j]],wcounts[highcor[j]]))
        print('\nHighest anti-correlations are: ')
        for j in range(10):
            print('Correlation with word {} is {:.2f}, word occurs {} times'.format(words[lowcor[j]],correlations[i,lowcor[j]],wcounts[lowcor[j]]))
               
    #notes: math-ph and math.PP seem to be the same cat.
    #cond-mat, is that even a subclass? -> yes apparently it is.
    ##
    

        
    #To set the scale, the minimum and maximum absolute correlations are:
    print('Minimum mean absolute correlation: {:.6f}'.format(np.mean(np.abs(correlations),axis = 0).min()))
    print('Maximum mean absolute correlation: {:.6f}'.format(np.mean(np.abs(correlations),axis = 0).max()))
    print('Word with min mean absolute correlation: {}'.format(words[np.mean(np.abs(correlations),axis = 0).argmin()]))
    print('Word with max mean absolute correlation: {}'.format(words[np.mean(np.abs(correlations),axis = 0).argmax()]))
    

    
    #show words with high counts and ubiquitously low correlations: they are stopwords
    #first let's see the average occurence of a 'stopword':
    occ = []
    stop_words = []
    tot_words = wcounts.sum()
    for word in stopwords.words('english'):
        try:
            occ.append(wcounts[cv.vocabulary_[word]]/tot_words)
            stop_words.append(word)
        except KeyError:
            pass
    occ = np.array(occ)
    print('\n\nStop words that occur most often:')
    occ_most = np.argsort(occ)[-1:-4:-1]
    for i in range(3):
        print('The word {} occurs {:.4f} over the corpus'.format(stop_words[occ_most[i]],occ[occ_most[i]]))
    
    print('The median for nltk defined stop words is: {:.4f}'.format(np.median(occ)))
    # so a good limit would seem to be about half the median:
    occ_limit = 0.5*np.median(occ)
    
    
    
    n_stopwords = 100
    ##now we just need a limit for correlation, we can take the correlation 
    # as a mean of the absolute value over all classes, filter by occurence > occ_limit,
    # and then take the n_stopwords lowest correlations
    #define a subset of words that are over teh occcurence limit:
    word_subset = np.argwhere(wcounts/tot_words > occ_limit)[:,0]
    #and their mean absolute correlations:
    cors = np.mean(np.abs(correlations[:,word_subset]),axis = 0)
    #find the minima of that, and translate back to original word idx:
    sw_idx = word_subset[np.argsort(cors)[0:n_stopwords]]
                                
    inferred_stop_words = []
    print('\n\nNow printing inferred stopwords:')
    for i in sw_idx:
        print('Word \'{}\' has an occurence of {:.4f} mean absolute correlation of {:.4f}'.format(words[i],np.mean(np.abs(correlations[:,i])),wcounts[i]/tot_words))
        inferred_stop_words.append(words[i])
    #save these inferred stopwords:
    with open('save/inferred_stop_words.boj','wb') as f:
        pickle.dump(inferred_stop_words,f)
        

    