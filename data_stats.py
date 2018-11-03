#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 11:38:46 2018

@author: berend
"""

## stats

### load data and analyze
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import sys

import data_preprocessing as dp
from sklearn.feature_extraction.text import CountVectorizer




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
    
    #easier for copying to  output files
    f = open('out.txt', 'w')
    original = sys.stdout
    sys.stdout = Tee(sys.stdout, f)  
    
    
    datapath = 'data/'
    
    
    #when working in the terminal, don't reload data if it's already there
    try:
        type(data[0])
    except NameError:
        data = dp.loaddata(datapath)

    # data preprocessing:
    ########################################
    print('Total length of data: {}'.format(len(data)))

    ## read a couple:
    print(data[0],'\n\n')
    print(data[1],'\n\n')
    print(data[2],'\n\n')
    
   

    
    ##abstract lengths:
    abs_length = np.array([len(entry['abstract']) for entry in data])
    print('Mean abstract character length: {:0.1f}\n\n'.format(np.mean(abs_length)))
    
    num_authors = np.array([len(entry['authors']) for entry in data])
    print('Mean number of authors: {:0.1f}\n\n'.format(np.mean(num_authors)))
    
    #check what features the entries have and how often they occur
    features = []
    occurences = []
    for entry in data:
        for key in entry.keys():
            if not key in features:
                features.append(key)
                occurences.append(0)
                
            occurences[features.index(key)] += 1

    #sort:
    features = [feature for _,feature in reversed(sorted(zip(occurences,features)))]
    occurences = list(reversed(sorted(occurences)))

    for feature,occurence in zip(features,occurences):
        print('Feature %s occurs %s times' % (feature, occurence))
        
    print('\n\n')
        
        
    #analyze labels:        
    journal_refs = []
    dois = []
    categories = []
    numcategories = []
    unique_categories = []
    unique_categories_occurences = []

    for entry in data:
        try:
            journal_ref = entry['journal-ref']
        except KeyError:
            journal_ref = None
            
        try:
            doi = entry['doi']
        except KeyError:
            doi = None
            
        journal_refs.append(journal_ref)
        dois.append(doi)
        cat = entry['categories'].split(' ')
        categories.append(cat)
        numcategories.append(len(cat))
        
        for c in cat:
            if not c in unique_categories:
                unique_categories.append(c)
                unique_categories_occurences.append(0)
            unique_categories_occurences[unique_categories.index(c)] += 1
        
    
    unique_categories = [cat for _,cat in reversed(sorted(zip(unique_categories_occurences,unique_categories)))]
    unique_categories_occurences = list(reversed(sorted(unique_categories_occurences)))

    print('There are {} unique categories'.format(len(cat)))
    for cat,occ in zip(unique_categories,unique_categories_occurences):
        print('Category %s occurs %s times' % (cat, occ))

                    
        
    #the original sorting is a little odd, but basically it starts in 2007 with the new
    # id labels, and then somewhere at 150k, you return to 1996 and get the rest of the articles        
    ids = []
    for entry in data:
        ids.append(entry['id'])
        
    
        
    
    #first: check which is the better feature to parse, doi or journal-ref?
    #encode: 'both' if article has both, 'ref', 'doi' or 'none'
    which_journal_ref = []
    for entry in data:
        keys = entry.keys()
        if 'doi' in keys and 'journal-ref' in keys:
            which_journal_ref.append('both')
        elif 'doi' in keys and not 'journal-ref' in keys:
            which_journal_ref.append('doi')
        elif not 'doi' in keys and 'journal-ref' in keys:
            which_journal_ref.append('ref')      
        else:
            which_journal_ref.append('none')
        
    print('\n\n')
    journal_ref_types = ['both','doi','ref','none']
    ref_type_counts = [which_journal_ref.count(ref_type) for ref_type in journal_ref_types]
    for ref_type_count,ref_type in zip(ref_type_counts,journal_ref_types):
        print(ref_type,ref_type_count)
    print('\n\n')
        
        

    ###########  
    #now foucssing more at the test at hand, predicting sub category from abstract:
    ##########Load and convert train and test data:
    #(requires having run split_train_test.py)
    trainpath = 'train_data/train_data.json'
    testpath = 'test_data/test_data.json'
    traindata,testdata = dp.loadfile(trainpath),dp.loadfile(testpath)                
        
    n_inc = 12 #number of included categories
    inc_categories = list(unique_categories[:n_inc])        
    print('Including {} unique categories'.format(len(inc_categories)))
    for cat,occ in zip(inc_categories,unique_categories_occurences[:n_inc]):
        print('Category %s occurs %s times' % (cat, occ))
   
        
    train_X,train_y = dp.generate_Xy_data_categories(traindata, inc_categories, ignore_others = False, 
                                shuffle_seed = 0, ydatatype = 'onehot',
                                clean_x = True, keep_latex_tags = True)
    test_X,test_y = dp.generate_Xy_data_categories(testdata, inc_categories, ignore_others = False, 
                                shuffle_seed = 0, ydatatype = 'onehot',
                                clean_x = True, keep_latex_tags = True)
    
    #check if train/test data is balanced:
    train_occ = np.mean(train_y, axis = 0)
    print('\n\nRelative category occurences in train data are:')
    for i,l in enumerate(inc_categories):
        print('Category {} occurence {:.2f}'.format(l,train_occ[i]))
    print('\nMean number of categories in train data is {}'.format(np.mean(np.sum(train_y,axis = 1))))
    print('Median number of categories in train data is {}'.format(np.median(np.sum(train_y,axis = 1))))
    print('Min number of categories in train data is {}'.format(np.min(np.sum(train_y,axis = 1))))
    print('Max number of categories in train data is {}'.format(np.max(np.sum(train_y,axis = 1))))

    test_occ = np.mean(test_y, axis = 0)
    print('\n\nRelative category occurences in test data are:')
    for i,l in enumerate(inc_categories):
        print('Category {} occurence {:.2f}'.format(l,train_occ[i]))
    print('\nMean number of categories in test data is {}'.format(np.mean(np.sum(test_y,axis = 1))))
    print('Median number of categories in test data is {}'.format(np.median(np.sum(test_y,axis = 1))))
    print('Min number of categories in test data is {}'.format(np.min(np.sum(test_y,axis = 1))))
    print('Max number of categories in test data is {}'.format(np.max(np.sum(test_y,axis = 1))))
         
    
    #load count vectorizer:
    cv = CountVectorizer()
    cvfit = cv.fit_transform(train_X + test_X)
    wcounts = np.sum(cvfit, axis = 0).A1    
    words = np.array(cv.get_feature_names())[wcounts.argsort()][::-1]
    wcounts = wcounts[wcounts.argsort()][::-1]
        
    
    #show: most occuring words
    print('\n\nMost occuring words in corpus:')
    for i in range(50):
        print('Word \'{}\' occurs {} times'.format(words[i],wcounts[i]))
        
    
    #show: most occuring words excluding stopwords
    #show: logplot of words vs occurences (compare to imdb dataset), indicate some words with annotations
    #show: Show a run without latex tags, show without stopwords
    #show highest occuring latex tag:
    #print: words with more than N occurences (N e [1e6, 1e5, 1e4, 1e3, 1e2,1e1, etc.])

    #histogram of occurence on x-axis, vs number of features on y-axis
        

#    f.close()