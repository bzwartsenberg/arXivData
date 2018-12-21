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
from nltk.corpus import stopwords

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
    print('Median abstract character length: {:0.1f}\n\n'.format(np.median(abs_length)))
    print('Min abstract character length: {:0.1f}\n\n'.format(np.min(abs_length)))
    print('Max abstract character length: {:0.1f}\n\n'.format(np.max(abs_length)))
    
    num_authors = np.array([len(entry['authors']) for entry in data])
    print('Mean number of authors: {:0.1f}\n\n'.format(np.mean(num_authors)))
    print('Median number of authors: {:0.1f}\n\n'.format(np.median(num_authors)))
    print('Min number of authors: {:0.1f}\n\n'.format(np.min(num_authors)))
    print('Max number of authors: {:0.1f}\n\n'.format(np.max(num_authors)))
    
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

    print('There are {} unique categories'.format(len(unique_categories)))
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
        
    n_inc = 16 #number of included categories
    inc_categories = list(unique_categories[:n_inc])        
    print('Including {} unique categories'.format(len(inc_categories)))
    cat_occ = {}
    for cat,occ in zip(inc_categories,unique_categories_occurences[:n_inc]):
        cat_occ[cat] = occ
        print('Category %s occurs %s times' % (cat, occ))
    cat_occ['others'] = sum(unique_categories_occurences[n_inc:])
    print('Others: ', cat_occ['others'])
    
    labels = ['others'] + inc_categories
    values = [cat_occ[l] for l in labels]
    
    fig,ax = plt.subplots(figsize = (6,3))
    scolors = palettable.cartocolors.qualitative.Prism_7.mpl_colors
    ax.pie(values, labels = labels, colors = scolors,autopct='%1.1f%%', explode = np.linspace(0.0,0.7,len(values)))
    plt.subplots_adjust(left=0.25, right=0.65, top=0.9, bottom=0.1)
    fig.savefig('Pie_categories.png', dpi = 300)
    plt.show()
    
    
    #lt is "keep_latex_tags"
    #rn is "run_name"
    #sw is "stopwords"
    #tk is tokeninzer
    
    print('\n\nStandard NLTK stopwords: ')
    for word in stopwords.words('english'):
        print(word)
            
    
#    lt_list = [ True, False,True,True]
#    sw_list = [stopwords.words('english'),None,None,None]
#    tk_list = [None,None,dp.tokenize,None]
    lt_list = [True]
    sw_list = [None]
    tk_list = [None]
    wcounts_dict = {}
    words_dict = {}
#    for lt, sw,tk,rn in zip(lt_list,sw_list,tk_list,['No Stopwords','No latex tags','with dp.tokenize','All']):
    for lt, sw,tk,rn in zip(lt_list,sw_list,tk_list,['All']):
        print('\n\n########################\n')
        print('Now doing run for ', rn)
            
        train_X,train_y = dp.generate_Xy_data_categories(traindata, inc_categories, ignore_others = False, 
                                    shuffle_seed = 0, ydatatype = 'onehot',
                                    clean_x = True, keep_latex_tags = lt)
        test_X,test_y = dp.generate_Xy_data_categories(testdata, inc_categories, ignore_others = False, 
                                    shuffle_seed = 0, ydatatype = 'onehot',
                                    clean_x = True, keep_latex_tags = lt)
        
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
        cv = CountVectorizer(tokenizer = tk, stop_words = sw)
        cvfit = cv.fit_transform(train_X)
        wcounts = np.sum(cvfit, axis = 0).A1    
        words = np.array(cv.get_feature_names())[wcounts.argsort()][::-1]
        wcounts = wcounts[wcounts.argsort()][::-1]
            
        
        #show: most occuring words
        print('\n\nMost occuring words in corpus:')
        for i in range(50):
            print('Word \'{}\' occurs {} times'.format(words[i],wcounts[i]))
            
        #show highest occuring latex tag:
        for i in range(words.shape[0]):
            if '\\' in words[i]:
                print(i, words[i], wcounts[i])
                break
            
        print('\n\nWord occurence info')
        print('Mean occurence: {}'.format(np.mean(wcounts)))
        print('Median occurence: {}'.format(np.median(wcounts)))
        print('Min occurence: {}'.format(np.min(wcounts)))
        print('Max occurence: {}'.format(np.max(wcounts)))
        
        
        ##print min token length, max token length, mean and median:
        wordlengths = np.char.str_len(words)
        print('\n\nToken length')        
        print('Median token length: ',np.median(wordlengths))
        print('Mean token length: ',wordlengths.mean())
        print('Max token length: ',wordlengths.max())
        print('Min token length: ',wordlengths.min())
        print('\n\nLongest words: ')
        for i in range(20):
              print(words[np.argsort(wordlengths)[-i-1]])
            
            
        
        #show: Show a run without latex tags, show without stopwords
    
        #print: words with more than N occurences (N e [1e6, 1e5, 1e4, 1e3, 1e2,1e1, etc.])
        print('\n\nWord occurences higher than x: ')
        for cut in [1e6, 1e5,1e4,1e3,1e2,1e1,1e0]:
            times = np.argwhere(wcounts > cut).shape[0]
            print('There are {} tokens that occur more than {} times'.format(times,cut))
        
        wcounts_dict[rn] = wcounts
        words_dict[rn] = words

        
    #show: logplot of words vs occurences (compare to imdb dataset), indicate some words with annotations
    scolors = palettable.cartocolors.qualitative.Prism_8.mpl_colors    
    fig,ax = plt.subplots(figsize = (4,3))
    for i,key in enumerate(wcounts_dict.keys()):
        ax.scatter(np.arange(wcounts_dict[key].shape[0]),wcounts_dict[key], c=scolors[2*i], alpha=1.0, s = 3, label = key)
    ax.set_yscale('log')
    ax.set_xlabel('token number')
    ax.set_xticks(np.arange(0,300000,50000))
    ax.set_xticklabels(['${:.1f}\cdot10^5$'.format(v) for v in np.arange(0,300000,50000)/100000])
    ax.legend()
    fig.savefig('token_stats_vs_methods')
    #histogram of occurence on x-axis, vs number of features on y-axis
    
    
    #big-grams, first 2,2
    params = dict(strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None, 
         stop_words=stopwords.words('english'), ngram_range=(2,2), analyzer='word', 
         max_df=1.0, min_df=5, max_features=None, vocabulary=None)
    
    cv = CountVectorizer(**params)
    cvfit = cv.fit_transform(train_X)
    wcounts = np.sum(cvfit, axis = 0).A1        
    words = np.array(cv.get_feature_names())[wcounts.argsort()][::-1]
    wcounts = wcounts[wcounts.argsort()][::-1]
    print('\n\nMost occuring 2-grams:')
    for i in range(50):
        print('Word \'{}\' occurs {} times'.format(words[i],wcounts[i]))
        
    print('\n\nWord occurences higher than x for 2-gram: ')
    for cut in [1e6, 1e5,1e4,1e3,1e2,1e1,1e0]:
        times = np.argwhere(wcounts > cut).shape[0]
        print('There are {} tokens that occur more than {} times'.format(times,cut))
        


    params = dict(strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None, 
         stop_words=stopwords.words('english'), ngram_range=(3,3), analyzer='word', 
         max_df=1.0, min_df=5, max_features=None, vocabulary=None)
    
    cv = CountVectorizer(**params)
    cvfit = cv.fit_transform(train_X)
    wcounts = np.sum(cvfit, axis = 0).A1        
    words = np.array(cv.get_feature_names())[wcounts.argsort()][::-1]
    wcounts = wcounts[wcounts.argsort()][::-1]
    print('\n\nMost occuring 3-grams:')
    for i in range(50):
        print('Word \'{}\' occurs {} times'.format(words[i],wcounts[i]))

    print('\n\nWord occurences higher than x for 3-gram: ')
    for cut in [1e6, 1e5,1e4,1e3,1e2,1e1,1e0]:
        times = np.argwhere(wcounts > cut).shape[0]
        print('There are {} tokens that occur more than {} times'.format(times,cut))


    params = dict(strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None, 
         stop_words=stopwords.words('english'), ngram_range=(4,4), analyzer='word', 
         max_df=1.0, min_df=5, max_features=None, vocabulary=None)
    
    cv = CountVectorizer(**params)
    cvfit = cv.fit_transform(train_X)
    wcounts = np.sum(cvfit, axis = 0).A1        
    words = np.array(cv.get_feature_names())[wcounts.argsort()][::-1]
    wcounts = wcounts[wcounts.argsort()][::-1]
    print('\n\nMost occuring 4-grams:')
    for i in range(50):
        print('Word \'{}\' occurs {} times'.format(words[i],wcounts[i]))
    print('\n\nWord occurences higher than x for 4-gram: ')
    for cut in [1e6, 1e5,1e4,1e3,1e2,1e1,1e0]:
        times = np.argwhere(wcounts > cut).shape[0]
        print('There are {} tokens that occur more than {} times'.format(times,cut))
        
              
           
#    f.close()