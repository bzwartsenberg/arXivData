#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 12:22:06 2018

@author: berend
"""

### load data and analyze
import json
import os
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from sklearn.preprocessing import OneHotEncoder


datapath = 'data/'

    


def loadfile(filename):
    """Load file in filename"""
    
    with open(filename, 'r') as f:
        
        doclist = json.load(f)
    
    
    return doclist
    
    
def loaddata(datapath):
    """Load all files in datapath"""
    
    data = []
    for fpath in os.listdir(datapath):
        if '.json' in fpath:
            data += loadfile(datapath + fpath)
        
    return data
    
    
def remove_latex(text, keep_tags = True):
    """Remove all latex tags from a text
    Args:
        text: the text to be filtered
        keep_tags: keep the tags as tokens (add spaces around it)"""
    


    ###conversions to make:
    #1. '\n' -> ' '
    text = text.replace('\n', ' ')

    #2. match '\'" and '\"'s
    pattern = r'\\[^a-zA-Z\d\s]'    
    text = re.sub(pattern, '', text)    
    
    
    
    #3. '\tag' -> ' ' or '\tag' -> ' \tag '
    # define tag: starts with '\\', ends with ' ', '(', ')', '{', '}'   
    pattern = r'\\[a-zA-Z]*(?=[\s=\^,_\.-\{\}\(\)\?\*\[\]\!\\$])'
    #r'\\[a-zA-Z]*(?=)'
    if keep_tags:
        text = re.sub(pattern, r' \g<0> ', text) 
    else:
        text = re.sub(pattern, '', text) 

    
    #4. '$_{' --> ''
    # '$^{' --> ''
    # '_'
    # '_{' --> ''
    # '$' --> ''
    
    text = text.replace('$','').replace('{','').replace('}','').replace('_','').replace('^','')
    
    return text
    
    
def cleandata(data_X, keep_tags = True):
    """"""
    
    clean_data_X = []
    for text in data_X:
        text = text.lower()
        text = remove_latex(text, keep_tags = keep_tags)
        clean_data_X.append(text)
    
    return clean_data_X



def tokenize(text, filterset = None):
    """Tokenize and filter a text sample.
    
    Args:
        text string to be tokenized and filtered.
        filterset: set of words that are filtered out

    returns:
        tokens: a list of the tokens/words in text.
    """
    ## NOTE: maybe update this function with a different nltk tokenizer that automatically removes interpunction and stopwords
    
    tokens = word_tokenize(text)
    
    if filterset == 'std':
        #nltk filter stopwords
        filterset = set(stopwords.words('english'))
        #nltk filter punctuation...
        filterset = filterset.union(set(['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', 
        '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', 
        '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '\'s']))
    
        
    if filterset is not None:
        tokens = [token for token in tokens if not token in filterset]
    
    return tokens
    
    
    
def generate_Xy_data_categories(data, inc_categories, ignore_others = True, 
                                shuffle_seed = None, ydatatype = 'onehot',
                                clean_x = True, keep_latex_tags = True):
    """Generate data_X and data_y from data, with y labels from categories
    Args:
        data: list of entries
        inc_categories: list of categories to include
        ignore_others: if set to True, no "others" category will be made

        """
               
    data_X, data_y = [],[]    

    if not ignore_others:
        if not 'other' in inc_categories:
            inc_categories.append('other')
                        
    for entry in data:
        abstract = entry['abstract']
        data_X.append(abstract)

        categories = entry['categories'].split(' ')
    
        if ignore_others:
            categories = [cat for cat in categories if cat in inc_categories]
        else:
            categories = [cat if cat in inc_categories else 'other' for cat in categories]
        
        if ydatatype == 'onehot':
            data_y.append(np.zeros((len(inc_categories))))
        else:
            data_y.append([])
        
        for cat in categories:
            if ydatatype == 'catnum':
                data_y[-1].append(inc_categories.index(cat))
            elif ydatatype == 'cat':
                data_y[-1].append(cat)
            elif ydatatype == 'onehot':
                #automatically avoids double categories
                data_y[-1][inc_categories.index(cat)] = 1.0

            
    if shuffle_seed is not None:
        np.random.seed(shuffle_seed)
        p = np.random.permutation(len(data_X))
        
        data_X = [data_X[i] for i in p]
        data_y = [data_y[i] for i in p]

    # convert to numpy array if onehot
    if ydatatype == 'onehot':
        data_y = np.array(data_y)
    if clean_x:
        data_X = cleandata(data_X, keep_tags = keep_latex_tags)
        
    return data_X, data_y

        
        

if __name__ == "__main__":
    
     
    trainpath = 'train_data/train_data.json'
    testpath = 'test_data/test_data.json'
    traindata,testdata = loadfile(trainpath),loadfile(testpath)
#
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
                         'hep-th']
#    
    train_X,train_y = generate_Xy_data_categories(traindata, inc_categories, ignore_others = False, 
                                shuffle_seed = 0, ydatatype = 'onehot',
                                clean_x = True, keep_latex_tags = True)
