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
    
#    try:
#        type(data[0])
#    except NameError:
#        data = loaddata(datapath)
#
#    #do some analysis:
#
#    ## read a couple:
#    print(data[10])
#
#    
#    ##abstract lengths:
#    abs_length = np.array([len(entry['abstract']) for entry in data])
#    print('Mean abstract character length: %0.1f' % np.mean(abs_length))
#    
#    num_authors = np.array([len(entry['authors']) for entry in data])
#    print('Mean number of authors: %0.1f' % np.mean(num_authors))
#    
#    #check what features the entries have and how often they occur
#    features = []
#    occurences = []
#    for entry in data:
#        for key in entry.keys():
#            if not key in features:
#                features.append(key)
#                occurences.append(0)
#                
#            occurences[features.index(key)] += 1
#
#    #sort:
#    features = [feature for _,feature in reversed(sorted(zip(occurences,features)))]
#    occurences = list(reversed(sorted(occurences)))
#
#    for feature,occurence in zip(features,occurences):
#        print('Feature %s occurs %s times' % (feature, occurence))
#        
#        
#    #analyze labels:e        
#    journal_refs = []
#    dois = []
#    categories = []
#    numcategories = []
#    unique_categories = []
#    unique_categories_occurences = []
#
#    for entry in data:
#        try:
#            journal_ref = entry['journal-ref']
#        except KeyError:
#            journal_ref = None
#            
#        try:
#            doi = entry['doi']
#        except KeyError:
#            doi = None
#            
#        journal_refs.append(journal_ref)
#        dois.append(doi)
#        cat = entry['categories'].split(' ')
#        categories.append(cat)
#        numcategories.append(len(cat))
#        
#        for c in cat:
#            if not c in unique_categories:
#                unique_categories.append(c)
#                unique_categories_occurences.append(0)
#            unique_categories_occurences[unique_categories.index(c)] += 1
#        
#    
#    unique_categories = [cat for _,cat in reversed(sorted(zip(unique_categories_occurences,unique_categories)))]
#    unique_categories_occurences = list(reversed(sorted(unique_categories_occurences)))
#
#    for cat,occ in zip(unique_categories,unique_categories_occurences):
#        print('Feature %s occurs %s times' % (cat, occ))
#                    
#        
#    #the original sorting is a little odd, but basically it starts in 2007 with the new
#    # id labels, and then somewhere at 150k, you return to 1996 and get the rest of the articles        
#    ids = []
#    for entry in data:
#        ids.append(entry['id'])
#        
#    
#        
#    ### to do: analyze labels on journal   
#    
#    #first: check which is the better feature to parse, doi or journal-ref?
#    #encode: 'both' if article has both, 'ref', 'doi' or 'none'
#    which_journal_ref = []
#    for entry in data:
#        keys = entry.keys()
#        if 'doi' in keys and 'journal-ref' in keys:
#            which_journal_ref.append('both')
#        elif 'doi' in keys and not 'journal-ref' in keys:
#            which_journal_ref.append('doi')
#        elif not 'doi' in keys and 'journal-ref' in keys:
#            which_journal_ref.append('ref')      
#        else:
#            which_journal_ref.append('none')
#        
#    journal_ref_types = ['both','doi','ref','none']
#    ref_type_counts = [which_journal_ref.count(ref_type) for ref_type in journal_ref_types]
#    for ref_type_count,ref_type in zip(ref_type_counts,journal_ref_types):
#        print(ref_type,ref_type_count)
#    
#
#    ##########Load and convert train and test data:
#    print(unique_categories_occurences[0:12])
#    inc_categories = list(unique_categories[0:12])
#    print('Inc cats: %s' % inc_categories)
#        
#        
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
#        
    ##find some latex tags:
#    orig = []
#    after = []
#    find_n = 200
#    i = 0
#    for a in train_X:
#        
#        if '\\' in a:
#            i += 1
#            orig.append(a)
#            after.append(remove_latex(a, keep_tags = False))
#            if i >= find_n:
#                break
#            
#    clean_train_X = [remove_latex(text, keep_tags = False) for text in train_X[0:1000]]
#        
    #train_X[102231] # has latex tags
    
#    clean_train_X = cleandata(train_X, keep_tags = True)
#    
#    run_word_diagnostics = True
#    if run_word_diagnostics:
#        #some diagnostics:
#            
#        all_words = [word for text in clean_train_X for word in text]        
#            
#        counts = Counter(all_words)
#        
#        counts_sort = list(reversed(sorted(counts.items(),key=lambda x:x[1])))
#
#        print('Total unique words = %s' % len(counts))
#        print('Most occuring tokens are are:\n'  + '\n'.join([str(x) for x in counts_sort[0:10]]) + '\n\n')
#        freqs = [x[1] for x in counts_sort]
#        for i in range(1,20):
#            print('There are %s of words with more than %s occurences' % (freqs.index(i),i))
#            
#        #create dictionary with occurences:
#        word_occurences = {}
#        [exec('word_occurences[a[0]] = a[1]') for a in counts_sort]
#                        
