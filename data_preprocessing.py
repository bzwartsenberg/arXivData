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
    
    
def cleandata():
    """"""
        
        

if __name__ == "__main__":
    
    try:
        type(data[0])
    except NameError:
        data = loaddata(datapath)

    #do some analysis:

    ## read a couple:
    print(data[10])

    
    ##abstract lengths:
    abs_length = np.array([len(entry['abstract']) for entry in data])
    print('Mean abstract character length: %0.1f' % np.mean(abs_length))
    
    num_authors = np.array([len(entry['authors']) for entry in data])
    print('Mean number of authors: %0.1f' % np.mean(num_authors))
    
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

        
        