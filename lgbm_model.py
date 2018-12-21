#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 13:54:26 2018

@author: berend
"""

### train on tfidf vectors of arxiv model:
    
import sys
sys.path.append('/Users/berend/Documents/Coding/ML-projects/ArxivData/')


import numpy as np
#import matplotlib.pyplot as plt

import data_preprocessing as dp

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords


import pickle

import lightgbm as lgb
import pandas as pd

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
     



class lgbmTextClassifier():


    def __init__(self, train_data, test_data, savename = None, run_transform = True,
                 ylabels = None,train_split = 0.66, random_seed = 0, tfidf_params = {},
                 lgbm_params = {}):
        """lgbm classifier
        Args:
            train_data: tuple containing (data_X,data_y)
                data_X: list of untokenized text samples
                data_y: numpy array of one-hot encoded classes
            test_data: same as train_data, for test data.
            savename: save base path, used to save savename_svd.pickle, etc.
            run_transform: bool, if True, run and fit the tf-idf vectorizer
            train_split: split training into a train and validation set
            ylabels: label names for the y categories
            random_seed: seed to pass to loading function, used to 
                        randomize training data before splitting into train/val set
                        can be used for x-validation
            tfidf_params: dictionary of parameters to pass to tfidfvectorizer
            lgbm_params: dictionary of parameters to pass lgbm"""
            
        self.savename = savename    
        self.ylabels = ylabels
        
        self.random_seed = random_seed

        if random_seed is not None:
            np.random.seed(random_seed)

            
        p = np.random.permutation(len(train_data[0]))
        split_n = int(train_split*len(train_data[0]))
        self.train_X = [train_data[0][i] for i in p[:split_n]]
        self.train_y = np.array([train_data[1][i] for i in p[:split_n]])
        #split val data off the train data, so can x-validate by changing random_seed
        self.val_X = [train_data[0][i] for i in p[split_n:]]
        self.val_y = np.array([train_data[1][i] for i in p[split_n:]])
        self.test_X = test_data[0]
        self.test_y = test_data[1]

        tfidf_std_params = {'stop_words' : None,
                            'min_df' : 5,
                            'max_features' : None,
                            'use_idf' : True, 
                            'tokenizer' : None,
                            'ngram_range' : (1,4)}
        
        lgbm_std_params = {'boosting_type': 'gbdt',
                            'objective': 'binary',
                            'metric': 'xentropy',
                            'num_leaves': 20,
                            'learning_rate': 0.1,
                            'feature_fraction': 0.9,
                            'bagging_fraction': 0.8,
                            'bagging_freq': 1,
                            'verbose': 0,
                            'num_boost_round': 500,
                            'early_stopping_rounds': 5,}
        
        
        self.tfidf_params = dict(tfidf_std_params, **tfidf_params)
        self.lgbm_params = dict(lgbm_std_params, **lgbm_params)

        self.train_word_vectors(self.train_X)
        
        self.transform_word_vectors()
        
    def train_word_vectors(self,docs):
        """Train the tfidf vectorizer
        Args:
            docs: list of input strings
        Returns:
            None"""
        
            
        #may need to remove interpunction too?
        print('Building tfidf vectorizer')
        
        self.tfidf = TfidfVectorizer(**self.tfidf_params)
        
        self.tfidf.fit(docs)  
        
        if self.savename is not None:
            with open(self.savename + '_tfidf.obj','wb') as f:
                pickle.dump(self.tfidf,f)     
        print('Done training tfidf vectorizer')        
                
        
        
    def get_word_vectors(self, docs):
        """Get the tfidf vectors corresponding to text
        Args:
            docs: list of docs to be transformed using tfidf
        Returns:
            sparse array containing word vectors according to the trained transformer"""
        return self.tfidf.transform(docs)
        

    def transform_word_vectors(self):
        """Transform the train, val and test data and save if savename is given"""
        print('Transforming word vectors')
        
        self.train_X_tfidfvec = self.get_word_vectors(self.train_X)
        self.val_X_tfidfvec = self.get_word_vectors(self.val_X)
        self.test_X_tfidfvec = self.get_word_vectors(self.test_X)
        if self.savename is not None:
            with open(self.savename + '_X_tfidfvec.obj','wb') as f:
                pickle.dump((self.train_X_tfidfvec,self.val_X_tfidfvec,self.test_X_tfidfvec),f)  
        print('Done transforming word vectors')
                
                
    def train_models(self, savepath = None):
        """Build and compile lgbm models for every category
        Note: since the categories are not mutually exclusive, and lgbm does not
        support multiple binary classes, train an lgbm model for every class."""
        
        self.gbms = []
        self.accs = []
                
        for i in range(self.train_y.shape[1]):
            
            if self.ylabels is not None:
                print('Training GBM for {}'.format(self.ylabels[i]))

            
            lgb_train = lgb.Dataset(self.train_X_tfidfvec, self.train_y[:,i].flatten())
            lgb_eval = lgb.Dataset(self.val_X_tfidfvec, self.val_y[:,i].flatten(), reference=lgb_train)
            
            
            gbm = lgb.train(self.lgbm_params,
                lgb_train,
                num_boost_round=self.lgbm_params['num_boost_round'],
                valid_sets=lgb_eval,
                early_stopping_rounds=self.lgbm_params['early_stopping_rounds'])
                        
            self.gbms.append(gbm)
            
            y_pred = gbm.predict(self.val_X_tfidfvec, num_iteration=gbm.best_iteration)
            y_pred_cls = np.round(y_pred)
            self.accs.append(np.mean(y_pred_cls == self.val_y[:,i].flatten()))
            
        for i in range(len(self.accs)):
            print('Validation acc for {} is {:.2f}'.format(self.ylabels[i],self.accs[i]))
            
        with open(self.savename + '_models.obj','wb') as f:
            pickle.dump((self.ylabels,self.gbms),f)     





   
if __name__ == "__main__":
    
    trainpath = '/users/berend/Documents/Coding/ML-projects/ArxivData/train_data/train_data.json'
    testpath = '/users/berend/Documents/Coding/ML-projects/ArxivData/test_data/test_data.json'
    traindata,testdata = dp.loadfile(trainpath),dp.loadfile(testpath)
        
#    inc_categories =  ['cond-mat.mes-hall',
#                         'cond-mat.mtrl-sci',
#                         'cond-mat.stat-mech',
#                         'cond-mat.str-el',
#                         'cond-mat.supr-con',
#                         'cond-mat.soft',
#                         'quant-ph',
#                         'cond-mat.dis-nn',
#                         'cond-mat',
#                         'cond-mat.quant-gas',
#                         'cond-mat.other',
#                         'hep-th']  
    inc_categories =  ['cond-mat.mes-hall',
                         'cond-mat.mtrl-sci',
                         'cond-mat.stat-mech',]
 
    
    train_X,train_y = dp.generate_Xy_data_categories(traindata, inc_categories, ignore_others = True, 
                                shuffle_seed = 0, ydatatype = 'onehot',
                                clean_x = True, keep_latex_tags = True)
    test_X,test_y = dp.generate_Xy_data_categories(testdata, inc_categories, ignore_others = True, 
                                shuffle_seed = 0, ydatatype = 'onehot',
                                clean_x = True, keep_latex_tags = True)



    
    #load stopwords inferred from correlations:
    with open('save/inferred_stop_words.boj','rb') as f:
        inferred_stop_words = pickle.load(f)
            

    tfidf_params = {'stop_words' : None,
                        'min_df' : 2,
                        'max_features' : None,
                        'use_idf' : True, 
                        'tokenizer' : None}
    
    lgbm_params = {'boosting_type': 'gbdt',
                        'objective': 'binary',
                        'metric': 'xentropy',
                        'num_leaves': 20,
                        'learning_rate': 0.1,
                        'feature_fraction': 0.9,
                        'bagging_fraction': 0.8,
                        'bagging_freq': 1,
                        'verbose': 0,
                        'num_boost_round': 500,
                        'early_stopping_rounds': 5,}
    
    savename = 'save/lm_save'
    lm = lgbmTextClassifier((train_X,train_y), (test_X,test_y),ylabels = inc_categories, 
                           savename = savename, train_split = 0.7, 
                           random_seed = 0,run_transform = False,tfidf_params = tfidf_params,
                           lgbm_params = lgbm_params)
    
    lm.train_models()
    
    
    
