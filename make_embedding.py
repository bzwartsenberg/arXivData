#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 09:48:37 2019

@author: berend
"""



import gensim
import time
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.data import find
import re
import string
import pickle

import data_preprocessing as dp




def prep_embedding_docs(docs, split_sentences = True, filter_punctuation = True, 
                        stopwords = stopwords.words('english')):
    """prepare documents for word embedding, filter stopwords, punctuation and split
    sentences
    Args:
        docs: list of tokenized unfiltered documents
        split_sentences: if true, split documents up into sentences
        filter_punctuation: if True, filter out punctuation
        stopwords: list of stopwords
    Returns:
        list of tokenized sentences"""
    
        
    #nltk filter stopwords
    filterset = set(stopwords)
    # filter punctuation...    
    if filter_punctuation:
        filterset = filterset.union(list(string.punctuation))
        filterset.add('``')
        filterset.add("''")        
        if split_sentences:
            filterset.remove('.') #we need to keep the periods
            
    ##split and filter:
    new_docs = dp.tokenize_filter_many(docs, filterset, reporting = 10000)    
    
    if split_sentences:
        sentences = []
        for doc in new_docs:
            sentence = []
            for token in doc:
                sentence.append(token)
                if token == '.':
                    if filter_punctuation:
                        sentence.pop(-1) ##remove the period
                    sentences.append(sentence)
                    sentence = []
            if sentence != []:
                sentences.append(sentence)
                sentence = []
        new_docs = sentences
    

    return new_docs





def make_embedding_matrix(docs, size, min_count = 5,  window = 5, n_iter = 5, savename = None, workers = 3):
    """Create an embedding matrix from a list of text samples.

    Args:
        docs: a list of text samples containing the vocabulary words.
        size: the size of the word-vectors.
        min_count: minimum number of occurences for word to be included
        window: ngram window
        n_iter: number of passes over corpus
        workers: number of threads
        savename: if not None, this name is used to save the embedding

    returns:
        embedding: gensim type embedding model
    """

    print('Starting the embedding generation')
    t0 = time.time()
    model = gensim.models.Word2Vec(docs, min_count=min_count, window = window,
                                   size = size, iter = n_iter, workers = workers)
    t1 = time.time()
    print('All done, total time %s' % (t1-t0))
    
    if savename is not None:
        model.save(savename)
        
    return model




def load_embedding_matrix(filepath, word2vec_format = False):
    """Load a pre-trained embedding matrix    
    Args:
        filepath: path to embedding
        word2vec_format: if True, use word2vecformat for loading
    
    returns:
        model: a dictionary mapping words to word-vectors (embeddings).
    """
    if word2vec_format:
        return gensim.models.KeyedVectors.load_word2vec_format(filepath, binary=True)
    else: #own pretrained model
        return gensim.models.Word2Vec.load(filepath)
    
    
    


if __name__ == "__main__":
    
    
    ### load data:
    trainpath = 'train_data/train_data.json'
    testpath = 'test_data/test_data.json'
    traindata = dp.loadfile(trainpath)
        
    inc_categories =    ['cond-mat.mes-hall',
                         'cond-mat.mtrl-sci',
                         'cond-mat.stat-mech',
                         'cond-mat.str-el',
                         'cond-mat.supr-con',
                         'cond-mat.soft',
                         'quant-ph',
                         'cond-mat.dis-nn',
                         'cond-mat.quant-gas',
                         'hep-th']
#    
    train_X,train_y = dp.generate_Xy_data_categories(traindata, inc_categories, ignore_others = True, 
                                shuffle_seed = 0, ydatatype = 'onehot',
                                clean_x = True, keep_latex_tags = True)

    
    with open('save/inferred_stop_words.obj','rb') as f:
        inferred_stop_words = pickle.load(f)
    
    do_embedding_gridsearch = True    
    if do_embedding_gridsearch:
        ## std for docs:
        docs = prep_embedding_docs(train_X, split_sentences = True, filter_punctuation = True, 
                            stopwords = stopwords.words('english'))
        
        size = 100
        min_count = 5
        window = 5
        n_iter = 5
        base_name = 'save/embedding'
        workers = 3        
        
        ### make embeddings
        savename = base_name + '_base'
        model = make_embedding_matrix(docs, size, min_count = min_count,  window = window, 
                              n_iter = n_iter, savename = savename, workers = workers)
    
        #size:
        for s in [50, 100, 200, 300]:
            savename = base_name + '_s_' + str(s)
            model = make_embedding_matrix(docs, s, min_count = min_count,  window = window, 
                                  n_iter = n_iter, savename = savename, workers = workers)
        
        
        #min count:
        for mc in [3, 5, 10]:
            savename = base_name + '_mc_' + str(mc)
            model = make_embedding_matrix(docs, size, min_count = mc,  window = window, 
                                  n_iter = n_iter, savename = savename, workers = workers)

        #window size:
        for wd in [3, 5, 7]:
            savename = base_name + '_wd_' + str(wd)
            model = make_embedding_matrix(docs, size, min_count = min_count,  window = wd, 
                                  n_iter = n_iter, savename = savename, workers = workers)
            
        #iterations:
        for it in [3, 5, 10]:
            savename = base_name + '_it_' + str(it)
            model = make_embedding_matrix(docs, size, min_count = min_count,  window = window, 
                                  n_iter = it, savename = savename, workers = workers)
    
    
        ##different preprocessing:
        size = 100
        min_count = 5
        window = 5
        n_iter = 5
        base_name = 'save/embedding'
        workers = 3        

        #no sentence split
        docs = prep_embedding_docs(train_X, split_sentences = False, filter_punctuation = True, 
                            stopwords = stopwords.words('english'))
        
        savename = base_name + '_nosent'
        model = make_embedding_matrix(docs, size, min_count = min_count,  window = window, 
                              n_iter = n_iter, savename = savename, workers = workers)

        #no punctuation filter
        docs = prep_embedding_docs(train_X, split_sentences = True, filter_punctuation = False, 
                            stopwords = stopwords.words('english'))
        
        savename = base_name + '_punct'
        model = make_embedding_matrix(docs, size, min_count = min_count,  window = window, 
                              n_iter = n_iter, savename = savename, workers = workers)

        
        #inferred stopwords
        wds = stopwords.words('english') + inferred_stop_words
        docs = prep_embedding_docs(train_X, split_sentences = True, filter_punctuation = True, 
                            stopwords = wds)
        
        savename = base_name + '_infwds'
        model = make_embedding_matrix(docs, size, min_count = min_count,  window = window, 
                              n_iter = n_iter, savename = savename, workers = workers)

    
        #no stopwords
        wds = stopwords.words('english') + inferred_stop_words
        docs = prep_embedding_docs(train_X, split_sentences = True, filter_punctuation = True, 
                            stopwords = [])
        
        savename = base_name + '_nowds'
        model = make_embedding_matrix(docs, size, min_count = min_count,  window = window, 
                              n_iter = n_iter, savename = savename, workers = workers)

        