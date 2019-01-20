#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 13:37:25 2019

@author: berend
"""

# import tensorflow as tf 	# (optional) feel free to build your models using keras

import data_preprocessing as dp


from nltk.corpus import stopwords
from nltk import word_tokenize

import gensim

import pickle

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM,Dropout, Activation, BatchNormalization,Flatten,GlobalMaxPooling1D, Conv1D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.regularizers import l2,l1,l1_l2
from keras import backend as K
from keras.callbacks import Callback

import numpy as np

import matplotlib.pyplot as plt

from lsamodel import sensitivity, precision, create_weighted_binary_crossentropy

class CNNTextClassifier():
    """
    Class for classifying text with CNN and word embeddings
    Following somewhat along:
    http://www.aclweb.org/anthology/D14-1181
    """

    def __init__(self, train_data, test_data, embedding, savename = None, 
                 ylabels = None, train_split = 0.66, random_seed = 0, load_vecs = False,
                 embed_params = {}, cnn_params = {}):
        """Initialize the classifier
        Args:
            path: path to training and test data, or tuple containing train_X,train_y,test_X,test_y
            embedding: gensim type embedding
            unk_token: if set to True, use a new token for "unk" with random word vector,
                       otherwise remove word from the sequence
            trunc_len: number of tokens to consider
            seed: random seed to provide
            """
        self.embedding = embedding
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
           
            
 


        embed_std_params = {'unk_token' : True,
                            'trunc_len' : 200,
                            'tokenizer': word_tokenize}
        
        
        cnn_std_params = {'cnn_filters' : [(3,100), (4,100),(5,100)],
                          'pos_weights' : np.ones(self.test_y.shape[1]),
                          'optimizer' : Adam(lr=0.001),
                          'wvec_trainable' : True, 
                          'norm_wvecs' : True}      


        self.embed_params = dict(embed_std_params, **embed_params)
        self.cnn_params = dict(cnn_std_params, **cnn_params)

        self.embed_params['word_vec_len'] = self.embedding.vector_size
        self.embed_params['vocab_size'] = len(self.embedding.wv.vocab)
        
        

        if load_vecs:
            with open(self.savename + '_ints.obj','rb') as f:
                self.train_X_ints,self.val_X_ints,self.test_X_ints = pickle.load(f)
                print('Loaded integer sequences')

        else:
            
            print('Transforming to int sequences')
            self.train_X_ints = self.to_int_sequences(self.train_X)
            self.val_X_ints = self.to_int_sequences(self.val_X)
            self.test_X_ints = self.to_int_sequences(self.test_X)
                      
            if savename is not None:
                with open(self.savename + '_ints.obj','wb') as f:
                    pickle.dump((self.train_X_ints,self.val_X_ints,self.test_X_ints), f)
             
        
        
    def to_int_sequences(self, data):
        """Use the model to convert the sentences to token sequences
        Args:
            data: list of documents to be converted
        Returns:
            numpy array of size len(data)*self.trunc_len, with integers as tokens"""
        
        data_vec = np.zeros((len(data), self.embed_params['trunc_len']), dtype = 'int')
        #use 0 for padding, so shift embedding values up by 1
        #use self.vocab_size+1 for 'unk' word
        
        tokenizer = self.embed_params['tokenizer']
        
        for i in range(data_vec.shape[0]):
            tokens = tokenizer(data[i])
            j = 0  ###j is the read-in
            k = 0  ###k is the write out 
            while (k < self.embed_params['trunc_len']) and (j < len(tokens)):
                try:
                    data_vec[i,k] = self.embedding.wv.vocab[tokens[j]].index + 1
                except KeyError:
                    if self.embed_params['unk_token']:
                        data_vec[i,k] = self.embed_params['vocab_size'] + 1
                    else: ## else skip the word:
                        k -= 1
                    
                k += 1
                j += 1
                            
        return data_vec

    def build(self, loadpath = None):
        """Build the model
        Args:
            filters: shapes of convolutional filters as list of tuples, 
                    where each tuple is the filtersize and number of filters
            wvec_trainable: treat the weight vectors ars trainable parameters or not
            norm_wvecs: use normalized weight vectors, or use unnormalized ones"""


        if loadpath is not None:
            print('Loading model from path: {}'.format(loadpath))
            lossfun = create_weighted_binary_crossentropy(self.pos_weights)            
#            self.model = load_model(loadpath, custom_objects={'lossfun': lossfun})            
            self.model.summary()
        else:
            print('Generating new model.')
            lossfun = create_weighted_binary_crossentropy(self.cnn_params['pos_weights'])
            

            embed_size = self.embed_params['vocab_size'] + 1
            if self.embed_params['unk_token']:
                embed_size += 1
    
            embedding_mat = np.zeros((embed_size, self.embed_params['word_vec_len']))
            if self.cnn_params['norm_wvecs']:
                embedding_mat[1:self.embed_params['vocab_size'] +1] = self.embedding.wv.syn0/np.linalg.norm(self.embedding.wv.syn0, axis = 1, keepdims = True)
                unk_word = np.random.uniform(-3,3,size = (self.embed_params['word_vec_len'])) 
                unk_word /= np.linalg.norm(unk_word)
            else:
                embedding_mat[1:self.embed_params['vocab_size'] +1] = self.embedding.wv.syn0
                unk_word = np.random.uniform(-3,3,size = (self.embed_params['word_vec_len'])) #the -3,3 is emprical: checked the word vectors and that is about the range
            
            if self.embed_params['unk_token']:
                embedding_mat[-1] = unk_word
    
            self.model = Sequential()
            #keras automatically pads the inputs, I could shorten the inputs?
            self.model.add(Embedding(embed_size,
                            self.embed_params['word_vec_len'],
    #                        weights=[embedding_mat.T],
    #                        embeddings_regularizer=l2(0.002),
                            input_length=self.embed_params['trunc_len'],
                            trainable=self.cnn_params['wvec_trainable']))
            #WITHOUT batch normalization
            #last dimension is word dimension so that should be full length filter:
            #for filter in filters:      
            #To add:
                #multiple filters
                #l2 regularization
                
            #change to multi-filter
            filt = self.cnn_params['cnn_filters'][0]
            self.model.add(Conv1D(filt[1], filt[0],kernel_regularizer=l2(0.000)))
    
            self.model.add(Activation('relu')) #1
            self.model.add(GlobalMaxPooling1D()) #2
            self.model.add(Dropout(0.5)) #2
            self.model.add(Dense(512,kernel_regularizer=l2(0.000))) #this layer is not used in 
            self.model.add(Activation('relu'))
            self.model.add(Dropout(0.5)) #2
            self.model.add(Dense(self.train_y.shape[1]))
            self.model.add(Activation('sigmoid'))
            self.model.summary()
    
            #multi onehot: binary cross entropy and binary accuracy
            self.model.compile(optimizer=self.cnn_params['optimizer'], loss=lossfun, metrics=[sensitivity, precision])
        
        



    def train(self, batch_size=50, num_epochs=5, validation = 1000):
        """Train the model on the training data."""

        
        history = self.model.fit(self.train_X_ints, self.train_y, batch_size=batch_size, epochs=num_epochs, verbose=1, validation_data=(self.val_X_ints,self.val_y))
        return history


    def evaluate(self):
        """Evaluate the model on the test data.
        
        returns:
            the model's accuracy classifying the test data.
            """
            
            
        y_true = self.val_y
        y_pred = self.model.predict(cn.val_X_ints, return_probabilities = False)
        
        true_pos = np.mean(np.logical_and((y_pred == 1.), (y_true == 1.)), axis = 0)
        true_neg = np.mean(np.logical_and((y_pred == 0.), (y_true == 0.)), axis = 0)
        false_neg = np.mean(np.logical_and((y_pred == 0.), (y_true == 1.)), axis = 0)
        false_pos = np.mean(np.logical_and((y_pred == 1.), (y_true == 0.)), axis = 0)
        
        accs = true_pos + true_neg
        sens = true_pos / (true_pos + false_neg)
        prec = true_pos / (true_pos + false_pos)
        
        for i,cat in enumerate(self.ylabels):
            print('\n\nFor category {}'.format(cat))
            print('True positive rate is  {:.3f}'.format(true_pos[i]))
            print('True negative rate is  {:.3f}'.format(true_neg[i]))
            print('False positive rate is {:.3f}'.format(false_pos[i]))
            print('False negative rate is {:.3f}'.format(false_neg[i]))
    
            print('Accuracy is    {:.3f}'.format(accs[i]))
            print('Sensitivity is {:.3f}'.format(sens[i]))
            print('Precision is   {:.3f}'.format(prec[i]))
    
    
        
        
        
        print('\n\nCategory              sens   prec   acc')
        for i,cat in enumerate(self.ylabels):
            print('{:<20}  {:.3f}  {:.3f}  {:.3f}'.format(cat, sens[i], prec[i], accs[i]))
        
        print('\n\nAverage accuracy: {:.3f}'.format(np.mean(accs)))
        print('Average precission: {:.3f}'.format(np.mean(prec)))
        print('Average sensitivity: {:.3f}'.format(np.mean(sens)))
        
            

        score = self.model.evaluate(self.test_X_vec, self.test_y, verbose=0)
    
        print('loss:', score[0])
        print('accuracy:', score[1])
        return score[1]            



    def predict(self, X_ints, return_probabilities = False):
        """Predict the sentiment of unlabelled docs.
        Args:
            docs: list of docs to be classified
            return_probabilities: if true, return the probabilities, 
                                otherwise return one-hot vectors (rounded)
        
        returns: the predicted labels or probabilities of docs
        """
        probabilities = self.model.predict(X_ints)
        
        if return_probabilities:
            return probabilities
        else:
            return np.round(probabilities)



if __name__ == "__main__":
    
    
    ### load data:
#    trainpath = 'train_data/train_data.json'
#    testpath = 'test_data/test_data.json'
#    traindata,testdata = dp.loadfile(trainpath),dp.loadfile(testpath)
#        
#    inc_categories =    ['cond-mat.mes-hall',
#                         'cond-mat.mtrl-sci',
#                         'cond-mat.stat-mech',
#                         'cond-mat.str-el',
#                         'cond-mat.supr-con',
#                         'cond-mat.soft',
#                         'quant-ph',
#                         'cond-mat.dis-nn',
#                         'cond-mat.quant-gas',
#                         'hep-th']
#    
#    train_X,train_y = dp.generate_Xy_data_categories(traindata, inc_categories, ignore_others = True, 
#                                shuffle_seed = 0, ydatatype = 'onehot',
#                                clean_x = True, keep_latex_tags = True)
#    test_X,test_y = dp.generate_Xy_data_categories(testdata, inc_categories, ignore_others = True, 
#                                shuffle_seed = 0, ydatatype = 'onehot',
#                                clean_x = True, keep_latex_tags = True)
#    
#    
#    print('Loaded data')
##    class_weights = 0.1/np.mean(train_y, axis = 0)
#    class_weights = np.ones((train_y.shape[1]))
#    
#    
#    ###truncated google news embedding:
#    from nltk.data import find
#    word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
#    embedding = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)    
#    
#    print('Loaded embedding')
#


    embed_params = {'unk_token' : True,
                        'trunc_len' : 200,}
    
    
    cnn_params = {'cnn_filters' : [(3,100), (4,100),(5,100)],
                          'pos_weights' : class_weights,
                          'optimizer' : Adam(lr=0.001),
                          'wvec_trainable' : False, 
                          'norm_wvecs' : True}      



    seed = 0
    trunc_len = 200
    unk_token = True
    

    savename = 'save/cn_save'

    
    cn = CNNTextClassifier((train_X,train_y), (test_X,test_y), embedding, savename = savename, 
                 ylabels = inc_categories, train_split = 0.7, random_seed = 0, load_vecs = True,
                 embed_params = embed_params, cnn_params = cnn_params)

    cn.build()
    
    cn.train(batch_size=50, num_epochs=5, validation = 1000)
    savepath = 'save/cnn_model.h5'
    cn.model.save(savepath)
    
    
    y_true = cn.val_y
    y_pred = np.round(cn.model.predict(cn.val_X_ints))
    
    true_pos = np.mean(np.logical_and((y_pred == 1.), (y_true == 1.)), axis = 0)
    true_neg = np.mean(np.logical_and((y_pred == 0.), (y_true == 0.)), axis = 0)
    false_neg = np.mean(np.logical_and((y_pred == 0.), (y_true == 1.)), axis = 0)
    false_pos = np.mean(np.logical_and((y_pred == 1.), (y_true == 0.)), axis = 0)
    
    accs = true_pos + true_neg
    sens = true_pos / (true_pos + false_neg)
    prec = true_pos / (true_pos + false_pos)
    
    for i,cat in enumerate(inc_categories):
        print('\n\nFor category {}'.format(cat))
        print('True positive rate is  {:.3f}'.format(true_pos[i]))
        print('True negative rate is  {:.3f}'.format(true_neg[i]))
        print('False positive rate is {:.3f}'.format(false_pos[i]))
        print('False negative rate is {:.3f}'.format(false_neg[i]))

        print('Accuracy is    {:.3f}'.format(accs[i]))
        print('Sensitivity is {:.3f}'.format(sens[i]))
        print('Precision is   {:.3f}'.format(prec[i]))


    
    
    
    print('\n\nCategory              sens   prec   acc')
    for i,cat in enumerate(inc_categories):
        print('{:<20}  {:.3f}  {:.3f}  {:.3f}'.format(cat, sens[i], prec[i], accs[i]))
    
    print('\n\nAverage accuracy: {:.3f}'.format(np.mean(accs)))
    print('Average precission: {:.3f}'.format(np.mean(prec)))
    print('Average sensitivity: {:.3f}'.format(np.mean(sens)))
    
