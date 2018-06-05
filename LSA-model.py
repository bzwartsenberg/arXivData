#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 21:25:02 2018

@author: berend
"""

#### LSA supervised classifier

import numpy as np
import matplotlib.pyplot as plt

import data_preprocessing as dp

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
#from nltk.corpus import stopwords

from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2




class LSATextClassifier():


    def __init__(self, data, train_split = 0.7, stopwords = None, random_seed = 0, N_vec = 100, min_df = 2):
        """LSA classifier
        Args:
            data: tuple containing (data_X,data_y)
                data_X: list of untokenized text samples
                data_y: numpy array of one-hot encoded classes
            stopwords: a list of stopwords to be passed to TfidfVectorizer, if 'english' is passed, the default is used
            random_seed: seed to pass to loading function
            N_vec: length of the word vectors
            min_df: minimal number doc frequency of words to be taken into account"""   
            
        self.random_seed = random_seed

        if random_seed is not None:
            np.random.seed(random_seed)

            
        p = np.random.permutation(len(data[0]))
        split_n = int(train_split*len(data[0]))
        self.train_X = [data[0][i] for i in p[:split_n]]
        self.train_y = np.array([data[1][i] for i in p[:split_n]])
        self.val_X = [data[0][i] for i in p[split_n:]]
        self.val_y = np.array([data[1][i] for i in p[split_n:]])
   
            
        self.stopwords = stopwords
        self.N_vec = N_vec
        self.min_df = min_df        
        self.train_word_vectors(self.train_X)
    
        self.transform_word_vectors()
        
     

    def train_word_vectors(self,docs):
        """Train the tfidf-svd vectorizer
        Args:
            docs: list of tokenized input strings
        Returns:
            None"""
        
            
        #may need to remove interpunction too?
        print('Building tfidf vectorizer')
        
        self.vectorizer = TfidfVectorizer(max_df=0.8, max_features=None,
                                 min_df=self.min_df, stop_words=self.stopwords,
                                 use_idf=True, tokenizer = dp.tokenize)
        
        self.vectorizer.fit(docs)

        print('Building svd transformer')
        
        self.svd = TruncatedSVD(self.N_vec)
        self.svd.fit(self.vectorizer.transform(docs))
        ##save?
        
        
    def get_word_vectors(self, docs):
        """Get the svd-tfidf vector(s) corresponding to text
        Args:
            text: list of docs to be transformed using tfidf-svd
        Returns:
            array containing word vectors according to the trained model"""
        return self.svd.transform(self.vectorizer.transform(docs))
        
        
    def transform_word_vectors(self):
        """Transform the train and test data"""
        print('Transforming word vectors')
        
        self.train_X_vec = self.get_word_vectors(self.train_X)
        self.val_X_vec = self.get_word_vectors(self.val_X)


    def build(self):
        """Build and compile the logistic regression model in Keras"""
        self.model = Sequential()
        self.model.add(Dense(self.train_y.shape[1], input_dim=self.N_vec, activation='softmax'))

        self.model.summary()

        self.model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
        


    def train(self, batch_size=50, nb_epoch=100):
        """Train the model on the training data.
        Args:
            batch_size: batch size used while training
            nb_epoch: number of epochs
        returns:
            keras training history"""
                
        validation_data = (self.val_X_vec,self.val_y)            


        history = self.model.fit(self.train_X_vec, self.train_y,
                        batch_size=batch_size, epochs=nb_epoch,
                        verbose=1, validation_data=validation_data)
        return history
        

    def evaluate(self, testdata):
        """Evaluate the model on the test data.
        
        Args:
            testdata: tuple containing:
                test_X: test data as list of abstracts
                test_y: test labels as array of one-hot encoded vectors
        
        returns:
            the model's accuracy classifying the test data.
            """

        score = self.model.evaluate(self.test_X_vector, self.test_y, verbose=0)
    
        print('loss:', score[0])
        print('accuracy:', score[1])
        return score[1]

    def predict(self, reviews):
        """Predict the sentiment of unlabelled reviews.
        
        returns: the predicted label of :review:
        """

        return self.model.predict_classes(self.get_word_vectors(reviews))
        
        
        
if __name__ == "__main__":
    
    trainpath = 'train_data/train_data.json'
    testpath = 'test_data/test_data.json'
    traindata,testdata = dp.loadfile(trainpath),dp.loadfile(testpath)
        
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
    train_X,train_y = dp.generate_Xy_data_categories(traindata, inc_categories,
                                                  convert_to_catnum = True, 
                                ignore_others = False, split_multilabel = True, 
                                shuffle_seed = 0, to_one_hot = True, 
                                clean_x = True, keep_latex_tags = True)
    
    ls = LSATextClassifier((train_X,train_y), train_split = 0.7, stopwords = None, random_seed = 0, N_vec = 100, min_df = 2)
    
    ls.transform_word_vectors()
    ls.build()
    ls.train()

