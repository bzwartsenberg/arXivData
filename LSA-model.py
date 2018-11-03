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
import keras.backend as K

import pickle



class LSATextClassifier():


    def __init__(self, data, savename = None, run_transform = True,
                 train_split = 0.66, random_seed = 0, tfidf_params = {},
                 svd_params = {},keras_params = {}):
        """LSA classifier
        Args:
            data: tuple containing (data_X,data_y)
                data_X: list of untokenized text samples
                data_y: numpy array of one-hot encoded classes
            savename: save base path, used to save savename_svd.pickle, etc.
            run_transform: bool, if True, run and fit the tf-idf vectorizer
            train_split: split training into a train and validation set
            random_seed: seed to pass to loading function, used to 
                        randomize training data before splitting into train/val set
                        can be used for x-validation
            tfidf_params: dictionary of parameters to pass to tfidfvectorizer
            svd_params: dictionary of parameters to pass to svd
            keras_params: dictionary of parameters to pass to logistic regression model
            """            
        ## put class initialization hyperparamters in a dict     
        self.savename = savename    
        
        self.random_seed = random_seed

        if random_seed is not None:
            np.random.seed(random_seed)

            
        p = np.random.permutation(len(data[0]))
        split_n = int(train_split*len(data[0]))
        self.train_X = [data[0][i] for i in p[:split_n]]
        self.train_y = np.array([data[1][i] for i in p[:split_n]])
        self.val_X = [data[0][i] for i in p[split_n:]]
        self.val_y = np.array([data[1][i] for i in p[split_n:]])

            
        self.stopwords = tfidf_params['stopwords'] if 'stopwords' in tfidf_params else None
        self.min_df = tfidf_params['min_df'] if 'min_df' in tfidf_params else 2     
        self.N_vec = svd_params['N_vec'] if 'N_vec' in svd_params else 100
        self.pos_weights = keras_params['pos_weights'] if 'pos_weights' in keras_params else np.ones((self.train_y.shape[1]))
        ## add option to load.
        ## add hyperparameter options to provide to TFidfVectorizer
        if run_transform:
            self.train_word_vectors(self.train_X)
            self.transform_word_vectors()
        else:
            print('Loading svd and tfidf')
            if savename is None:
                print('Please Provide savename if not training for the first time')
            else:
                with open(self.savename + '_svd.obj','rb') as f:
                    self.svd = pickle.load(f)
                with open(self.savename + '_tfidf.obj','rb') as f:
                    self.vectorizer = pickle.load(f)
                with open(self.savename + '_X_vec.obj','rb') as f:
                    self.train_X_vec,self.val_X_vec = pickle.load(f)        
     

    def train_word_vectors(self,docs,):
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
        
        if self.savename is not None:
            with open(self.savename + '_svd.obj','wb') as f:
                pickle.dump(self.svd,f)
            with open(self.savename + '_tfidf.obj','wb') as f:
                pickle.dump(self.vectorizer,f)        
        
        
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
        if self.savename is not None:
            with open(self.savename + '_X_vec.obj','wb') as f:
                pickle.dump((self.train_X_vec,self.val_X_vec),f)        


    def build(self):
        """Build and compile the logistic regression model in Keras"""
        
        lossfun = create_weighted_binary_crossentropy(self.pos_weights)
        
        self.model = Sequential()
#        #single:
#        self.model.add(Dense(self.train_y.shape[1], input_dim=self.N_vec, activation='sigmoid'))

        #double:
        self.model.add(Dense(100, input_dim=self.N_vec, activation='relu'))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(self.train_y.shape[1], activation='sigmoid'))
        
        self.model.summary()

        #multi onehot: binary cross entropy and binary accuracy
        self.model.compile(optimizer='adam', loss=lossfun, metrics=['binary_accuracy'])
        


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
        
        
        
        
        
def create_weighted_binary_crossentropy(pos_weights):
    """Generate a loss weighted binary crossentropy lossfunction.
    Arguments:
        pos_weights: weights for different one_hot positions
    Returns:
        a function with behavior:
        Arguments:
            output: A tensor containing network outputs.
            target: A tensor containing the required outputs (labels)
            from_logits: If False, output is expected to be probabilities, if
                    True, output is expectedto be logits.
        Returns:
            Weighted binary crossentropy for the given input tensors
    """
    
    
    # Note: tf.nn.sigmoid_cross_entropy_with_logits
    # expects logits, Keras expects to pass probabilities to the function.
    
    pos_weight = K.tf.convert_to_tensor(pos_weights, dtype = K.tf.float32)        
        
    def loss_fun(target, output, from_logits = False):
    
        if not from_logits:
            # transform back to logits
            _epsilon = K.tf.convert_to_tensor(K.epsilon(), dtype=output.dtype.base_dtype)
            output = K.tf.clip_by_value(output, _epsilon, 1-_epsilon)
            output = K.tf.log(output/(1-output))
    
        return K.mean(K.tf.nn.weighted_cross_entropy_with_logits(targets = target,
                                                       logits = output,pos_weight = pos_weight))
    return loss_fun


    
        
        
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
    train_X,train_y = dp.generate_Xy_data_categories(traindata, inc_categories, ignore_others = False, 
                                shuffle_seed = 0, ydatatype = 'onehot',
                                clean_x = True, keep_latex_tags = True)

    class_weights = 1/np.mean(train_y, axis = 0)
    
    tfidf_params = {'min_df' : 2,
                    'stopwords' : None}
    svd_params = {'N_vec' : 100}

    keras_params = {'pos_weights' : class_weights}    
    savename = 'ls_save'
    ls = LSATextClassifier((train_X,train_y),savename = savename, train_split = 0.7, 
                           random_seed = 0,run_transform = False,tfidf_params = tfidf_params,
                           svd_params = svd_params,keras_params = keras_params)
    
    ls.build()
    ls.train(batch_size=200,nb_epoch=75)
    

