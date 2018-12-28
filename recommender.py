#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 21:58:51 2018

@author: berend
"""

###recommender
import numpy as np
import matplotlib.pyplot as plt

import data_preprocessing as dp

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import pickle



class recommender():
    
    def __init__(self,savepath, run_transform = True, datapath = dp.datapath,
                 interests_path = None):
        """pass"""
        
        print('Loading svd and tfidf')
        self.savepath = savepath
        with open(self.savepath + '_svd.obj','rb') as f:
            self.svd = pickle.load(f)
        with open(self.savepath + '_tfidf.obj','rb') as f:
            self.tfidf = pickle.load(f)    
            
        
        
        self.data = dp.loaddata(datapath)
        self.data_ids = [entry['id'] for entry in self.data]
        if run_transform:
            print('Transforming data vectors')
            self.data_vectors = self.get_vectors([entry['abstract'] for entry in  self.data])
            
            print('Normalizing')
            norm = np.linalg.norm(self.data_vectors, axis = 1)
            
            ##filter empty abstracts
            self.labels = [self.data[i]['id'] for i in np.argwhere(norm != 0).flatten()]
            self.data_vectors = self.data_vectors[np.argwhere(norm != 0).flatten()]
            
            ##normalize:
            self.data_vectors /= np.linalg.norm(self.data_vectors, axis = 1, keepdims = True)
            ## save ids
            with open(self.savepath + '_vecs.obj','wb') as f:
                pickle.dump((self.labels,self.data_vectors),f)     
            
        else:
            with open(self.savepath + '_vecs.obj','rb') as f:
                self.labels,self.data_vectors = pickle.load(f)

        if interests_path is not None:
            self.load_interests(interests_path)
    
    def load_interests(self, interests_path):
        """Load a file containing lists of ID's defined as "interests" 
        
        file should be an id on every line, with groups of interests separated 
        by a line of white space. The first line of a group is the name of the 
        interest
        
        Args: 
            interests_path: path to the file"""
            
        with open(interests_path, 'r') as f:
            interest_groups = f.read().split('\n\n')
            
        interest_groups = [group.split('\n') for group in interest_groups]
        
        self.interest_names = [group.pop(0) for group in interest_groups]
        
        self.interest_vecs = [self.get_interest_vector(group) for group in interest_groups]
        
        
    def get_interest_vector(self, group):
        """Given a list of ids, produce a vector that defines the interest
        Args:
            group: list of ids"""
        
        vecs = [self.data_vectors[self.labels.index(idstring)]  for idstring in group]

        return np.array(vecs).mean(axis = 0)
    
        
    def get_vectors(self, docs):
        """Get the svd-tfidf vector(s) corresponding to text
        Args:
            text: list of docs to be transformed using tfidf-svd
        Returns:
            array containing word vectors according to the trained model"""
        return self.svd.transform(self.tfidf.transform(docs))  
        
        
    def recommend_from_docs(self,docs, n):
        """Give recommendations based on docs
        Args:
            docs: document or list of documents to base recommendation on
            n: number of docs to recommend"""
        if not type(docs) == list:
            docs = [docs]

        sim_vec = np.mean(self.get_vectors(docs), axis = 0)
        sim_vec /= np.linalg.norm(sim_vec)
        
        cos_sim = np.dot(self.data_vectors, sim_vec)
        
        rec_articles = np.argsort(cos_sim)[::-1][:n]
        
        return rec_articles
        
    def recommend_from_interest(self,interest_vec, n):
        """Give recommendations based on docs
        Args:
            interest_vec: vector describing an interest
            n: number of docs to recommend"""

        
        cos_sim = np.dot(self.data_vectors, interest_vec)
        
        rec_articles = np.argsort(cos_sim)[::-1][:n]
        
        article_ids = [self.labels[i] for i in rec_articles]
        
        return article_ids    
    
    def article_from_id(self,article_id, return_idx = False):
        """Return an article based on the id
        Args:
            article_id: string with the id of the article
            return_idx: just return the index of the article"""
            
        article_idx = self.data_ids.index(article_id)
        if return_idx:
            return article_idx
        else:
            return self.data[article_idx]
        
        
        
        

        
if __name__ == "__main__":
    

        
    savename = 'save/ls_save_recommender'
    interests_path = 'save/interests_file'
        
    rc = recommender(savename, interests_path = interests_path)
        