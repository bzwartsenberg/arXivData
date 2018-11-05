#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 12:44:02 2018

@author: berend
"""

#post processing:
    
    
import numpy as np
import matplotlib.pyplot as plt

import data_preprocessing as dp

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
#from keras.regularizers import l2
import keras.backend as K

import pickle


from lsamodel import LSATextClassifier



if __name__ == "__main__":
    
#    
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
    train_X,train_y = dp.generate_Xy_data_categories(traindata, inc_categories, ignore_others = False, 
                                shuffle_seed = 0, ydatatype = 'onehot',
                                clean_x = True, keep_latex_tags = True)
    test_X,test_y = dp.generate_Xy_data_categories(testdata, inc_categories, ignore_others = False, 
                                shuffle_seed = 0, ydatatype = 'onehot',
                                clean_x = True, keep_latex_tags = True)

    
    class_weights = 1/np.mean(train_y, axis = 0)
    
    #load stopwords inferred from correlations:
    with open('save/inferred_stop_words.boj','rb') as f:
        inferred_stop_words = pickle.load(f)
            
    
    #note: from the corrlations, it seems that "that" and "and" actually hold some 
    # value, the model might train better without the stopwords.words('english') added.
    tfidf_params = {'min_df' : 2,
                    'stopwords' : inferred_stop_words + stopwords.words('english')}
    svd_params = {'N_vec' : 100}

    keras_params = {'pos_weights' : class_weights}    
    savename = 'save/ls_save'
    
    

    ls = LSATextClassifier((train_X,train_y), (test_X,test_y),savename = savename, train_split = 0.7, 
                       random_seed = 0,run_transform = False,tfidf_params = tfidf_params,
                       svd_params = svd_params,keras_params = keras_params)
    savepath = 'save/keras_model.h5'    
    ls.build(loadpath = savepath)

    
    ## find class vectors and transform back to word vectors:
    weights, biasses = ls.model.get_weights()
    
    word_vecs = ls.svd.inverse_transform(weights.T)
    
    words = ls.vectorizer.get_feature_names()
    for i,cat in enumerate(inc_categories):
        print('\n\n\nFor category ', cat)
        
        highest = np.argsort(word_vecs[i])[-1:-11:-1]
        lowest = np.argsort(word_vecs[i])[0:10]
        print('\nhighest correlations:')
        for num in highest:
            print('Word \'{}\' correlates {:.2f}'.format(words[num],word_vecs[i,num]))
        print('\nhighest anti-correlations:')
        for num in lowest:
            print('Word \'{}\' correlates {:.2f}'.format(words[num],word_vecs[i,num]))

    
    #make plots:
    def plot_correlations(word_vec, words, n_bars, cat_name, savename = None):
        highest = np.argsort(word_vec)[-1:-1-n_bars:-1]
        lowest = np.argsort(word_vec)[0:n_bars]
        
        bars = np.concatenate((lowest,highest[::-1]))
        tick_label = [words[i] for i in lowest] + ['','',''] + [words[i] for i in highest[::-1]]
        corrs = [word_vec[i] for i in lowest]
        corrs += [0.,0.,0.]+[word_vec[i] for i in highest[::-1]]
        corrs = np.array(corrs)
        
        vmax = np.abs(corrs).max()
        
        colors = [plt.cm.coolwarm((corr/(2*vmax))+0.5) for corr in corrs]
        
        x = np.arange(len(corrs))
        plt.tight_layout()

        fig,ax = plt.subplots(figsize = (4,3.5))
        ax.barh(x,corrs, tick_label = tick_label, color = colors)
        ax.set_yticklabels(tick_label, rotation = 0)
        fig.subplots_adjust(left=0.35)
        fig.subplots_adjust(bottom=0.15)
        fig.subplots_adjust(top=0.95)
        ax.set_xlabel('Correlation')

        if savename is not None:
            fig.savefig(savename, dpi = 600)
        
    plot_correlations(word_vecs[4],words,10,inc_categories[4], savename = 'supercon_corr.png')




    #categories most classified wrong:
    #binary accuracy: K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)
    val_pred_y = ls.model.predict(ls.get_word_vectors(ls.val_X)) 
    print('validation binary accuracy: {:.2f}'.format(np.mean(np.equal(np.round(val_pred_y),ls.val_y))))
    
    ##binary accuracies per category:    
    true_pos = np.logical_and((np.round(val_pred_y) == 1.), (ls.val_y == 1.))
    true_neg = np.logical_and((np.round(val_pred_y) == 0.), (ls.val_y == 0.))
    false_neg = np.logical_and((np.round(val_pred_y) == 0.), (ls.val_y == 1.))
    false_pos = np.logical_and((np.round(val_pred_y) == 1.), (ls.val_y == 0.))
    
    accs = np.mean(true_pos + true_neg, axis = 0)
    false_negs = np.mean(false_neg, axis = 0)    
    false_poss = np.mean(false_pos, axis = 0)    
    for i,cat in enumerate(inc_categories):
        print('\n\nBinary accuracy for cat \'{}\' is {:.2f}'.format(cat,accs[i]))
        print('False positive rate for cat \'{}\' is {:.2f}'.format(cat,false_poss[i]))
        print('False negative rate for cat \'{}\' is {:.2f}'.format(cat,false_negs[i]))
        print('Class occurence for this class: {}'.format(1/class_weights[i]))
    
    #This clearly shows that most of the inaccuracies come from false posirtives, for particular classes
    # the highest false negative rate is around 3%, while the highest false positive rate is around 30%
    
    print('Total false negative rate:{:.2f}'.format(np.mean(false_negs)))
    print('Total false positive rate:{:.2f}'.format(np.mean(false_poss)))
    
    #conclusion: most wrong classifications come from false positives,
    #mostly in categories 'others', 'cond-mat' and and 'cond-mat.other',
    #the categories that have somewhat ill defined properties.
    #also it does not seem to be related to class occurence, 
    #so changing the weight function will not work perse
    
    
    