# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 15:11:23 2022

@author: _K
"""

from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Input
from tensorflow.keras import Sequential
import matplotlib.pyplot as plt
import scipy.stats as ss 
import seaborn as sns
import numpy as np



class Cramers_V():
    def __init__(self):
        pass
    def cramers_corrected_stat(self,confussion_matrix):
        """ calculate Cramers V statistic for categorial-categorial association.
            uses correction from Bergsma and Wicher, 
            Journal of the Korean Statistical Society 42 (2013): 323-328
        """
        chi2 = ss.chi2_contingency(confussion_matrix)[0]
        n = confussion_matrix.sum()
        phi2 = chi2/n
        r,k = confussion_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

class Plot_graph():
    def __init__(self):
        pass
    def plot_categorical(self,df,categorical_column):
        for cat in categorical_column:
            plt.figure()
            sns.countplot(df[cat])
            plt.show()
            
    def plot_continuous(self,df,continuous_column):
        for con in continuous_column:
            plt.figure()
            sns.distplot(df[con])
            plt.show()
            
    def groupby(self,df):
        df.groupby(['job_type','term_deposit_subscribed']).agg(
            {'term_deposit_subscribed':'count'}).plot(kind='bar')

        df.groupby(['education','term_deposit_subscribed']).agg(
            {'term_deposit_subscribed':'count'}).plot(kind='bar')
        
        df.groupby(['marital','term_deposit_subscribed']).agg(
            {'term_deposit_subscribed':'count'}).plot(kind='bar')
            
        
        
class model_creation():
    def __init__(self):
        pass
    def model_development(self,nb_features,nb_classes,num_node=128,
                          dropout=0.3):
                              
        model = Sequential()
        model.add(Input(shape=(nb_features)))
        model.add(Dense(num_node,activation='relu',name='Hidden_layer1'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        model.add(Dense(num_node,activation='relu',name='Hidden_layer2'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        model.add(Dense(nb_classes,'softmax',name='Output_layer'))
        model.summary()
        
        return model
                              
        







