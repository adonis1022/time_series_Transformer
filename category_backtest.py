#!/usr/bin/env python
# coding: utf-8

# In[15]:


from keras.layers import Input,Dense,Embedding,Lambda,TimeDistributed,LSTM, Reshape, Dropout
from keras.models import Model
from TransformerEncoder import TransformerEncoder
from TrainablePositionalEmbeddings import TransformerPositionalEmbedding
import os
import pickle
from sklearn.model_selection import train_test_split
from keras.optimizers import Nadam, Adam
from Attention import MultiHeadedAttention
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib.pylab as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from hyperopt import hp, tpe, STATUS_OK, Trials
from hyperopt.fmin import fmin
import math
from keras.utils import to_categorical
from keras.models import Model
import keras.backend as K


# In[2]:


sku_id = 48341
n_in = 12
n_out = 1
n_test = 12
file_name = 'data/sku_day_20191023_v2.csv'


# In[3]:


#data load
def load_data(file_name):

    df = pd.read_csv(file_name)
    data_beer = df[df["sku_id"] == sku_id].copy()
    data_beer['time'] = pd.to_datetime(data_beer['time'], format='%Y-%m')
    data_beer['week'] = ((data_beer['time'] - pd.datetime(year=2014, month=1, day=6)).dt.days / 7).astype(int)
    data_beer_week = data_beer.groupby(['week'])['y'].sum().to_frame().sort_values(['week']).reset_index()
    data_beer_week = pd.DataFrame(data_beer_week.y.values).values
    
    return data_beer_week

# drop outlier val>mean+2std
def drop_outlier(data):
    
    data_beer_week = data
    data_beer_week = data_beer_week.reshape(len(data_beer_week),)
    point = np.mean(data_beer_week)+2*np.std(data_beer_week)
    for i in range(len(data_beer_week)):
        if data_beer_week[i] > point:
            data_beer_week[i] = point
    data_beer_week = data_beer_week.reshape(len(data_beer_week),1)
    
    return data_beer_week

#data split
def data_split(data):
    
    data_beer_week = data
    df = pd.DataFrame(data_beer_week)
    cols = list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    agg = pd.concat(cols, axis=1)
    agg.dropna(inplace=True)
    data_supervised = agg.values
    
    mu = pd.Series(data_beer_week.reshape(-1,)).mean()
    sigma = pd.Series(data_beer_week.reshape(-1,)).std()
    max_volumn = mu + 2*sigma
    
    data_supervised[(data_supervised>=0)&(data_supervised<max_volumn/5)]=0
    data_supervised[(data_supervised>=max_volumn/5) & (data_supervised<2*max_volumn/5)]=1
    data_supervised[(data_supervised>=2*max_volumn/5) & (data_supervised<3*max_volumn/5)]=2
    data_supervised[(data_supervised>=3*max_volumn/5) & (data_supervised<4*max_volumn/5)]=3
    data_supervised[data_supervised>=4*max_volumn/5]=4
    
    data_train = data_supervised[:-n_test,:]
    data_test = data_supervised[-n_test:,:]
    x_train = data_train[:,:-1]
    y_train = to_categorical(data_train[:,-1],num_classes=5) # one-hot
    x_test = data_test[:,:-1]
    y_test = to_categorical(data_test[:,-1],num_classes=5) # one-hot
    
    return x_train, x_test, y_train, y_test


# In[4]:


data_beer_week = load_data(file_name)
data_beer_week = drop_outlier(data_beer_week)
x_train, x_test, y_train, y_test = data_split(data_beer_week)


# In[5]:


def data_split_new(num_test, x_train, y_train):
    x_train_set = x_train[0:len(x_train)-num_test,]
    y_train_set = y_train[0:len(y_train)-num_test,]
    x_test_set = x_train[len(x_train)-num_test:,]
    y_test_set = y_train[len(y_train)-num_test:,]
    
    return x_train_set, y_train_set, x_test_set, y_test_set


# In[6]:


def build_model(hidden_dim, positional_ff_dim, heads, dropout):
    '''
    embedding: uniform initialization
    encoder + softmax
    '''
    encoder_inputs = Input((12,),name='Encoder_input')
    next_step_input = Embedding(5,hidden_dim,input_length=12)(encoder_inputs)
    next_step_input = Dropout(0.2)(next_step_input)


    positional_embedding_layer    = TransformerPositionalEmbedding(name='Positional_embedding')
    next_step_input               = positional_embedding_layer(next_step_input)

    next_step_input,attention =TransformerEncoder(hidden_dim,
                                                             heads,
                                                             hidden_dim,
                                                             hidden_dim,
                                                             positional_ff_dim,
                                                             dropout_rate= dropout,
                                                             name= 'Transformer')(next_step_input)
    
    next_step_input = Lambda(lambda x: x[:,0])(next_step_input)
    next_step_input = Dropout(0.5)(next_step_input)
    outputs = Dense(5, activation='softmax')(next_step_input)
    model = Model(encoder_inputs, outputs)
#     model.summary()
    
    return model


# In[11]:


def model_fit(x_train, y_train):    
    
    def objective(space): # Backtest return loss

        #count = 0
        metric = list()
        for index in range(len(x_train)//2, len(x_train), 20):  # 
            
            data_x = x_train[0:index,]
            data_y = y_train[0:index,]
            
            x_train_set, y_train_set, x_test_set, y_test_set = data_split_new(12, data_x, data_y)
            
            print(space)
            model = build_model(
                                hidden_dim        = int(space['hidden_dim']),
                                positional_ff_dim = int(space['positional_ff_dim']),
                                heads             = int(space['heads']),
                                dropout           = space['dropout_rate']
            )
            
            model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
            model.fit(x_train_set,y_train_set, epochs=200, verbose=0)
            loss = model.evaluate(x_test_set, y_test_set)
            print(loss)
            metric.append(loss[0])
            K.clear_session()
            
        ave_logloss = np.mean(metric)
        print(ave_logloss)
        
        return{'loss': ave_logloss, 'status': STATUS_OK }

    space = {
            'hidden_dim':       hp.quniform('hidden_dim', 16, 64, 16),
            'positional_ff_dim':hp.quniform('positional_ff_dim', 16, 64, 16),
            'heads':            hp.quniform('heads', 1, 5, 1),
            'dropout_rate':     hp.uniform('dropout_rate', 0.1, 0.5)
            }

    
    trials = Trials()
    best = fmin(
                fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=100,
                trials=trials
               )

    print(best)
    return best


# In[12]:


def get_predict(space, x_train, x_test, y_train, y_test):
    model = build_model(
                                hidden_dim        = int(space['hidden_dim']),
                                positional_ff_dim = int(space['positional_ff_dim']),
                                heads             = int(space['heads']),
                                dropout           = space['dropout_rate']
            )
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train)
    loss = model.evaluate(x_test, y_test)
    print('logloss',loss[0],'acc:',loss[1])
    pre = model.predict(x_test)
    return pre,y_test


# In[13]:


best = model_fit(x_train, y_train)
# pre,y_test = get_predict(best, x_train, x_test, y_train, y_test)
# print(pre, y_test)
print('Best param:', best)


# In[14]:


pre,y_test = get_predict(best, x_train, x_test, y_train, y_test)


# In[32]:


print(pre, y_test)


# In[ ]:
'''
Best param: {'dropout_rate': 0.4759756666055627, 'heads': 1.0, 'hidden_dim': 64.0, 'positional_ff_dim': 48.0}

logloss 1.9766827821731567 acc: 0.5
[[0.85969365 0.04647004 0.01900198 0.04733295 0.02750145]
 [0.726568   0.0171862  0.03808558 0.12071923 0.097441  ]
 [0.9671447  0.00892267 0.01067425 0.00515143 0.00810694]
 [0.8781355  0.03839016 0.01700077 0.04133606 0.02513748]
 [0.7464286  0.01480593 0.0341262  0.11400263 0.09063665]
 [0.96945167 0.00795645 0.01013419 0.00474288 0.00771481]
 [0.9381849  0.00640529 0.00440318 0.01655219 0.03445429]
 [0.81319886 0.06275356 0.0287076  0.03054262 0.06479733]
 [0.96713644 0.00892415 0.01067778 0.00515189 0.0081097 ]
 [0.7768596  0.01355588 0.03087895 0.09062622 0.08807934]
 [0.7977812  0.01221501 0.02677983 0.07706379 0.08616012]
 [0.8532379  0.0474603  0.02019763 0.02071419 0.05838988]] 
 [[1. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0.]
 [0. 0. 0. 1. 0.]
 [1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1.]
 [1. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0.]]

Best param: {'dropout_rate': 0.47405815292879644, 'heads': 1.0, 'hidden_dim': 16.0, 'positional_ff_dim': 32.0}
logloss 1.7849326133728027 acc: 0.1666666716337204
[[0.6695136  0.08406448 0.02975885 0.05870314 0.15796001]
 [0.15548235 0.09061003 0.33624062 0.17925115 0.23841587]
 [0.28367376 0.136507   0.17554022 0.12494173 0.2793373 ]
 [0.7094348  0.06826835 0.02599211 0.04962675 0.146678  ]
 [0.15195079 0.07973026 0.3564655  0.16215694 0.2496966 ]
 [0.30332887 0.12157679 0.16932653 0.11875749 0.2870103 ]
 [0.52168417 0.17153355 0.07097082 0.12702608 0.10878541]
 [0.2393081  0.20353015 0.34942397 0.08846921 0.11926859]
 [0.2835308  0.13652217 0.17563666 0.12496987 0.2793405 ]
 [0.197388   0.08044084 0.31242317 0.17960033 0.2301476 ]
 [0.20849386 0.07715539 0.3034884  0.17557168 0.23529065]
 [0.3138043  0.16836175 0.32686952 0.08580077 0.10516363]] [[1. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0.]
 [0. 0. 0. 1. 0.]
 [1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1.]
 [1. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0.]]

'''




