#!/usr/bin/env python
# coding: utf-8




from keras.layers import Input,Dense,Embedding,Lambda,TimeDistributed,LSTM, Reshape, Dropout
from keras.models import Model
from TransformerEncoder import TransformerEncoder
from TrainablePositionalEmbeddings import TransformerPositionalEmbedding
import os
import pickle
from sklearn.model_selection import train_test_split
from keras.optimizers import Nadam, Adam
from Attention import MultiHeadedAttention
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
#import gc


# In[3]:

skulst = [30851, 59546, 32194, 31621, 48341]
#sku_id = 32194
n_in = 12
n_out = 6
n_test = 8
scaler = MinMaxScaler(feature_range=(0.02, 1))
file_name = 'data/sku_buc_day_20191210.csv'


# In[3]:


#data load
def load_data(file_name, sku_id):

    df = pd.read_csv(file_name)
    data_beer = df[df["sku_id"] == sku_id].copy()
    data_beer['time'] = pd.to_datetime(data_beer['time'], format='%Y-%m')
    data_beer['week'] = ((data_beer['time'] - pd.datetime(year=2014, month=1, day=6)).dt.days / 7).astype(int)
    data_beer_week = data_beer.groupby(['week'])['y'].sum().to_frame().sort_values(['week']).reset_index()
    data_beer_week = pd.DataFrame(data_beer_week.y.values).values
    
    return data_beer_week

#MinMaxScaler
def minmaxscaler(data):
    
    data = scaler.fit_transform(data)
    
    return data

# drop outlier val>mean+2std
def drop_outlier(data):
    
    data_beer_week = data
    data_beer_week = data_beer_week.reshape(len(data_beer_week),)
    point = np.mean(data_beer_week)+2*np.std(data_beer_week)
    for i in range(len(data_beer_week)):
        if data_beer_week[i] > point:
            data_beer_week[i] = point
        if data_beer_week[i] <= 0.1:
            data_beer_week[i] =0.1
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
    
    data = data_supervised
    data_train = data[:-(n_test+n_out-1),:]
    data_test = data[-n_test:,:]
    x_train = data_train[:,:-n_out]
    y_train = data_train[:,-n_out:]
    x_test = data_test[:,:-n_out]
    y_test = data_test[:,-n_out:]

#     y_train_new = []
#     for i in range(len(y_train)):
#         y_train_new.append(sum(y_train[i]))
#     y_train = np.array(y_train_new)
#     y_test_new = []
#     for i in range(len(y_test)):
#         y_test_new.append(sum(y_test[i]))
#     y_test = np.array(y_test_new)

    x_train = x_train.reshape(y_train.shape[0],n_in,1)
    x_test = x_test.reshape(y_test.shape[0],n_in,1)

    y_train = y_train.reshape(y_train.shape[0],n_out,1)
    y_test = y_test.reshape(y_test.shape[0],n_out,1)
    
    return x_train, x_test, y_train, y_test


# In[4]:








# In[5]:


import keras.backend as K

def custom_loss_acc(y_true, y_pred):
    """
    This is used to define the custom loss function. Accuracy!
    @y_true: 
    @y_pred: 
    """
    print(y_true, y_pred)
    return K.mean( K.abs(y_true - y_pred)) /K.mean( y_true )



# In[8]:


def time_series_model_LSTM_Transformer(input_seq_len, input_dim, heads, hidden_dim, positional_ff_dim, dropout_rate, output_seq_len, output_dim):
    
    encoder_inputs = Input((input_seq_len,input_dim,),dtype='float32',name='Encoder_input')
    positional_embedding_layer    = TransformerPositionalEmbedding(name='Positional_embedding')
    next_step_input               = positional_embedding_layer(encoder_inputs)
    
    next_step_input,attention =TransformerEncoder(hidden_dim,
                                                             heads,
                                                             hidden_dim,
                                                             hidden_dim,
                                                             positional_ff_dim,
                                                             dropout_rate= dropout_rate,
                                                             name= 'Transformer')(next_step_input)
    
    next_step_input,attention =TransformerEncoder(hidden_dim,
                                                         heads,
                                                         hidden_dim,
                                                         hidden_dim,
                                                         positional_ff_dim,
                                                         dropout_rate= dropout_rate,
                                                         name= 'Transformer1')(next_step_input)       
    
    state_h = Lambda(lambda x : x[:,0, :] )(next_step_input)
    state_c = Lambda(lambda x : x[:,1, :] )(next_step_input)
    
    decoder_inputs = Input(tensor=encoder_inputs[:,-(output_seq_len):,:])
    decoder = LSTM(hidden_dim,return_sequences=True,return_state=True,name='Decoder')
    predicted_values,_,_ = decoder(decoder_inputs,initial_state = [state_h,state_c])

    reshaped_outputs = TimeDistributed(Dense(output_dim))(predicted_values)

    return Model(inputs= [encoder_inputs,decoder_inputs],output=[reshaped_outputs])


# In[9]:


def data_split_new(num_test, x_train, y_train):
    x_train_set = x_train[0:len(x_train)-num_test,]
    y_train_set = y_train[0:len(y_train)-num_test,]
    x_test_set = x_train[len(x_train)-num_test:,]
    y_test_set = y_train[len(y_train)-num_test:,]
    
    return x_train_set, y_train_set, x_test_set, y_test_set


# In[11]:


def model_fit(x_train, y_train):    
    
    def objective(space): # Backtest return loss
        #count = 0
        #metric = list()
        #error = []
        #sum_volumn = []
        cusloss = []
        for index in range(len(x_train)//2, len(x_train),20):  # 
            
            data_x = x_train[0:index,]
            data_y = y_train[0:index,]
            
            x_train_set, y_train_set, x_test_set, y_test_set = data_split_new(1, data_x, data_y)
            
            print(space)
            model = time_series_model_LSTM_Transformer(
                                input_seq_len     = 12,
                                input_dim         = 1,
                                heads             = int(space['heads']),
                                hidden_dim        = int(space['hidden_dim']),
                                positional_ff_dim = int(space['positional_ff_dim']),
                                dropout_rate      = space['dropout_rate'],
                                #dropout_rate      = 0.5,
                                output_seq_len       = 6,
                                output_dim        = 1
            )
            
            model.compile(optimizer=Nadam(), loss = custom_loss_acc)
            model.fit(x_train_set,y_train_set, epochs=200, verbose=0)
            custom_loss = model.evaluate(x_test_set, y_test_set)
            cusloss.append(custom_loss)
            # fit into attention 
            #calculate metric
            #e,s = get_metrix_second(model, x_test_set, y_test_set)
            #print(e,s)
            #error.append(e)

            #sum_volumn.append(s)
            #acc = get_acc(model, x_test_set, y_test_set)
            #loss = 1 - acc
            #metric.append(loss)
            K.clear_session()
        
        
        #print(np.sum(error), np.sum(sum_volumn))
        #print(np.sum(error)/np.sum(sum_volumn))
        #ave = np.mean(metric)
        
        #print('acc',ave)
        #return{'loss': np.sum(error)/np.sum(sum_volumn), 'status': STATUS_OK }
        return{'loss': np.mean(cusloss), 'status': STATUS_OK }


    space = {
            'heads':            hp.quniform('heads', 1, 5, 1),
            'hidden_dim':       hp.quniform('hidden_dim', 50, 200, 25),
            'positional_ff_dim':hp.quniform('positional_ff_dim', 128, 512, 128),
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


def get_metrix(model, x_test_set, y_test_set):
    pred = model.predict(x_test_set)
    pred = scaler.inverse_transform(pred.reshape(1,-1))
    y_test = scaler.inverse_transform(y_test_set.reshape(1,-1))
    error = []
    summery = []
    for i in range(len(y_test[0])):
        if y_test[0][i] >1:
            error.append(abs(pred[0][i]-y_test[0][i]))
            summery.append(y_test[0][i])
    return np.sum(error), np.sum(summery)
            


# In[13]:


def get_metrix_second(model, x_test_set, y_test_set):
    pred = model.predict(x_test_set)
    pred = scaler.inverse_transform(pred.reshape(1,-1))
    y_test = scaler.inverse_transform(y_test_set.reshape(1,-1))
    print(pred[0][1], y_test[0][1])
    error = []
    summery = []
    error.append(abs(pred[0][1]-y_test[0][1]))
    summery.append(y_test[0][1])
    return np.sum(error), np.sum(summery)
            


# In[14]:


def get_acc(model, x_test_set, y_test_set):
    pred = model.predict(x_test_set)
    pred = scaler.inverse_transform(pred.reshape(n_out,-1))
    y_test = scaler.inverse_transform(y_test_set.reshape(n_out,-1))
    acc = []
#     print(y_test,pred)
    for i in range(len(y_test[0])):
        if y_test[0][i] >1:
            acc.append(1-abs(pred[0][i]-y_test[0][i])/y_test[0][i])
    return np.mean(acc)


# In[15]:


def get_predict(space, x_train, x_test, y_train, y_test):
    model = time_series_model_LSTM_Transformer(
                                input_seq_len     = 12,
                                input_dim         = 1,
                                heads             = int(space['heads']),
                                hidden_dim        = int(space['hidden_dim']),
                                positional_ff_dim = int(space['positional_ff_dim']),
                                dropout_rate      = space['dropout_rate'],
                                #dropout_rate      = 0.5,
                                output_seq_len       = 6,
                                output_dim        = 1
            )
    model.compile(optimizer=Nadam(), loss=custom_loss_acc)
    model.fit(x_train,y_train, epochs=200, verbose=0)
    e,s = get_metrix_second(model, x_test, y_test)
    #acc = get_acc(model, x_test, y_test)
    print(e,s,1-e/s)
    fitted = model.predict(x_train)
    pre = model.predict(x_test)

    #print('acc:',acc)
    #return 1-e/s,acc
    return fitted, pre, 1-e/s


# In[16]:
for sku_id in skulst:
    output = dict()
    output['SKU'] = sku_id
    data_beer_week = load_data(file_name, sku_id)
    data_beer_week = minmaxscaler(data_beer_week)
    data_beer_week = drop_outlier(data_beer_week)
    x_train, x_test, y_train, y_test = data_split(data_beer_week)

    best = model_fit(x_train, y_train)
    # pre,y_test = get_predict(best, x_train, x_test, y_train, y_test)
    # print(pre, y_test)
    print('best param: ')
    print(best)
    output['best_param'] = best


    fitted, pre, err= get_predict(best, x_train, x_test, y_train, y_test)
    print('data:')
    print(y_train, fitted, y_test, pre)
    print('1-e/s:')
    print(err)

    output['y_train'] = y_train
    output['fitted'] = fitted
    output['y_test'] = y_test
    output['pre'] = pre
    output['1-error'] = err

    with open('./output/'+str(sku_id)+'output.pkl','wb') as f:
        pickle.dump(output, f)
    print('SKU '+str(sku_id)+' finished')
    # In[8]:
print('All done!!')


    #hp.quniform('test',1,5,1)


# In[ ]:
'''
Loss: MAPE 间隔20
best param: 
{'heads': 2.0, 'hidden_dim': 175.0, 'positional_ff_dim': 384.0}
1961.5093 5778.432000000001
3816.9227226562507 5778.432000000001 0.3394535537224891
acc: 0.6090572345364561
predict
0.3394535537224891 0.6090572345364561

Loss: custom_loss 间隔20
best param: 
{'dropout_rate': 0.36326657717914684, 'heads': 4.0, 'hidden_dim': 150.0, 'positional_ff_dim': 512.0}
2739.0635 5778.432000000001
3039.3685234375007 5778.432000000001 0.4740150055521116
acc: 0.49223030352657793
predict
0.4740150055521116 0.49223030352657793

Loss: custom_loss 间隔4
best param: 
{'dropout_rate': 0.12203359553438016, 'heads': 5.0, 'hidden_dim': 175.0, 'positional_ff_dim': 384.0}
2373.6853 5778.432000000001
3404.7466972656257 5778.432000000001 0.41078363520317873
acc: 0.5330540936426007
predict
0.41078363520317873 0.5330540936426007
'''





# In[ ]:




