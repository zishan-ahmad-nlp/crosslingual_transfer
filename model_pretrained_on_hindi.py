
# coding: utf-8

# In[1]:


import fastText
import math
import linecache
import numpy as np 
from numpy import random
from random import sample
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint
from keras.layers import *
from keras import *
import keras
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import RandomUniform
import re
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score


# In[2]:


from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
set_session(tf.Session(config=config))


# In[3]:


ft = fastText.load_model("wiki.hi.bin")

nb_embedding_dims = ft.get_dimension()
nb_sequence_length = 75


# In[4]:


def twitter_tokenizer(textline):
    textLine = re.sub(r'http\S+', 'URL', textline)
    textline = re.sub('@[\w_]+', 'USER_MENTION', textline)
    textline = re.sub('\|LBR\|', '', textline)
    textline = re.sub('\.\.\.+', '...', textline)
    textline = re.sub('!!+', '!!', textline)
    textline = re.sub('\?\?+', '??', textline)
    words = re.compile('[\U00010000-\U0010ffff]|[\w-]+|[^ \w\U00010000-\U0010ffff]+', re.UNICODE).findall(textline.strip())
    words = [w.strip() for w in words if w.strip() != '']
    # print(words)
    return(words)


# In[5]:


word_vectors_ft = {}
def process_features(textline, nb_sequence_length, nb_embedding_dims, tokenize=True):
    if not tokenize:
        words = textline.split()
    else:
        words = twitter_tokenizer(textline)
    features_ft = np.zeros((nb_sequence_length, nb_embedding_dims))
    features_idx = np.zeros(nb_sequence_length)
    max_words = min(len(words), nb_sequence_length)
    idx = nb_sequence_length - len(words[:max_words])
    for w in words[:max_words]:
        if w in word_vectors_ft:
            wv = word_vectors_ft[w]
        else:
            wv = ft.get_word_vector(w.lower())
            word_vectors_ft[w] = wv
        features_ft[idx] = wv
        
        idx = idx + 1
    return features_ft


# In[6]:


def sequential_generator(filename, 
                         batch_size, 
                         labels2Idx:'dict to make output labels',
                         check:'to check if all lines in file are of same length.To check enter the len of line after splitting it by tabs' = None,
                         tokenize:'specify if using twitter tokenzor to preprocess lines'=False, 
                        ):    
    
    f = open(filename)
    n_labels = len(labels2Idx)
    while True:
        batch_features_ft = np.zeros((batch_size, nb_sequence_length, nb_embedding_dims))
        batch_labels = np.zeros((batch_size, len(labels2Idx)))
        for i in range(batch_size):
            line = f.readline()
            if ("" == line):
                f.seek(0)
                line = f.readline()
            data = line.strip().split('\t')
            if check:
                if len(data)!=check:
                    i-=1
                    continue
            batch_features_ft[i] = process_features(data[0], nb_sequence_length, nb_embedding_dims, tokenize= tokenize)
            if len(labels2Idx)==2:
                batch_labels[i] = to_categorical(0 if data[1] == 'OTHER' else 1, n_labels)
            else:
                batch_labels[i] = to_categorical(labels2Idx[data[1]], n_labels)
        yield ([batch_features_ft], batch_labels)


# In[7]:


def train_dev_sentences(filetrain, filedev, check:'to check if lines of file are all same lenght after separating by tab'):
    labels2Idx = {}
    train_lines = [line.strip().split("\t") for line in open(filetrain) if len(line.split('\t'))==check]
    dev_lines = [line.strip().split("\t") for line in open(filedev) if len(line.strip().split('\t'))==check]
    train_sentences = [x[0] for x in train_lines]
    for dataset in [train_lines, dev_lines]:
        for line in dataset:
            label = line[1]
            if label not in labels2Idx.keys():
                labels2Idx[label]= len(labels2Idx)
                
    train_labels = [labels2Idx[x[1]] for x in train_lines]
    dev_sentences = [x[0] for x in dev_lines]
    dev_labels = [labels2Idx[x[1]] for x in dev_lines]
    return (train_sentences, train_labels, dev_sentences, dev_labels, labels2Idx)


# In[8]:


train_file = 'Data/movie_sentiments.txt'
dev_file = 'Data/review_sentiments.txt'
train_sentences, train_labels, dev_sentences, dev_labels, labels2Idx = train_dev_sentences(train_file, dev_file, 2)


# In[9]:


print(set(train_labels))
print(set(dev_labels))
print(labels2Idx)


# In[10]:


print(len(train_sentences))
print(len(dev_sentences))


# In[11]:


train_sentences.extend(dev_sentences)
train_labels.extend(dev_labels)


# In[12]:


print(len(train_sentences))


# In[13]:


print(labels2Idx)
labels2Idx.pop('cpn',None)
print(labels2Idx)


# In[14]:


from collections import Counter
print(Counter(train_labels))
print(Counter(dev_labels))


# In[ ]:


def compile_model_bilstm(no_labels:'total labels for classification'):
    model_input_embedding = Input(shape = (nb_sequence_length, nb_embedding_dims))
    lstm_block = Bidirectional(LSTM(100, dropout = 0.5, return_sequences=True))(model_input_embedding)
    lstm_block = LeakyReLU()(lstm_block)
    model_concatenated = Flatten()(lstm_block)
    model_concatenated = Dense(100)(model_concatenated)
    model_output = Dense(no_labels, activation = "softmax")(model_concatenated)
    new_model = Model(model_input_embedding, model_output)
    new_model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics = ['accuracy'])
    new_model.summary()
    return new_model


# In[ ]:


def compile_model_bilstm_3cnn(no_labels:'total labels for classification'):
    model_input_embedding = Input(shape = (nb_sequence_length, nb_embedding_dims))
    lstm_block = Bidirectional(LSTM(100, dropout = 0.5, return_sequences=True))(model_input_embedding)
    lstm_block = LeakyReLU()(lstm_block)

    filter_sizes = (3, 4, 5)
    conv_blocks = []
    for sz in filter_sizes:
        conv = Conv1D(
            filters = 200,
            kernel_size = sz,
            padding = 'valid',
            strides = 1
        )(lstm_block)
        conv = LeakyReLU()(conv)
        conv = GlobalMaxPooling1D()(conv)
        conv = Dropout(0.5)(conv)
        conv_blocks.append(conv)
    model_concatenated = concatenate([conv_blocks[0], conv_blocks[1], conv_blocks[2]])
    model_concatenated = Dense(100)(model_concatenated)
    model_concatenated = LeakyReLU()(model_concatenated)
    model_output = Dense(no_labels, activation = "softmax")(model_concatenated)
    new_model = Model(model_input_embedding, model_output)
    new_model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics = ['accuracy'])
    new_model.summary()
    return new_model


# In[ ]:


no_labels = len(labels2Idx)


# In[ ]:


model = compile_model_bilstm_3cnn(no_labels)


# In[ ]:


train_file = 'Data/movie_and_review_sentiments.txt'
weights_file ='weights/pretrain_bilstm_3cnn_dropout=0.5.h5'
# log_file = '/home1/zishan/raghav/logs/bilstm.txt'
batch_size = 16
check_for_generator = 2
labels2Idx = labels2Idx
tokenize = True
samples_per_epoch = len(train_sentences)
steps_per_epoch = math.ceil(samples_per_epoch / batch_size)


# In[ ]:


callback = keras.callbacks.ModelCheckpoint(weights_file, monitor='acc', verbose=0, save_best_only=True, save_weights_only=True)


# In[ ]:


model.fit_generator(sequential_generator(filename = train_file, batch_size = batch_size, check = check_for_generator, 
                                             labels2Idx= labels2Idx,tokenize= tokenize),
                        steps_per_epoch= steps_per_epoch, epochs=20, callbacks = [callback])

