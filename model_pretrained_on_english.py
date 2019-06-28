
# coding: utf-8

# In[1]:


import fastText
import sys
import math
import linecache
import numpy as np 
from numpy import random
from random import sample
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint
from keras.layers import *
from keras import *
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import RandomUniform
import re
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import keras
import argparse

parser = argparse.ArgumentParser(description='Pretraining on english datasets')

parser.add_argument('--data', default='Data/text_emotion_crowdflower.csv', type=str, metavar='N', help='data for pretraining')
# In[2]:

args = parser.parse_args()

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
set_session(tf.Session(config=config))


# In[3]:


ft = fastText.load_model("wiki.en.bin")

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
def process_features(textline, nb_sequence_length, nb_embedding_dims, tokenize=True, transmat = None):
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
            wv = np.matmul(wv, transmat) # applying transformation on the word vector to make the vector in same space
            word_vectors_ft[w] = wv
        features_ft[idx] = wv
        
        idx = idx + 1
    return features_ft


# In[7]:


def sequential_generator_semeval(filename, 
                         batch_size, 
                         labels2Idx:'dict to make output labels',
                         transmat:'Matrix to make embeddings in same vector space'= None,
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
            batch_features_ft[i] = process_features(data[1], nb_sequence_length, nb_embedding_dims, tokenize= tokenize, transmat = transmat)
            if len(labels2Idx)==2:
                batch_labels[i] = to_categorical(0 if data[2] == 'OTHER' else 1, n_labels)
            else:
                batch_labels[i] = to_categorical(labels2Idx[data[2]], n_labels)
#         print(batch_features_ft.shape, batch_labels)
        yield ([batch_features_ft], batch_labels)


# In[8]:


def sequential_generator_crowdflower(filename, 
                         batch_size, 
                         labels2Idx:'dict to make output labels',
                         transmat:'Matrix to make embeddings in same vector space'= None,
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
            data = line.strip().split(',')
            if check:
                if len(data)!=check:
                    i-=1
                    continue
            batch_features_ft[i] = process_features(data[3], nb_sequence_length, nb_embedding_dims, tokenize= tokenize, transmat = transmat)
            if len(labels2Idx)==2:
                batch_labels[i] = to_categorical(0 if data[1] == 'OTHER' else 1, n_labels)
            else:
                try:
                    #print(data[1])
                    batch_labels[i] = to_categorical(labels2Idx[data[1]], n_labels)
                except:
                    i-=1
                    continue
#         print(batch_features_ft.shape, batch_labels)
        yield ([batch_features_ft], batch_labels)


# In[9]:


def train_dev_sentences_semeval(filetrain, filedev, check:'to check if lines of file are all same lenght after separating by tab'):
    labels2Idx = {}
    train_lines = [line.strip().split("\t") for line in open(filetrain) if len(line.split('\t'))==check]
    dev_lines = [line.strip().split("\t") for line in open(filedev) if len(line.strip().split('\t'))==check]
    train_sentences = [x[1] for x in train_lines]
    for dataset in [train_lines, dev_lines]:
        for line in dataset:
            label = line[2]
            if label not in labels2Idx.keys():
                labels2Idx[label]= len(labels2Idx)
                
    train_labels = [labels2Idx[x[2]] for x in train_lines]
    dev_sentences = [x[1] for x in dev_lines]
    dev_labels = [labels2Idx[x[2]] for x in dev_lines]
    return (train_sentences, train_labels, dev_sentences, dev_labels, labels2Idx)


# In[10]:


def train_dev_sentences_crowdflower(filetrain, filedev, check:'to check if lines of file are all same lenght after separating by tab'):
    labels2Idx = {}
    train_lines = [line.strip().split(",") for line in open(filetrain) if len(line.split(','))==check]
    dev_lines = [line.strip().split(",") for line in open(filedev) if len(line.strip().split(','))==check]
#     print(train_lines[0])
    del train_lines[0]
    del dev_lines[0]
    train_sentences = [x[3] for x in train_lines]
    for dataset in [train_lines, dev_lines]:
        for line in dataset:
            label = str(line[1])
#             print(label)
#             label.replace('"','')
#             print(label)
            if label not in labels2Idx.keys():
                labels2Idx[label]= len(labels2Idx)
                
    train_labels = [labels2Idx[x[1]] for x in train_lines]
    dev_sentences = [x[3] for x in dev_lines]
    dev_labels = [labels2Idx[x[1]] for x in dev_lines]
    return (train_sentences, train_labels, dev_sentences, dev_labels, labels2Idx)


# In[14]:

if args.data == 'crowd':
    train_file = '/home1/zishan/raghav/Data/text_emotion_crowdflower.csv'
    dev_file = '/home1/zishan/raghav/Data/text_emotion_crowdflower.csv'
    train_sentences, train_labels, dev_sentences, dev_labels, labels2Idx = train_dev_sentences_crowdflower(train_file, dev_file, 4)


# In[12]:

if args.data =='semeval':
    train_file = '/home1/zishan/raghav/Data/semEval2017.txt'
    dev_file = '/home1/zishan/raghav/Data/semEval2017.txt'
    train_sentences, train_labels, dev_sentences, dev_labels, labels2Idx = train_dev_sentences_semeval(train_file, dev_file, 4)


# In[15]:


from collections import Counter
print(set(train_labels))
print(labels2Idx)
print(train_sentences[0])
print(Counter(train_labels))


# In[13]:


n_words = 0
for sentence in train_sentences:
    n_words+=len(sentence)
print(n_words)


# In[14]:


n_labels = len(labels2Idx)


# In[15]:


def compile_model_bilstm_cnn(no_labels:'total labels for classification'):
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


# In[16]:


model = compile_model_bilstm_cnn(no_labels = n_labels)


# In[17]:

if args.data == 'crowd':
    train_file = 'Data/text_emotion_crowdflower.csv'
    weights_file ='weights/pretrain_crowdflower_bilstm_3cnn.h5'

if args.data == 'semeval':
    train_file = 'Data/semEval2017.txt'
    weights_file ='weights/pretrain_semeval_bilstm_3cnn.h5'

# log_file = '/home1/zishan/raghav/logs/bilstm_3cnn_dropout=0.5.txt'
batch_size = 16
check_for_generator = 4
labels2Idx = labels2Idx
tokenize = True
transmat = np.loadtxt('fastText_multilingual/alignment_matrices/en.txt')
samples_per_epoch = len(train_sentences)
steps_per_epoch = math.ceil(samples_per_epoch / batch_size)


# In[18]:


callback = keras.callbacks.ModelCheckpoint(weights_file, monitor='acc', verbose=0, save_best_only=True, save_weights_only=True)


# In[27]:

if args.data == 'crowd':
    model.fit_generator(sequential_generator_crowdflower(filename = train_file, batch_size = batch_size, check = check_for_generator, 
                                                 labels2Idx= labels2Idx, transmat = transmat, tokenize= tokenize),
                            steps_per_epoch= steps_per_epoch, epochs=100, callbacks = [callback])

if args.data == 'semeval':
    model.fit_generator(sequential_generator_semeval(filename = train_file, batch_size = batch_size, check = check_for_generator, 
                                                 labels2Idx= labels2Idx, transmat = transmat, tokenize= tokenize),
                            steps_per_epoch= steps_per_epoch, epochs=100, callbacks = [callback])


# In[6]:


from nltk.corpus import stopwords
import nltk


# In[21]:


train_file = 'Data/semEval2017.txt'
dev_file = 'Data/semEval2017.txt'
train_sentences, train_labels, dev_sentences, dev_labels, labels2Idx = train_dev_sentences_semeval(train_file, dev_file, 4)


# In[22]:


x = train_sentences[:2]
print(x)


# In[23]:


stop_words = set(stopwords.words('english'))
not_stop_words =[]
count=0
for dataset in [train_sentences, dev_sentences]:
    for sentence in dataset:
#         print(stopwords.words(sentence))
        temp = [i for i in sentence.split() if i not in stop_words]
        not_stop_words.extend(temp)
        
print(len(not_stop_words))
print(len(set(not_stop_words)))
print(list(set(not_stop_words))[:10])
print(not_stop_words[:10])


# In[24]:


count =0
for dataset in [train_sentences, dev_sentences]:
    for sentence in dataset:
        count+=len(sentence)
print(count)

