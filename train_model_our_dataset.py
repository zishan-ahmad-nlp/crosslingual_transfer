from __future__ import print_function
log = open("logs/hindi_baseline.txt", "w")

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
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score,classification_report
import keras
from keras.models import load_model
from keras.layers import Embedding
from keras.utils import to_categorical
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '4'

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
set_session(tf.Session(config=config))

ft = fastText.load_model("wiki.hi.bin")

nb_embedding_dims = ft.get_dimension()
nb_sequence_length = 75

# train_file = '/home1/zishan/Zishan/Emotion/News/Code/improve_semeval/data/total_data_with_intensity.txt'
# dev_file = '/home1/zishan/Zishan/Emotion/News/Code/improve_semeval/data/data_with_intensity_test.txt'

train_file = 'Data/train.txt'
dev_file = 'Data/dev.txt'


######-------Clean and Tokenize a Line-----###############
def twitter_tokenizer(textline):
    textLine = re.sub(r'http\S+', 'URL', textline)
    textline = re.sub('@[\w_]+', 'USER_MENTION', textline)
    textline = re.sub('\|LBR\|', '', textline)
    textline = re.sub('\.\.\.+', '...', textline)
    textline = re.sub('!!+', '!!', textline)
    textline = re.sub('\?\?+', '??', textline)
    words = re.compile('[\U00010000-\U0010ffff]|[\w-]+|[^ \w\U00010000-\U0010ffff]+', re.UNICODE).findall(textline.strip())
    words = [w.strip() for w in words if w.strip() != '']
    return words

labels2Idx = {'SADNESS': 0, 'FEAR/ANXIETY': 1, 'SYMPATHY/PENSIVENESS': 2, 'JOY': 3, 'OPTIMISM': 4, 'NO-EMOTION': 5, 'DISGUST': 6, 'ANGER': 7, 'SURPRISE': 8}

############-----------Create Dev Set---------###########
def get_dev_sentences(dev_file,check):
#    labels2Idx = {'anger': 0, 'fear': 1, 'joy': 2, 'sadness': 3}
    dev_lines = [line.strip().split("\t") for line in open(dev_file) if len(line.split('\t'))==check]
    dev_sentences = []
    dev_labels = []
    dev_intensity = []
    for x in dev_lines:
        if len(x) == check:
            dev_sentences.append(x[0])
            dev_labels.append(labels2Idx[x[1]])
            dev_intensity.append(float(x[2]))

    return (dev_sentences,dev_labels,dev_intensity)

#print(dev_sentences(dev_file,3))


##########---------Return Embedded Sentences in Batches-----------#############
word_vectors_ft = {}
def process_features(textline, nb_sequence_length, nb_embedding_dims, transform=False, tokenize=True):
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
            if transform:
                wv = np.matmul(wv, transmat)
            word_vectors_ft[w] = wv
        features_ft[idx] = wv
        idx = idx + 1
  return np.asarray([features_ft])

def sequential_generator(filename,batch_size,labels2idx,check,transform=False,tokenize=False):
    f=open(filename)
    n_labels=len(labels2idx)
    while True:
        batch_features_ft = np.zeros((batch_size,nb_sequence_length,nb_embedding_dims))
        batch_labels = np.zeros((batch_size,n_labels))
        batch_intensity = np.zeros((batch_size,1))
        for i in range(batch_size):
            line = f.readline()
            if (""==line):
                f.seek(0)
                line=f.readline()
            data=line.strip().split('\t')
            if check:
                if len(data) != check:
                    i=-1
                    continue
            batch_features_ft[i] = process_features(data[0],nb_sequence_length,nb_embedding_dims,transform)
            batch_labels[i] = to_categorical(labels2Idx[data[1]], n_labels)
            batch_intensity[i] = float(data[2])
            #print (labels2Idx[data[1]])
        
#        yield ([batch_features_ft],[data[2],float(data[3].split(':')[0])/10])

        yield ([batch_features_ft],[batch_labels,batch_intensity])


#transmat = np.loadtxt('/home1/zishan/raghav/fastText_multilingual/alignment_matrices/hi.txt')
#for i,j in sequential_generator(filename = train_file, batch_size = 16, check = 4,
 #       labels2idx= labels2Idx, transmat = transmat,transform=False, tokenize= True):
 #   print(i,j)

############------Build Model-----------############

def compile_model_bilstm_cnn(no_labels:'total labels for classification'):
    model_input_embedding = Input(shape = (nb_sequence_length, nb_embedding_dims))
    lstm_block = Bidirectional(LSTM(100, dropout = 0.5, return_sequences=True))(model_input_embedding)
    lstm_block = LeakyReLU()(lstm_block)
    filter_sizes = (3, 4, 5)
    conv_blocks = []
    for sz in filter_sizes:
        conv = Conv1D(filters = 200,kernel_size = sz,padding = 'valid', strides = 1)(lstm_block)
        conv = LeakyReLU()(conv)
        conv = GlobalMaxPooling1D()(conv)
        conv = Dropout(0.5)(conv)
        conv_blocks.append(conv)
    model_concatenated = concatenate([conv_blocks[0], conv_blocks[1], conv_blocks[2]])
    model_concatenated = Dense(100,name='outer_dense')(model_concatenated)
    model_concatenated = LeakyReLU()(model_concatenated)
    model_output = Dense(no_labels, activation = "softmax", name='classes')(model_concatenated)
    model_output_intensity = Dense(1, activation='sigmoid', name='intensity')(model_concatenated)
    new_model = Model(model_input_embedding, [model_output,model_output_intensity])
    new_model.compile(loss=['categorical_crossentropy','mse'], optimizer='nadam', metrics = ['accuracy'])
        #new_model.summary()
    return new_model

#print (compile_model_bilstm_cnn(13).summary())

########---Setting Parameters---#########
batch_size = 16
check_for_generator = 3
# transmat = np.loadtxt('/home1/zishan/raghav/fastText_multilingual/alignment_matrices/hi.txt')
tokenize = True

train_sentences,train_labels,_ = get_dev_sentences(train_file,3)
print('train_sent',len(train_sentences))
samples_per_epoch = len(train_sentences)
steps_per_epoch = math.ceil(samples_per_epoch / batch_size)

#############---Performance without Transfer Learning---############
no_transfer_weights = 'weights/hindi_baseline.hdf5'
no_tl = compile_model_bilstm_cnn(len(labels2Idx))
callback = keras.callbacks.ModelCheckpoint(no_transfer_weights, monitor='classes_acc', save_best_only=True, save_weights_only=True)
no_tl.fit_generator(sequential_generator(filename = train_file, batch_size = batch_size, check = check_for_generator, 
                                                 labels2idx= labels2Idx, transform=False, tokenize= tokenize), verbose=2,
                                                                         steps_per_epoch= steps_per_epoch, epochs=100, callbacks = [callback])

#no_tl.save(no_transfer_weights)

no_tl = compile_model_bilstm_cnn(len(labels2Idx))
no_tl.load_weights(no_transfer_weights)

dev_sentences,dev_labels,dev_intensity = get_dev_sentences(dev_file,3)

testset_features = np.zeros((len(dev_sentences), nb_sequence_length, nb_embedding_dims))

for i in range(len(dev_sentences)):
    testset_features[i] = process_features(dev_sentences[i], nb_sequence_length, nb_embedding_dims, transform = False)

results_classes, results_intensity = no_tl.predict(testset_features)



predLabels = results_classes.argmax(axis=-1)
devLabels = dev_labels

correlation = np.corrcoef([np.asarray(dev_intensity).reshape(len(dev_intensity)), np.asarray(results_intensity).reshape(len(results_intensity))])
cosine = cosine_similarity(np.asarray(dev_intensity).reshape(1,len(dev_intensity)), np.asarray(results_intensity).reshape(1,len(dev_intensity)))
mse = mean_squared_error(np.asarray(dev_intensity).reshape(len(dev_intensity)), np.asarray(results_intensity).reshape(len(results_intensity)))
#print ('real_labels-->',devLabels)
print('-----------Performance Without Transfer Learning on Our Data--------',file=log)
print('F1-Score micro',f1_score(devLabels, predLabels, average='micro'),file=log)
print('F1-Score macro',f1_score(devLabels, predLabels, average='macro'),file=log)
print(classification_report(devLabels,predLabels),file=log)

print('------Intensity Scores---------',file=log)
print('correlation = ',correlation,file=log)
print('cosine = ',cosine,file=log)
print('mse = ', mse,file=log)

#a = accuracy_score(devLabels, predLabels)
#print(f1,a)
