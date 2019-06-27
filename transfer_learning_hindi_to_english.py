from __future__ import print_function
log = open("logs/improve_semeval/transfer_learning_strategies_gradual_unfreeze_3.txt", "w")
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
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
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score,classification_report,precision_recall_fscore_support
import keras
from keras.models import load_model
from keras.layers import Embedding
from keras.utils import to_categorical
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import argparse

parser = argparse.ArgumentParser(description='select transfer strategy')
parser.add_argument('--strategy', default='au', type=str, metavar='N', help='au, guf, sutd, subu')

args = parser.parse_args()

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
set_session(tf.Session(config=config))


ft = fastText.load_model("wiki.en.bin")

nb_embedding_dims = ft.get_dimension()
#nb_embedding_dims = 300
nb_sequence_length = 75

train_file = 'data/train.txt'
dev_file = 'data/test.txt'


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

labels2Idx = {'anger': 0, 'fear': 1, 'joy': 2, 'sadness': 3}

############-----------Create Dev Set---------###########
def get_dev_sentences(dev_file,check):
    labels2Idx = {'anger': 0, 'fear': 1, 'joy': 2, 'sadness': 3}
    dev_lines = [line.strip().split("\t") for line in open(dev_file) if len(line.split('\t'))==check]
    dev_sentences = []
    dev_labels = []
    dev_intensity = []
    for x in dev_lines:
        if len(x) == check:
            dev_sentences.append(x[1])
            dev_labels.append(labels2Idx[x[2]])
            dev_intensity.append(float(x[3].split(':')[0])/3)

    return (dev_sentences,dev_labels,dev_intensity)

#print(dev_sentences(dev_file,3))


##########---------Return Embedded Sentences in Batches-----------#############
word_vectors_ft = {}
def process_features(textline, nb_sequence_length, nb_embedding_dims, transmat, transform=True, tokenize=True):
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

def sequential_generator(filename,batch_size,labels2idx,transmat,check,transform=True,tokenize=False):
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
            batch_features_ft[i] = process_features(data[1],nb_sequence_length,nb_embedding_dims,transmat,transform)
            batch_labels[i] = to_categorical(labels2Idx[data[2]], n_labels)
            batch_intensity[i] = float(data[3].split(':')[0])/3
            #print (labels2Idx[data[1]])

#        yield ([batch_features_ft],[data[2],float(data[3].split(':')[0])/10])

        yield ([batch_features_ft],[batch_labels,batch_intensity])


#transmat = np.loadtxt('/home1/zishan/raghav/fastText_multilingual/alignment_matrices/en.txt')
#for i,j in sequential_generator(filename = train_file, batch_size = 16, check = 4,
#        labels2idx= labels2Idx, transmat = transmat,transform=False, tokenize= True):
#	print(i,j)

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
check_for_generator = 4
transmat = np.loadtxt('fastText_multilingual/alignment_matrices/en.txt')
tokenize = True

dev_sentences,dev_labels, dev_intensity = get_dev_sentences(dev_file,4)
train_sentences,train_labels,_ = get_dev_sentences(train_file,4)
print('train_sent',len(train_sentences))
samples_per_epoch = len(train_sentences)
steps_per_epoch = math.ceil(samples_per_epoch / batch_size)

#########--Test Set Load---#########
testset_features = np.zeros((len(dev_sentences), nb_sequence_length, nb_embedding_dims))

for i in range(len(dev_sentences)):
    testset_features[i] = process_features(dev_sentences[i], nb_sequence_length, nb_embedding_dims, transmat, transform = True)


######---Loading Model----#########
transfer_weights_now = 'weights/our_data_full_no_transfer_weight.hdf5'
no_tl = compile_model_bilstm_cnn(9)
no_tl.load_weights(transfer_weights_now)

for i,j in enumerate(no_tl.layers):
	print(i, j.name)

#print(no_tl.layers)

output = Dense(len(labels2Idx), activation = 'softmax', name='classes')(no_tl.layers[17].output)
output1 = Dense(1, activation = 'sigmoid', name='intensity')(no_tl.layers[17].output)
final_model_transfer_last = Model(inputs=no_tl.layers[0].input, outputs=[output,output1])
final_model_transfer_last.compile(loss=['categorical_crossentropy','mse'], optimizer='nadam', metrics=['accuracy'])
final_model_transfer_last.summary()


single_unfreeze_bottom_up = [(18, 18), (17, 16), (15, 3), (2, 1), (18,1)]
single_unfreeze_top_down = [(18, 18),   (2, 1),(15, 3), (17, 16), (18,1)]
all_unfreeze = [(18,1)]
gradual_unfreezing = [(18,18), (18,16), (18,3), (18,1)]

strategy = {'au': all_unfreeze, 'guf': gradual_unfreezing, 'subu': single_unfreeze_bottom_up, 'sutd': single_unfreeze_top_down}

z=0
for i,j in strategy[args.strategy]:
	z=z+1
	for k,l in enumerate(final_model_transfer_last.layers):
		if k<=i and k>=j:
		    l.trainable=True
		else:
		    l.trainable=False

	final_model_transfer_last.compile(loss=['categorical_crossentropy','mse'], optimizer='nadam', metrics=['accuracy'])
	transfer_inner_weights = '/home1/zishan/Zishan/Emotion/News/Models/sem_eval_improve/our_data_transfer_gradual_unfreezing'+str(i)+'_'+str(j)+'_fold3.hdf5'

	callback = keras.callbacks.ModelCheckpoint(transfer_inner_weights, monitor='classes_acc', save_best_only=True, save_weights_only=True)

	final_model_transfer_last.fit_generator(sequential_generator(filename = train_file, batch_size = batch_size, check = check_for_generator,
		                                                 labels2idx= labels2Idx, transmat = transmat,transform=True, tokenize= tokenize),
		                                                                                                                          steps_per_epoch= steps_per_epoch, epochs=200, verbose=2, callbacks = [callback])


	tl_inner = compile_model_bilstm_cnn(len(labels2Idx))
	tl_inner.load_weights(transfer_inner_weights)

	results,results_intensity = tl_inner.predict(testset_features)

	predLabels = results.argmax(axis=-1)
	devLabels = dev_labels

	correlation = np.corrcoef([np.asarray(dev_intensity).reshape(len(dev_intensity)), np.asarray(results_intensity).reshape(len(results_intensity))])
	cosine = cosine_similarity(np.asarray(dev_intensity).reshape(1,len(dev_intensity)), np.asarray(results_intensity).reshape(1,len(dev_intensity)))
	mse = mean_squared_error(np.asarray(dev_intensity).reshape(len(dev_intensity)), np.asarray(results_intensity).reshape(len(results_intensity)))
	#print ('real_labels-->',devLabels)
	print('-----------Performance on transfer learning layer ',i,'to layer',j,'Fold 3--------',file=log)
	print(precision_recall_fscore_support(devLabels, predLabels, average='micro'), file=log)
	print(precision_recall_fscore_support(devLabels, predLabels, average='macro'), file=log)
#	print('F1-Score micro',f1_score(devLabels, predLabels, average='micro'),file=log)
#	print('F1-Score macro',f1_score(devLabels, predLabels, average='macro'),file=log)
	print(classification_report(devLabels,predLabels),file=log)

	print('------Intensity Scores---------',file=log)
	print('correlation = ',correlation,file=log)
	print('cosine = ',cosine,file=log)
	print('mse = ', mse,file=log)
	final_model_transfer_last = compile_model_bilstm_cnn(len(labels2Idx))
	final_model_transfer_last.load_weights(transfer_inner_weights)
