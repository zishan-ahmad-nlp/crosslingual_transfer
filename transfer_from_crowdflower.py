
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
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import RandomUniform
import re
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score


# In[2]:


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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


train_file = 'Data/train_total.txt'
dev_file = 'Data/test_total.txt'
train_sentences, train_labels, dev_sentences, dev_labels, labels2Idx = train_dev_sentences(train_file, dev_file, 3)


# In[9]:


n_words = 0
for sentence in train_sentences:
    n_words+=len(sentence)
print(n_words)


# In[10]:


from collections import Counter
print(Counter(train_labels))
print(labels2Idx)


# In[11]:


print(train_sentences[:2])
print(train_labels[:2])
print(labels2Idx)
print(len(train_labels))


# In[12]:


n_labels = len(labels2Idx)


# In[13]:



# In[14]:


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


# # Pretrain

# In[ ]:


model = compile_model_bilstm_cnn(no_labels = n_labels)


# In[ ]:


train_file = '/home1/zishan/raghav/Data/train.txt'
weights_file ='/home1/zishan/raghav/weights/bilstm_3cnn_dropout=0.5.h5'
log_file = '/home1/zishan/raghav/logs/bilstm_3cnn_dropout=0.5.txt'
batch_size = 16
check_for_generator = 2
tokenize = True
samples_per_epoch = len(train_sentences)
steps_per_epoch = math.ceil(samples_per_epoch / batch_size)


# In[ ]:



# # transfer learning

# In[15]:


word_vectors_ft = {}
def process_features_crosslingual(textline, nb_sequence_length, nb_embedding_dims, tokenize=True, transmat = None):
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
#             wv = np.matmul(wv, transmat) # applying transformation on the word vector to make the vector in same space
            word_vectors_ft[w] = wv
        features_ft[idx] = wv
        
        idx = idx + 1
    return features_ft


# In[16]:


def sequential_generator_crosslingual(filename, 
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
            batch_features_ft[i] = process_features_crosslingual(data[0], nb_sequence_length, nb_embedding_dims, tokenize= tokenize, transmat = transmat)
            if len(labels2Idx)==2:
                batch_labels[i] = to_categorical(0 if data[1] == 'OTHER' else 1, n_labels)
            else:
                batch_labels[i] = to_categorical(labels2Idx[data[1]], n_labels)
#         print(batch_features_ft.shape, batch_labels)
        yield ([batch_features_ft], batch_labels)


# In[17]:


def test_model_tl_unfreezing(generator, 
               train_sentences, 
               devLabels, 
               number_of_tests,
               number_of_epochs,
               filename_to_log, 
               labels2Idx,
               filename_to_save_weigths,
               batch_size, 
               unfreezing_strategy: 'list containing a tuple of indices to unfreeze at each step',
               train_file:'filepath for traininig',
               f1_measure:'binary/macro etc', 
               pos_label:'only if binary f1',
               load_model_weights=False,
               model_weights_file:'give filepath as str'=None, 
               tokenize=True,
               nb_sequence_length = nb_sequence_length,
               nb_embedding_dims= nb_embedding_dims, 
               transmat: 'matrix if crosslingual training'=None,
               check_for_generator=None):
    
    f = open(filename_to_log, 'w', encoding='utf-8')
    f.close()
   
    total_f1=0
    total_prec=0
    total_acc=0
    total_recall=0
    
    for test_number in range(number_of_tests):
        print("Test %d/%d" %(test_number+1, number_of_tests))
        model = compile_model_bilstm_cnn(13)

        # transfer learning
        if load_model_weights and model_weights_file:
                model.load_weights(model_weights_file)
                print("removing top layer")
                model.layers.pop()
                output = Dense(len(labels2Idx), activation = 'softmax')(model.layers[-1].output)
                final_model = Model(inputs=model.input, outputs=[output])

        samples_per_epoch = len(train_sentences)
        epochs = number_of_epochs
        batch_size = batch_size
        steps_per_epoch = math.ceil(samples_per_epoch / batch_size)

        max_f1=0
        max_p=0
        max_r=0
        max_a=0
        
        # load pretrained weights
        # model.compile
        # save tmp weights
        # iterate over layers
        #    load tmp weights
        #    iterate over epochs
        #        unfreeze top frozen layer
        #        save best model as tmp weights
        
        
        final_model.save_weights(filename_to_save_weigths)
        
        # layers_to_unfreeze = [18, 16, 3, 1]
        
        for ulayer in unfreezing_strategy:
            text = "************\nUnfreezing {}\n****************\n".format(final_model.layers[ulayer[0]].name)
            with open(filename_to_log,'a') as f:
                f.write(text)         
            print(text)
            print("---------------------------------------")
            final_model.load_weights(filename_to_save_weigths)            
            for i, layer in enumerate(final_model.layers):
                
                # TF strategy: gradual unfreezing
                #if i >= ulayer:
                #    layer.trainable = True
                #else:
                #    layer.trainable = False
                # 
                ## TF strategy: single
                
                if i >= ulayer[1] and i <= ulayer[0]:
                    layer.trainable = True
                else:
                    layer.trainable = False
                    
                print(str(i) + ' ' + layer.name + ' ' + str(layer.trainable))
            final_model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
        
            for epoch in range(epochs):
                print("Epoch: %d/%d" %(epoch+1, epochs))
                final_model.fit_generator(
                    generator(filename = train_file, batch_size = batch_size, check = check_for_generator, 
                              labels2Idx= labels2Idx, transmat = transmat, tokenize= tokenize), 
                    steps_per_epoch= steps_per_epoch, epochs=1
                )

                testset_features = np.zeros((len(dev_sentences), nb_sequence_length, nb_embedding_dims))
                for i in range(len(dev_sentences)):
                    testset_features[i] = process_features_crosslingual(dev_sentences[i], nb_sequence_length, nb_embedding_dims, transmat= transmat)
                results = final_model.predict(testset_features)

                predLabels = results.argmax(axis=-1)
                devLabels = devLabels
                f1 = f1_score(devLabels, predLabels, average=f1_measure, pos_label=pos_label) # offensive is the major class. So other is minor
                r = recall_score(devLabels, predLabels, average=f1_measure, pos_label=pos_label)
                p = precision_score(devLabels, predLabels, average=f1_measure, pos_label=pos_label)
                a = accuracy_score(devLabels, predLabels)
                if max_f1 < f1:
                    print("model saved. F1 is %f" %(f1))
                    final_model.save(filename_to_save_weigths)
                    max_f1 = f1
                    max_p = p
                    max_r = r
                    max_a = a
                    to_write = "p:{0}\t r:{1}\t f1(macro):{2}\t acc:{3}\n".format(max_p,max_r,max_f1,max_a)
                    with open(filename_to_log,'a') as f:
                        f.write(to_write)
                        
                text = "prec: "+ str(p)+" rec: "+str(r) +" f1: "+str(f1) +" acc: "+str(a)+" \n"
                print("Test-Data: Prec: %.3f, Rec: %.3f, F1: %.3f, Acc: %.3f" % (p, r, f1, a))
                
        to_write = "p:{0}\t r:{1}\t f1(macro):{2}\t acc:{3}\n".format(max_p,max_r,max_f1,max_a)
        print(to_write)
        with open(filename_to_log,'a') as f:
            f.write(to_write)
        total_f1+=max_f1
        total_prec+=max_p
        total_acc+=max_a
        total_recall+=max_r    
        print("*****************************************************************************")
    final_text = "avg_prec: " +str(total_prec/number_of_tests)+" total_rec: "+str(total_recall/number_of_tests) +" total_f1: "+str(total_f1/number_of_tests) +" total_acc: "+str(total_acc/number_of_tests)+" \n"
    print(final_text)
    with open(filename_to_log,'a') as f:
        f.write(final_text)


# In[18]:


# list of tuples. Every tuple contains range of layers which need to be unfrozen. Rest all are frozen
single_unfreeze_bottom_up = [(18, 18), (17, 16), (15, 3), (2, 1), (18,1)] 
single_unfreeze_top_down = [(18, 18),   (2, 1),(15, 3), (17, 16), (18,1)]
all_unfreeze = [(18,1)]
gradual_unfreezing = [(18,18), (18,16), (18,3), (18,1)]

strings =['suf_bu', 'suf_td','all_unfreeze','gradual_unfreeze']
# strings = ['suf_td','all_unfreeze', 'gradual_unfreeze']
# strings=['gradual_unfreeze']
unfreeze_strategy = [single_unfreeze_bottom_up, single_unfreeze_top_down, all_unfreeze, gradual_unfreezing]
# unfreeze_strategy = [gradual_unfreezing]
# unfreeze_strategy = [ single_unfreeze_top_down, all_unfreeze, gradual_unfreezing]


# In[ ]:


for test_number in range(1):    
    for i in range(len(strings)):
        string = strings[i]
        print("approach: %s" %(string))

        generator = sequential_generator_crosslingual
        train_sentences = train_sentences
        devLabels = dev_labels
        number_of_tests = 1
        transmat = np.loadtxt('/home1/zishan/raghav/fastText_multilingual/alignment_matrices/hi.txt')
        number_of_epochs = 200
        labels2Id = labels2Idx
        log_file = '/home1/zishan/raghav/logs/tl_crowdflower_not_crosslingual_embedding_' +string+'_'+str(test_number)+'.txt' 
        print("log file: %s" %(log_file))
        weights_file='/home1/zishan/raghav/weights/tl_crowdflower_not_crosslingual_embedding_'+string+'_'+str(test_number)+'.h5'
        print("save weights file: %s" %(weights_file))
        batch_size=16
        train_file='/home1/zishan/raghav/Data/train_total.txt'
        f1_measure='macro'
        pos_label=1
        strategy = unfreeze_strategy[i]
        print(strategy)
        load_model_weights=True
        load_weights_file = '/home1/zishan/raghav/weights/pretrain_crowdflower_bilstm_3cnn.h5'
        nb_sequence_length = nb_sequence_length
        nb_embedding_dims= nb_embedding_dims
        check_for_generator=3

        test_model_tl_unfreezing(generator=generator, 
               train_sentences=train_sentences, 
               devLabels=devLabels, 
               number_of_tests= number_of_tests,
               number_of_epochs=number_of_epochs, 
               filename_to_log=log_file, 
               labels2Idx = labels2Id,
               filename_to_save_weigths=weights_file,
               batch_size=batch_size,
               unfreezing_strategy = strategy,       
               train_file=train_file, 
               f1_measure=f1_measure, 
               pos_label=pos_label, 
               load_model_weights=True,
               model_weights_file = load_weights_file, 
               nb_sequence_length=nb_sequence_length, 
               nb_embedding_dims=nb_embedding_dims, 
               transmat = transmat,
               check_for_generator= check_for_generator)


# In[ ]:


# print(len(dev_labels))


# In[ ]:


# print(Counter(dev_labels))

