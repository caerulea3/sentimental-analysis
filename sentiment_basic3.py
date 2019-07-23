from data_preprocessor import *
import numpy as numpy
import pandas as pd
import tensorflow as tf
from tqdm import tqdm as tqdm
import matplotlib.pyplot as plt
import pickle

device = '/cpu:0'  
ACTIVE = True

tf.config.threading.set_intra_op_parallelism_threads(16 if ACTIVE else 1)

def makedic(whole_comments):
    words = [x.split(" ") for x in whole_comments]
    flat = []
    for cm in words:
        flat += cm
    unique, counts = np.unique(np.array(flat), return_counts = True)
    word2index = {word:index for (index, word) in enumerate(unique) }
    wordcntdict = dict(zip(unique, counts))
    return wordcntdict , word2index

def convert(batch, vectorsize, word2index):
    result = np.zeros((vectorsize, len(batch)))
    for i, sentence in enumerate(batch):
        if isinstance(sentence, tuple):
            sentence = sentence[0]
        vector = np.zeros((vectorsize, ))
        words = sentence.split(" ")
        for w in words:
            vector[word2index[w]] += 1
        result[:, i] += vector.T
    return result.T

# class SentimentNetwork():

class SentimentNetwork(tf.keras.Model):
    def __init__(self, hidden_size, num_classes):
        super(SentimentNetwork, self).__init__()
        initializer = tf.initializers.VarianceScaling(scale = 2.0)
        self.addings = tf.keras.layers.Dense(hidden_size, kernel_initializer = tf.keras.initializers.zeros())
        self.fc1 = tf.keras.layers.Dense(2000, kernel_initializer = initializer,  activation = 'sigmoid')
        self.fc2 = tf.keras.layers.Dense(2000, kernel_initializer = initializer,  activation = 'sigmoid')
        self.fc3 = tf.keras.layers.Dense(num_classes, kernel_initializer = initializer,  activation = 'sigmoid')

    def call(self, x):
        x = self.addings(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


totalset, testset, valset, trainset = sentiment_preprocessor("./tweet_processed.txt", "./sentiment2.xlsx")
print(len(totalset), len(trainset), len(valset), len(testset))
whole_wordcnt_dict, word2index = makedic(totalset)
words_cnt = len(whole_wordcnt_dict)

test_X, val_X, train_X = convert(testset, words_cnt, word2index), convert(valset, words_cnt, word2index), convert(trainset, words_cnt, word2index)
test_Y, val_Y, train_Y = np.array(testset)[:, 1],np.array(valset)[:, 1],np.array(trainset)[:, 1]


lr_low, lr_high = 1e-9, (1e-9 * (10**0.25))
candidates = []
for i in range(25 if ACTIVE else 2):
    candidates.append((lr_low, lr_high))
    lr_low *= (10**0.25)
    lr_high *= (10**0.25)

for LR_LOW, LR_HIGH in candidates:
    lr_candidates = np.random.uniform(low = LR_LOW, high = LR_HIGH, size = (50 if ACTIVE else 2,))
    his = []
    VERBOSE = 1
    for learning_rate in lr_candidates:
        model = SentimentNetwork(10, 3)
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate),
                        loss = 'sparse_categorical_crossentropy',
                        metrics=[tf.keras.metrics.sparse_categorical_accuracy])
        model.fit(train_X, train_Y, batch_size = 64, epochs = 5 if ACTIVE else 1, validation_data = (val_X, val_Y), verbose = VERBOSE)
        res = model.evaluate(val_X, val_Y, verbose = VERBOSE)
        if VERBOSE == 1:
            VERBOSE = 0
        his.append((learning_rate, res[1]))
    his = sorted(his, key=lambda l:l[0], reverse=True)
    his = np.array(his)
    with open("./testresult.pkl", 'wb') as f:
        pickle.dump(his, f)

    plt.plot(his[:, 0], his[:, 1])
    plt.savefig("./experiment/experiment_result_{:2E}_{:2E}.png".format(LR_LOW, LR_HIGH))

