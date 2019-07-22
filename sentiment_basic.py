from data_preprocessor import *
import numpy as numpy
import pandas as pd
import tensorflow as tf


# class SentimentNetwork(tf.keras.Model):
#     def __init__(self, word_cnt, hidden_nodes, output_nodes):
#         super(SentimentNetwork, self).__init__()


def makedic(whole_comments):
    words = [x.split(" ") for x in whole_comments]
    flat = []
    for cm in words:
        flat += cm
    unique, counts = np.unique(np.array(flat), return_counts = True)
    word2index = {word:index for (index, word) in enumerate(unique) }
    wordcntdict = dict(zip(unique, counts))
    return wordcntdict , word2index

class SentimentNetwork():
    def __init__(self, word_cnt, word2index, hidden_nodes = 10, output_nodes = 3, learning_rate = 1e-4):
        self.wordW = tf.Variable(tf.zeros((word_cnt, hidden_nodes)), name = 'wordW')
        self.fc_w = tf.Variable(tf.keras.backend.random_normal((hidden_nodes, output_nodes)) * np.sqrt(1 / output_nodes), name='fc_w')
        self.params = [self.wordW, self.fc_w]
        self.layer1 = tf.zeros((1, hidden_nodes))
        self.word2index = word2index
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.lossfunc = tf.keras.backend.softmax
        self.acc = tf.keras.losses.SparseCategoricalCrossentropy()
        train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')

    def forward(self, thisx):
        #layer 1 : Add weights of each words in comment
        for word in thisx.split(" "):
            self.layer1 = tf.math.add(self.layer1, self.wordW[self.word2index[word]])
        #layer 2 : Fully connected net
        scores = tf.matmul(self.layer1, self.fc_w)
        return scores

    def _train_singlestep(self, x,y, train = True):
        with tf.GradientTape() as tape:
            scores = self.forward(x)
            #Caution : as labels are [-1, 0, 1], add 1 to make it [0, 1, 2] lable
            loss = self.lossfunc(scores)
            loss = self.acc(y, scores)
            total_loss = tf.reduce_mean(loss)
        grad_params = tape.gradient(total_loss, self.params)
        self.optimizer.apply_gradients(zip(grad_params, self.params))
        
    
    def train(self, trainset, print_every=20, num_epoches = 3):
        fw_train = tf.summary.FileWriter("./TrainHistory", )
        for epoch in range(num_epoches):
            for t, (xt, yt) in enumerate(trainset):
                self._train_singlestep(xt, yt)
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', self.lossfunc(), step = epoch)
                tf.summary.scalar('accuracy', self.acc(), step = epoch)
                #place to print & Save values
                if t % print_every ==0:
                    template = 'epoch {}, Iteration {} / {}'
                    print (template.format(epoch, t, len(trainset)))
            self.lossfunc.reset_states()
            self.acc.reset_states()
    
    def test(self, testset):
        predicted = []
        whole, right = 0, 0
        for (x, y) in testset:
            scores = self.forward(x)
            prediciton = numpy.array(scores).argmax(axis = 1)
            if prediciton[0] == y:
                right+=1
            whole+=1
        return (right, whole, right/whole)




totalset, testset, valset, trainset = sentiment_preprocessor("./tweet_processed.txt", "./sentiment2.xlsx")
print(len(totalset), len(trainset), len(valset), len(testset))
whole_wordcnt_dict, word2index = makedic(totalset)
# print(trainset)

model = SentimentNetwork(len(whole_wordcnt_dict), word2index, learning_rate=1e-6)
for i in range(3):
    model.train(trainset, print_every = 500,  num_epoches=1)
    print(model.test(valset))
merged = tf.summary(merge_all)

