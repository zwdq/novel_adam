from __future__ import print_function
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Activation, LSTM
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt

import numpy as np
import random
import sys
import os


class LanguageModel:
    def __init__(self, step=3, embed_size=128, seq_length=20):
        """
        :param step:  y is the (step's) word after the x seqence 步长？
        :param embed_size: the ebmedding size of all words  不懂
        :param seq_length: the length of sequence  
        """

        self.seq_length = seq_length
        self.step = step
        self.embed_size = embed_size

    def load_data(self, path):
        # read the entire text
        text = open(path).read().strip().replace('\u3000', '').replace('\n', '') #把换行符、空格去掉
        print('corpus length:', len(text))

        # all the vocabularies
        vocab = sorted(list(set(text)))  #把所有字去重置入一个集合
        print('total words:', len(vocab))

        # create word-index dict
        word_to_index = dict((c, i) for i, c in enumerate(vocab)) #字典
        index_to_word = dict((i, c) for i, c in enumerate(vocab)) #字典

        # cut the text into fixed size sequences
        sentences = []
        next_words = []

        for i in range(0, len(text) - self.seq_length, self.step):
            sentences.append(list(text[i:i + self.seq_length])) #lstm模型,自己写过
            next_words.append(text[i + self.seq_length]) #lstm模型,自己写过
        print('nb sequences:', len(sentences))

        # generate training samples  生成训练样本
        X = np.asarray([[word_to_index[w] for w in sent[:]] for sent in sentences]) #一列序号
        y = np.zeros((len(sentences), len(vocab))) #类似onehot编码，y是横着的1个1一堆0
        for i, word in enumerate(next_words): #
            y[i, word_to_index[word]] = 1

        self.text = text
        self.vocab = vocab
        self.word_to_index = word_to_index
        self.index_to_word = index_to_word
        self.X = X
        self.y = y

    def load_model(self):
        # load a Sequential model
        model = Sequential()
        model.add(Embedding(len(self.vocab), self.embed_size, input_length=self.seq_length))
        model.add(LSTM(self.embed_size, input_shape=(self.seq_length, self.embed_size), return_sequences=False))
        model.add(Dense(len(self.vocab)))
        model.add(Activation('softmax'))

        self.model = model

    def visualize_model(self):
        print(self.model.input_shape)
        print(self.model.output_shape)

    def compile_model(self, lr = 0.01):
        # compile the model
        optimizer = RMSprop(lr = lr)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    def fit_model(self, model_path, batch_size=128, nb_epoch=1):
        if os.path.exists(model_path):
            print("---------Loading Model---------")
            self.model.load_weights(model_path)
        # fit the model with trainind data
        log = keras.callbacks.TensorBoard()
        earlystop = keras.callbacks.EarlyStopping(monitor= "accuracy",min_delta = 0.1,patience = 1)
        modelckpt = keras.callbacks.ModelCheckpoint(model_path,save_best_only = (True),save_weights_only = (True))
        self.history = self.model.fit(self.X, self.y, callbacks = [log,earlystop,modelckpt],batch_size=batch_size, epochs=nb_epoch).history

    def save(self,path):
        #print(self.history['acc'])
        #print(self.history['val_acc'])
        #plt.plot(history['acc'])
        #plt.plot(history['val_acc'])
        #plt.title('model accuracy')
        #plt.ylabel('accuracy')
        #plt.xlabel('epoch')
        #plt.legend(['train', 'test'], loc='upper left')
        self.model.save(path)

    def predict(self, x, verbose=0):
        return self.model.predict([x], verbose=verbose)[0]

    def _sample(self, preds, diversity=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / diversity
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def generate_text(self):
        # generate text from random text seed
        start_index = random.randint(0, len(self.text) - self.seq_length - 1)

        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print()
            print('--------diversity:', diversity)

            generated = ''
            sentence = self.text[start_index:start_index + self.seq_length]
            generated += sentence
            print('--------Generating with seed:', sentence)
            sys.stdout.write(generated)

            for i in range(400):
                x = np.asarray([self.word_to_index[w] for w in sentence]).reshape([1, self.seq_length])
                preds = self.predict(x)
                next_index = self._sample(preds, diversity)
                next_word = self.index_to_word[next_index]

                generated += next_word
                sentence = sentence[1:] + next_word

                sys.stdout.write(next_word)
                sys.stdout.flush()
            print()


if __name__ == '__main__':
    model = LanguageModel(seq_length = 20)
    model.load_data('novels/天龙八部.txt')
    model.load_model()
    model.visualize_model()
    model.compile_model(lr = 0.00005)
    model.fit_model(nb_epoch = 50,model_path = "./model/keras_lstm_1000.h5")
    model.save("./model/keras_lstm_1000.h5")

    for i in range(1):
        print('Iteration:', i)
        model.generate_text()
        print()
