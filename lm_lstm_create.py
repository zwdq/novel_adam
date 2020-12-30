from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Activation, LSTM
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
from keras.models import load_model

import numpy as np
import random
import sys


class LanguageModel:
    def __init__(self, step=3, embed_size=128, seq_length=20):
        """
        :param step:  y is the (step's) word after the x seqence 步长？
        :param embed_size: the ebmedding size of all words  不懂
        :param seq_length: the length of sequence  不懂
        """

        self.seq_length = seq_length
        self.step = step
        self.embed_size = embed_size

    def load_data(self, path):
        # read the entire text
        text = open(path).read().strip().replace('\u3000', '').replace('\n', '') #把换行符、空格去掉
        print('corpus length:', len(text))

        # all the vocabularies
        vocab = sorted(list(set(text)))  #排序？
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
        X = np.asarray([[word_to_index[w] for w in sent[:]] for sent in sentences]) #生成一个鬼数组
        y = np.zeros((len(sentences), len(vocab))) #生成一个鬼空数据
        for i, word in enumerate(next_words): #一个不知道是什么的迭代器
            y[i, word_to_index[word]] = 1

        self.text = text
        self.vocab = vocab
        self.word_to_index = word_to_index
        self.index_to_word = index_to_word
        self.X = X
        self.y = y

    def load_model(self):
        # load a Sequential model
        model = load_model("./model/keras_lstm_1000.h5")

        self.model = model

    def _sample(self, preds, diversity=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / diversity
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)


    def predict(self, x, verbose=0):
        return self.model.predict([x], verbose=verbose)[0]


    def generate_text(self):
        # generate text from random text seed
        start_index = random.randint(0, len(self.text) - self.seq_length - 1)

        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print()
            print('--------diversity:', diversity)

            generated = ''
            sentence = self.text[start_index:start_index + self.seq_length]
            sentence = '嘉穗今天又被主管怼了'
            generated += sentence
            print('--------Generating with seed:', sentence)
            sys.stdout.write(generated)

            for i in range(400):
                x = np.asarray([self.word_to_index[r] for r in sentence]).reshape([1, self.seq_length])
                preds = self.predict(x)
                next_index = self._sample(preds, diversity)
                next_word = self.index_to_word[next_index]

                generated += next_word
                sentence = sentence[1:] + next_word

                sys.stdout.write(next_word)
                sys.stdout.flush()
            print()


if __name__ == '__main__':
    
    model = LanguageModel(seq_length=10)
    model.load_data('novels/诡秘之主.txt')
    model.load_model()

    for i in range(1, 3):
        print('Iteration:', i)
        model.generate_text()
        print()
