 #coding=utf-8
from __future__ import unicode_literals
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import codecs
import gensim
import jieba


with codecs.open('names.txt', encoding="utf8") as f:
    # 去掉结尾的换行符
    data = [line.strip() for line in f]

novels = data[::2]
names = data[1::2]

novel_names = {k: v.split() for k, v in zip(novels, names)}

def find_main_charecters(novel, num=10):
    with codecs.open('novels/{}.txt'.format(novel), encoding="utf8") as f:
        data = f.read()
    chars = novel_names[novel]
    count = map(lambda x: data.count(x), chars)
    print(count)
    idx = count.argsort()
    
    plt.barh(range(num), count[idx[-num:]], color='red', align='center')
    plt.title(novel, 
              fontsize=14)
    plt.yticks(range(num), chars[idx[-num:]], 
               fontsize=14)
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
find_main_charecters("天龙八部")
