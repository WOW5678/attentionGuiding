# -*- coding:utf-8 -*-
"""
@Time: 2021/04/22 9:42
@Author:
@Version: Python 3.7
@Function:
"""
import pandas as pd


from nltk import bigrams
from itertools import combinations
import math
import numpy as np
import nltk
nltk.data.path.append('../nltk_data')
from nltk.corpus import wordnet
import spacy

nlp = spacy.load('en_core_web_sm')
sci_nlp = spacy.load("en_core_sci_sm")


def processor(file):
    #分别打开两个数据集
    df = pd.read_csv(file, encoding='latin-1')
    print('len(df):', len(df))
    #df = df.sample(min(datasize, len(df)))
    # 删除多于的列 这两列并不对于 很有用的
    # df = df.drop(['PMID'], axis=1)

    #删除为空的元素
    df = df.dropna()

    #确保sent1 sent2两列都是str类型
    df['Premise'] = df['Premise'].astype(str)
    df['Hypothesis'] = df['Hypothesis'].astype(str)

    # 确保句子对中每个句子都不为空
    df = df[(df['Premise'].str.split().str.len() > 0) & (df['Hypothesis'].str.split().str.len() > 0)]

    return df



class ColPMI:
    def __init__(self, tokens):
        #self.model = spacy.load('en_core_web_sm')
        self.tokens = tokens
        self.n_tokens = len(self.tokens)
        self.bgs = self.__generate_bigrams(tokens)

    def __generate_bigrams(self, tokens):
        bgs = bigrams(tokens)
        return [str(bg) for bg in bgs]

    def __count_word_frequency(self, word):
        freq = 0
        for tok in self.tokens:
            if tok == word:
                freq = freq + 1
        return freq

    def __count_bigram_frequency(self, k):
        freq = 0
        for bg in self.bgs:
            if bg == k:
                freq = freq + 1
        return freq

    def __probability(self, x, n):
        return x / n

    def __pmi(self, P_x, P_y, P_xy):
        try:
            return math.log2(P_xy / (P_x * P_y))
        except:
            return 0

    def PMI(self, x, y):
        bg = f"('{x}', '{y}')"

        # frequency
        n_x = self.__count_word_frequency(word=x)
        n_y = self.__count_word_frequency(word=y)
        n_bg = self.__count_bigram_frequency(k=bg)

        # probability
        p_x = self.__probability(n_x, self.n_tokens)
        p_y = self.__probability(n_y, self.n_tokens)
        p_bg = self.__probability(n_bg, len(self.bgs))

        # pmi
        pmi = self.__pmi(p_x, p_y, p_bg)

        return pmi

    def pmi_matrix(self):
        pmi_mat = np.zeros((len(self.tokens), len(self.tokens)))
        for (index1, index2) in combinations(range(len(self.tokens)), 2):
            score = self.PMI(self.tokens[index1], self.tokens[index2])
            pmi_mat[index1][index2] = score
            pmi_mat[index2][index1] = score
        #pmi_mat += np.eye(len(self.tokens), len(self.tokens))
        # 对pmi_mat进行按行求和值
        #print('pmi:', pmi_mat)
        pmi_mat = np.mean(pmi_mat, axis=0)
        return pmi_mat

def tokens_recover(tokens):
    # 先针对每个token生成BI指示器
    #flag = ['I'] * len(tokens)
    words = [''] * len(tokens)
    for i in range(len(tokens)):
        if tokens[i][:2] != '##':
            #flag[i] = 'B'  # 找见了start
            start = i
            # 查找end

            for j in range(i+1, len(tokens)):
                if tokens[j][:2] != '##':
                    end = j
                    break

            if i == len(tokens)-1:
                word = ''.join(tokens[start:]).replace('#', '')
            else:
                word = ''.join(tokens[start:end]).replace('#', '')
            words[i] = word

    # 对于空格的部分都填充后面第一个不为0的word
    for i in range(len(words)):
        if words[i] == '':
            words[i] = words[i-1]
    # 返回复原后的string
    return words


class WordnetSim:
    def __init__(self, tokens):
        self.tokens = tokens

    def sim(self, x, y):
        #print('wordnet.synsets(x):',wordnet.synsets(x))
        syn1 = wordnet.synsets(x)
        syn2 = wordnet.synsets(y)
        if len(syn1) == 0 or len(syn2) == 0:
            return 0
        else:
            return syn1[0].wup_similarity(syn2[0])

    def sim_matrix(self):
        sim_mat = np.zeros((len(self.tokens), len(self.tokens)))

        for (index1, index2) in combinations(range(len(self.tokens)), 2):
            score = self.sim(self.tokens[index1], self.tokens[index2])
            sim_mat[index1][index2] = score
            sim_mat[index2][index1] = score
        # 对角线上的pmi
        sim_mat += np.eye(len(self.tokens), len(self.tokens))
        sim_mat = np.mean(sim_mat, axis=0)
        return sim_mat

class DependencySim:
    def __init__(self, tokens, max_len=None):
        self.tokens = tokens
        self.max_len = max_len

    def dependency_sim(self):

        doc = nlp(' '.join(self.tokens))

        dependency_list = []
        for token in doc:
            #print('{0}({1}) <-- {2} -- {3}({4})'.format(token.text, token.tag_, token.dep_, token.head.text, token.head.tag_))
            dependency_list.append((token.text, token.head.text))
        sim_mat = np.zeros((len(self.tokens), len(self.tokens)))
        for (index1, index2) in combinations(range(len(self.tokens)), 2):
            if (self.tokens[index1], self.tokens[index2]) in dependency_list:
                sim_mat[index1][index2] += 1
                sim_mat[index2][index1] += 1 #依赖次数
            elif (self.tokens[index2], self.tokens[index1]) in dependency_list:
                sim_mat[index2][index1] += 1
                sim_mat[index1][index2] += 1  # 依赖次数

        sim_mat = np.mean(sim_mat, axis=0)
        return sim_mat
