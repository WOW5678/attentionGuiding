# -*- coding:utf-8 -*-
"""
@Time: 2021/04/21 16:15
@Author:
@Version: Python 3.7
@Function: organize different dataset
"""
import torch
from torch.utils.data import Dataset, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
import nltk
nltk.data.path.append('../nltk_data')
from tqdm import tqdm
import os
import pickle
import spacy

nlp = spacy.load('en_core_web_sm')
sci_nlp = spacy.load("en_core_sci_sm")

class IRDataset(Dataset):
    def __init__(self, df, tokenizer, args, flag):
        self.df = df
        self.tokenizer = tokenizer
        self.args = args
        self.flag = flag
        basedir = os.path.join(args.project_dir, 'data/IR/%s' % flag)
        if not os.path.exists(basedir):
            os.makedirs(basedir)

        if args.model_name in ['bert-base-uncased', 'ttumyche/bluebert', 'nfliu/scibert_basevocab_uncased']:
            dataset_filename = os.path.join(basedir, 'dataset.pkl')

        elif args.model_name in ['emilyalsentzer/Bio_ClinicalBERT', 'dmis-lab/biobert-v1.1']:
            dataset_filename = os.path.join(basedir, 'bio_dataset.pkl')

        elif args.model_name == 'albert-base-v2':
            dataset_filename = os.path.join(basedir, 'albert_dataset.pkl')

        elif args.model_name == 'roberta-base':
            dataset_filename = os.path.join(basedir, 'roberta_dataset.pkl')


        self.dataset = self.prepare4BERT(dataset_filename)

    def prepare4BERT(self, dataset_filename):
        if os.path.exists(dataset_filename):
            with open(dataset_filename, 'rb') as f:
                dataset = pickle.load(f)
        else:
            pair_token_ids_list = []
            pair_mask_ids_list = []
            pair_seg_ids_list = []
            y = []

            premise_list = self.df['Premise'].apply(lambda x: x.lower()).to_list()
            hypothesis_list = self.df['Hypothesis'].apply(lambda x: x.lower()).to_list()
            label_list = self.df['Entails'].to_list()

            for (premise, hypothesis, label) in tqdm(zip(premise_list, hypothesis_list, label_list)):
                token_hypothesis = self.tokenizer.tokenize(hypothesis)
                token_premise = self.tokenizer.tokenize(premise)
                token_hypothesis, token_premise = self.truncate_seq_pair(token_hypothesis, token_premise)
                # merge sentences
                tokens = [self.tokenizer.cls_token] + token_hypothesis + token_premise
                padding_len = self.args.max_len - len(tokens)
                # convert token to indices
                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                # pad sequence
                input_ids = input_ids + [0] * padding_len
                # create input masks
                attention_masks = [min(1, x) for x in input_ids]
                token_type_ids = [0] + [0] * len(token_hypothesis) + [1] * len(token_premise)
                # pad sequence
                token_type_ids = token_type_ids + [0] * padding_len

                #return input_ids, attention_masks, token_type_ids
                pair_token_ids_list.append(input_ids)
                pair_mask_ids_list.append(attention_masks)
                pair_seg_ids_list.append(token_type_ids)
                y.append(label)

            pair_token_ids = np.array(pair_token_ids_list)
            pair_mask_ids = np.array(pair_mask_ids_list)
            pair_seg_ids = np.array(pair_seg_ids_list)
            y = np.array(y)

            y = y.reshape(y.shape[0], -1)
            dataset = np.concatenate((pair_token_ids, pair_mask_ids, pair_seg_ids, y), -1)  # [200, 256+256+256]

            # 将kgdataset和dataset进行保存
            with open(dataset_filename, 'wb') as f:
                pickle.dump(dataset, f, protocol=4)

        return dataset

    def truncate_seq_pair(self, tokens_1, tokens_2):

        while True:
            total_len = len(tokens_1) + len(tokens_2)
            # 不需要截断
            if total_len <= self.args.max_len-3: # 为3个特殊符号留出位置
                break
            if len(tokens_1) > len(tokens_2):
                # 截断token_1
                tokens_1.pop()
            else:
                tokens_2.pop()
        tokens_1.append(self.tokenizer.sep_token)
        tokens_2.append(self.tokenizer.sep_token)
        return tokens_1, tokens_2

    def __getitem__(self, index):

        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)




