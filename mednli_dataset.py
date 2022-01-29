# -*- coding:utf-8 -*-
"""
@Time: 2021/04/21 16:15
@Author:
@Version: Python 3.7
@Function: organize different dataset
"""

from torch.utils.data import Dataset
import numpy as np
import nltk
nltk.data.path.append('../nltk_data')
from tqdm import tqdm
import os
import pickle
import spacy
from utils import ColPMI, WordnetSim, DependencySim, tokens_recover
import random

nlp = spacy.load('en_core_web_sm')
sci_nlp = spacy.load("en_core_sci_sm")


class MedNLIDataset(Dataset):
    def __init__(self, df, tokenizer, args, flag):
        self.df = df
        self.tokenizer = tokenizer
        self.args = args
        self.flag =  flag


        basedir = os.path.join(args.project_dir, 'data/medNLI/%s' % self.flag)
        if not os.path.exists(basedir):
            os.makedirs(basedir)

        if args.model_name == 'bert-base-uncased':
            dataset_filename = os.path.join(basedir, 'dataset.pkl')
        elif args.model_name in ['ttumyche/bluebert','bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12']:
            dataset_filename = os.path.join(basedir, 'blue_dataset.pkl')
        elif args.model_name == 'nfliu/scibert_basevocab_uncased':
            dataset_filename = os.path.join(basedir, 'sci_dataset.pkl')
        elif args.model_name == 'emilyalsentzer/Bio_ClinicalBERT':
            dataset_filename = os.path.join(basedir, 'clinical_dataset.pkl')
        elif args.model_name == 'dmis-lab/biobert-v1.1':
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

            sentence1_list = self.df['sentence1'].apply(lambda x: x.lower()).tolist()
            sentence2_list = self.df['sentence2'].apply(lambda x: x.lower()).tolist()
            label_list = self.df['gold_label'].tolist()

            for (sent1, sent2, label) in zip(sentence1_list, sentence2_list, label_list):
                token_sent1 = self.tokenizer.tokenize(sent1)
                token_sent2 = self.tokenizer.tokenize(sent2)
                token_sent1, token_sent2 = self.truncate_seq_pair(token_sent1, token_sent2)

                tokens = [self.tokenizer.cls_token] + token_sent1 + [self.tokenizer.sep_token] + token_sent2 + [
                    self.tokenizer.sep_token]

                # convert token to indices
                token_type_ids = [0] + [0] * (len(token_sent1) + 1) + [1] * (len(token_sent2) + 1)
                # pad sequence
                token_type_ids = token_type_ids + [0] * (self.args.max_len - len(token_type_ids))
                # merge sentences

                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                # pad sequence
                input_ids = input_ids + [0] * (self.args.max_len - len(input_ids))
                # create input masks
                attention_masks = [min(1, x) for x in input_ids]
                # return input_ids, attention_masks, token_type_ids
                pair_token_ids_list.append(input_ids)
                pair_mask_ids_list.append(attention_masks)
                pair_seg_ids_list.append(token_type_ids)
                y.append(self.args.label2id.get(label))

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

        return tokens_1, tokens_2

    def __getitem__(self, index):

        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

