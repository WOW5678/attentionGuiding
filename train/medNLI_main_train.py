#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2021/9/23 11:43
# @Author :
# @File : mednli_main_train_trainsize.py
# @Function: 对医疗句子关系进行分类


import warnings
warnings.filterwarnings("ignore")

import argparse
import random
import numpy as np
import os

import torch
from torch import cuda
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
import pandas as pd

import sys
project_dir = '/data/0shared/lidongdong/WSS/SIGIR2022-20220113/attentionRegularization'
sys.path.append(project_dir)
import mednli_dataset

from models import mednli_model
from common import mednli_train_with_test


#判断设备
device ='cuda:6' if cuda.is_available() else 'cpu'
#参数定义
PARSER = argparse.ArgumentParser(description='The code for multi-label classification with attention guiding')
PARSER.add_argument('-device', '--device', default=device)
PARSER.add_argument('-max_len', '--max_len', default=256, type=int)
# bert-base-uncased
# bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12(抛弃)
# ttumyche/bluebert
# dmis-lab/biobert-v1.1
# emilyalsentzer/Bio_ClinicalBERT
# nfliu/scibert_basevocab_uncased
# albert-base-v2
# roberta-base
PARSER.add_argument('-model_name', '--model_name', default='roberta-base', type=str, choices=('bert-base-uncased', 'albert-xxlarge-v2', 'emilyalsentzer/Bio_ClinicalBERT'))
PARSER.add_argument('-project_dir', '--project_dir', default=project_dir, type=str)
PARSER.add_argument('-train_file', '--train_file', default=os.path.join(project_dir, 'data/medNLI/mli_train_v1_kg.csv'), type=str)
PARSER.add_argument('-val_file', '--val_file', default=os.path.join(project_dir, 'data/medNLI/mli_dev_v1_kg.csv'), type=str)
PARSER.add_argument('-epochs', '--epochs', default=20, type=int)
PARSER.add_argument('-batch_size', '--batch_size', default=64, type=int)
PARSER.add_argument('-learning_rate', '--learning_rate', default=1e-05, type=float)


PARSER.add_argument('-eval_epochs', '--eval_epochs', default=1, type=int)
PARSER.add_argument('-save_path', '--save_path', default=os.path.join(project_dir, 'save_model/medNLI'), type=str)
PARSER.add_argument('-label2id', '--label2id', type=dict)  # 保存着挑选出的每个icd和对应的id
PARSER.add_argument('-loss_type', '--loss_type', type=str, default='task', choices=('task+pdg', 'task+adg', 'bce+both'))

PARSER.add_argument('-ad_factor', '--pd_factor', default=0.001, type=float)
PARSER.add_argument('-diff_factor', '--ad_factor', default=0.001, type=float)

args = PARSER.parse_args()

if __name__ == '__main__':
    print('model:', args.model_name)
    print('loss_type:', args.loss_type)
    print('batch_size:', args.batch_size)

    # 固定随机种子（据说只有放在main里才能起作用）
    seed = 2021
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


    # step-1:加载预训练模型
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, truncation=True, do_lower_case=True, reteurn_dict=True, local_files_only=False)
    bert_model = AutoModel.from_pretrained(args.model_name, output_attentions=True, output_hidden_states=True, local_files_only=False)  #, output_attentions=True


    # step-2:创建dataset和dataloader
    train_df = pd.read_csv(args.train_file)
    train_df = train_df.dropna()
    print('len of train_df:', len(train_df))
    # train_df = train_df.sample(min(200, len(train_df)))

    val_df = pd.read_csv(args.val_file)
    val_df = val_df.dropna()
    print('len of val_df:', len(val_df))
    # val_df = val_df.sample(min(200, len(val_df)))

    print(train_df.head(5))
    print(val_df.head(5))

    args.label2id = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    # 如果已经有处理好的数据 则直接加载

    mednli_train_dataset = mednli_dataset.MedNLIDataset(train_df, tokenizer, args, flag='train')
    mednli_val_dataset = mednli_dataset.MedNLIDataset(val_df, tokenizer, args, flag='val')


    train_dataloader = DataLoader(
        mednli_train_dataset,
        shuffle=True,
        batch_size=args.batch_size,

    )

    val_dataloader = DataLoader(
        mednli_val_dataset,
        shuffle=False,
        batch_size=args.batch_size,

    )

    # step-3:创建分类模型

    mednliClassifier = mednli_model.MedNLIClassifier(tokenizer, bert_model, args)
    mednliClassifier = mednliClassifier.to(args.device)
    pytorch_total_params = sum(p.numel() for p in mednliClassifier.parameters() if p.requires_grad)
    pytorch_trainable_params = sum(p.numel() for p in mednliClassifier.parameters())
    print("Total number of classifier params", pytorch_total_params)
    print("Total number of trainable classifier params", pytorch_trainable_params)

    # step-4：创建优化器
    optimizer = torch.optim.Adam(mednliClassifier.parameters(), lr=args.learning_rate)

    # step-4：训练模型
    mednli_train_with_test.train(mednliClassifier, train_dataloader, val_dataloader, optimizer, args)








