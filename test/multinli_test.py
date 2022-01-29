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

import torch
from torch import cuda
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
import pandas as pd


import sys
project_dir = '/data/0shared/*/attentionRegularization'
sys.path.append(project_dir)
import multinli_dataset

from models import multiNLI_model
from common import multinli_train_with_test
import os

#判断设备
device ='cuda:2' if cuda.is_available() else 'cpu'
#参数定义
PARSER = argparse.ArgumentParser(description='The code for multiNLI classification with attention guiding')
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
PARSER.add_argument('-model_name', '--model_name', default='bert-base-uncased', type=str)
PARSER.add_argument('-project_dir', '--project_dir', default=project_dir, type=str)
PARSER.add_argument('-test_file', '--test_file', default=os.path.join(project_dir, 'data/multiNLI/multinli_0.9_dev_mismatched.csv'), type=str)
PARSER.add_argument('-epochs', '--epochs', default=5, type=int)
PARSER.add_argument('-batch_size', '--batch_size', default=64, type=int)
PARSER.add_argument('-learning_rate', '--learning_rate', default=1e-05, type=float)


PARSER.add_argument('-eval_epochs', '--eval_epochs', default=1, type=int)
PARSER.add_argument('-save_path', '--save_path', default=os.path.join(project_dir, 'save_model/multiNLI'), type=str)
PARSER.add_argument('-label2id', '--label2id', type=dict)  # 保存着挑选出的每个icd和对应的id
PARSER.add_argument('-loss_type', '--loss_type', type=str, default='task', choices=('task+pdg', 'task+adg', 'bce+both'))

args = PARSER.parse_args()

if __name__ == '__main__':
    # 固定随机种子（据说只有放在main里才能起作用）
    seed = 2021
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # step-1:加载预训练模型
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, truncation=True, do_lower_case=True, reteurn_dict=True, local_files_only=True)
    bert_model = AutoModel.from_pretrained(args.model_name, output_attentions=True, output_hidden_states=True, local_files_only=True)  #, output_attentions=True


    # step-2:创建dataset和dataloader
    test_df = pd.read_csv(args.test_file)
    test_df = test_df.dropna()
    print('len of test_df:', len(test_df))


    args.label2id = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    # 如果已经有处理好的数据 则直接加载

    multinli_test_dataset = multinli_dataset.MultiNLIDataset(test_df, tokenizer, args, flag='test')


    test_dataloader = DataLoader(
        multinli_test_dataset,
        shuffle=False,
        batch_size=args.batch_size,

    )

    # step-3:创建分类模型

    test_multinliClassifier = multiNLI_model.MultiNLIClassifier(tokenizer, bert_model, args)

    # step-3:加载保存下来的参数
    if '/' in args.model_name:
        model_name = args.model_name.replace('/', '-')
    else:
        model_name = args.model_name
    print('current test model:', 'model_%s_%s.pt' % (model_name, args.loss_type))
    test_multinliClassifier.load_state_dict(
        torch.load(os.path.join(args.save_path, 'model_%s_%s.pt' % (model_name, args.loss_type)), map_location=device))

    test_multinliClassifier.to(args.device)

    # step-4：训练模型
    multinli_train_with_test.test(test_multinliClassifier, test_dataloader, args)








