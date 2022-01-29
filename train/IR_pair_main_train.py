#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2021/9/18 11:10
# @Author :
# @File : IR_pair_main_train.py
# @Function:
import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import random
import numpy as np

import torch
from torch import cuda
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader

import sys
project_dir = '/data/0shared/lidongdong/WSS/SIGIR2022-20220113/attentionRegularization'
sys.path.append(project_dir)  # 将当前目录的上一级目录加载进来，即attentionRegularization
from models import IR_pair_model
from common import IR_pair_train_with_test
import ir_dataset
import utils

#判断设备
device ='cuda:1' if cuda.is_available() else 'cpu'
#参数定义
PARSER = argparse.ArgumentParser(description='The code for attention visualization')
PARSER.add_argument('-device', '--device', default=device)
PARSER.add_argument('-max_len', '--max_len', default=256, type=int)
#bert-base-uncased
#ttumyche/bluebert
#dmis-lab/biobert-v1.1
#nfliu/scibert_basevocab_uncased
#albert-base-v2
#emilyalsentzer/Bio_ClinicalBERT
#roberta-base
PARSER.add_argument('-model_name', '--model_name', default='roberta-base', type=str)
PARSER.add_argument('-project_dir', '--project_dir', default=project_dir, type=str)
PARSER.add_argument('-train_file', '--train_file', default=os.path.join(project_dir, 'data/IR/train_bmp_process_augmented_10n_4p_processed.csv'), type=str)
PARSER.add_argument('-val_file', '--val_file', default=os.path.join(project_dir, 'data/IR/dev_bmp_process_processed.csv'), type=str)
PARSER.add_argument('-test_file', '--test_file', default=os.path.join(project_dir, 'data/IR/test_bmp_process_processed.csv'), type=str)
PARSER.add_argument('-epochs', '--epochs', default=5, type=int)
PARSER.add_argument('-batch_size', '--batch_size', default=64, type=int)
PARSER.add_argument('-learning_rate', '--learning_rate', default=1e-05, type=float)
PARSER.add_argument('-loss_type', '--loss_type', type=str, default='task+both', choices=('task+pdg', 'task+adg','bce+both'))

PARSER.add_argument('-eval_epochs', '--eval_epochs', default=1, type=int)
PARSER.add_argument('-class_nums', '--class_nums', default=1, type=int)  # 0-1 两个类别 是个二分类问题
PARSER.add_argument('-save_path', '--save_path', default=os.path.join(project_dir, 'save_model/IRPair'), type=str)

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

    # step-1:加载预训练模型
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, truncation=True, do_lower_case=True, reteurn_dict=True, local_files_only=False,  TOKENIZERS_PARALLELISM=True)
    bert_model = AutoModel.from_pretrained(args.model_name, output_attentions=True, output_hidden_states=True, local_files_only=False)  #, output_attentions=True
    bert_total_parameters = sum(p.numel() for p in bert_model.parameters() if p.requires_grad)
    bert_trainable_parameters = sum(p.numel() for p in bert_model.parameters())
    print('params number of bert:', bert_total_parameters)
    print('trainable params number of bert:', bert_trainable_parameters)

    # step-2:创建dataset和dataloader
    train_df = utils.processor(args.train_file)
    #train_df = train_df.sample(200)

    print(train_df.head(5))

    medicalIR_train_dataset = ir_dataset.IRDataset(train_df, tokenizer, args, flag='train')

    train_dataloader = DataLoader(
        medicalIR_train_dataset,
        shuffle=True,
        batch_size=args.batch_size
    )

    # step-3:创建分类模型
    IRModel = IR_pair_model.BertPairClassifier(tokenizer, bert_model, args)
    IRModel = IRModel.to(args.device)
    pytorch_total_params = sum(p.numel() for p in IRModel.parameters() if p.requires_grad)
    pytorch_trainable_params = sum(p.numel() for p in IRModel.parameters())
    print("Total number of classifier params", pytorch_total_params)
    print("Total number of trainable classifier params", pytorch_trainable_params)

    # step-4：创建优化器
    optimizer = torch.optim.Adam(params=IRModel.parameters(), lr=args.learning_rate)

    # step-4：训练模型
    IR_pair_train_with_test.train(IRModel, train_dataloader, optimizer, args)

