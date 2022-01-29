#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2021/9/9 11:05
# @Author :
# @File : modelTest.py
# @Function:

import warnings
warnings.filterwarnings("ignore")

import argparse
import random
import numpy as np

import torch
from torch import cuda
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
import os

import sys
project_dir = '/data/0shared/*/attentionRegularization'
sys.path.append(project_dir)
import utils
import mednli_dataset
import alignment
from common import mednli_train_with_test
from models import mednli_model
import pandas as pd

#判断设备
device ='cuda:9' if cuda.is_available() else 'cpu'
#参数定义
PARSER = argparse.ArgumentParser(description='The code for attention visualization')
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
PARSER.add_argument('-model_name', '--model_name', default='dmis-lab/biobert-v1.1', type=str)
PARSER.add_argument('-project_dir', '--project_dir', default=project_dir, type=str)
PARSER.add_argument('-test_file', '--test_file', default=os.path.join(project_dir, 'data/medNLI/mli_test_v1_kg.csv'), type=str)
PARSER.add_argument('-epochs', '--epochs', default=2, type=int)
PARSER.add_argument('-batch_size', '--batch_size', default=1, type=int)
PARSER.add_argument('-learning_rate', '--learning_rate', default=1e-05, type=float)
PARSER.add_argument('-class_nums', '--class_nums', default=1, type=int)  # 0-1 两个类别 是个二分类问题
PARSER.add_argument('-save_path', '--save_path', default=os.path.join(project_dir, 'save_model/medNLI'), type=str)
PARSER.add_argument('-loss_type', '--loss_type', type=str, default='task', choices=('task+pdg', 'task+adg','bce+both'))

# 绘图时使用
PARSER.add_argument('-attention_pca', '--attention_pca', type=bool, default=True)

args = PARSER.parse_args()


if __name__ == '__main__':
    # 固定随机种子（据说只有放在main里才能起作用）
    seed = 2021
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    args.label2id = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    # step-1:加载预训练模型
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, truncation=True, do_lower_case=True, reteurn_dict=True, local_files_only=False)
    bert_model = AutoModel.from_pretrained(args.model_name, output_attentions=True, output_hidden_states=True) #, output_attentions=True
    bert_total_parameters = sum(p.numel() for p in bert_model.parameters() if p.requires_grad)
    bert_trainable_parameters = sum(p.numel() for p in bert_model.parameters())
    print('params number of bert:', bert_total_parameters)
    print('trainable params number of bert:', bert_trainable_parameters)

    # 加载已经训练好的保存下来的模型
    # step-2:构建新的模型

    test_medNlIModel = mednli_model.MedNLIClassifier(tokenizer, bert_model, args)

    # step-3:加载保存下来的参数
    if '/' in args.model_name:
        model_name = args.model_name.replace('/', '-')
    else:
        model_name = args.model_name
    print('current test model:', 'model_%s_%s.pt' % (model_name, args.loss_type))
    test_medNlIModel.load_state_dict(torch.load(os.path.join(args.save_path, 'model_%s_%s.pt' % (model_name, args.loss_type)), map_location=device))

    test_medNlIModel = test_medNlIModel.to(args.device)

    # step-4: 准备好测试数据集
    test_df = pd.read_csv(args.test_file)
    medNLI_test_dataset = mednli_dataset.MedNLIDataset(test_df, tokenizer, args, flag='test')
    test_dataloader = DataLoader(medNLI_test_dataset, shuffle=False, batch_size=args.batch_size)

    # step-5: 模型推理
    test_score_list = mednli_train_with_test.test(test_medNlIModel, test_dataloader, args)

