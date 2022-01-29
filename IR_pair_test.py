#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2021/9/17 15:19
# @Author :
# @File : IR_val.py
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
project_dir = '/data/0shared/lidongdong/WSS/SIGIR2022-20220113/attentionRegularization'
sys.path.append(project_dir)
import utils
import ir_dataset
import alignment
from common import IR_pair_train_with_test
from models import IR_pair_model
from evaluations import IR_evaluation
import pandas as pd

#判断设备
device ='cuda:7' if cuda.is_available() else 'cpu'
#参数定义
PARSER = argparse.ArgumentParser(description='The code for attention visualization')
PARSER.add_argument('-device', '--device', default=device)
PARSER.add_argument('-max_len', '--max_len', default=256, type=int)
#bert-base-uncased
#ttumyche/bluebert
#dmis-lab/biobert-v1.1s
#nfliu/scibert_basevocab_uncased
#albert-base-v2
#emilyalsentzer/Bio_ClinicalBERT
#roberta-base
PARSER.add_argument('-model_name', '--model_name', default='dmis-lab/biobert-v1.1', type=str)
PARSER.add_argument('-project_dir', '--project_dir', default=project_dir, type=str)
PARSER.add_argument('-val_file', '--val_file', default=os.path.join(project_dir, 'data/IR/test_bmp_process_processed.csv'), type=str)
PARSER.add_argument('-epochs', '--epochs', default=10, type=int)
PARSER.add_argument('-batch_size', '--batch_size', default=64, type=int)
PARSER.add_argument('-learning_rate', '--learning_rate', default=5e-05, type=float)
PARSER.add_argument('-class_nums', '--class_nums', default=1, type=int)  # 0-1 两个类别 是个二分类问题
PARSER.add_argument('-save_path', '--save_path', default=os.path.join(project_dir, 'save_model/IRPair'), type=str)
PARSER.add_argument('-loss_type', '--loss_type', type=str, default='task+pdg', choices=('task+pdg', 'task+adg', 'task+both'))
PARSER.add_argument('-best_model_id', '--best_model_id', type=int, default=0)


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
    bert_model = AutoModel.from_pretrained(args.model_name, output_attentions=True, output_hidden_states=True) #, output_attentions=True
    bert_total_parameters = sum(p.numel() for p in bert_model.parameters() if p.requires_grad)
    bert_trainable_parameters = sum(p.numel() for p in bert_model.parameters())
    print('params number of bert:', bert_total_parameters)
    print('trainable params number of bert:', bert_trainable_parameters)

    # 加载已经训练好的保存下来的模型
    # step-2:构建新的模型
    test_IRModel = IR_pair_model.BertPairClassifier(tokenizer, bert_model, args)

    best_mrr = 0
    best_result = [0] * 5

    # step-3:加载保存下来的参数


    if '/' in args.model_name:
        model_name = args.model_name.replace('/', '-')
    else:
        model_name = args.model_name

    test_IRModel.load_state_dict(torch.load(os.path.join(args.save_path, 'model_%s_%s_%d.pt' % (model_name, args.loss_type, args.best_model_id)), map_location=device))

    test_IRModel = test_IRModel.to(args.device)

    # step-4: 准备好测试数据集
    val_df = utils.processor(args.val_file)
    #val_df = val_df.sample(2000)
    medicalIR_val_dataset = ir_dataset.IRDataset(val_df, tokenizer, args, flag='test')
    val_dataloader = DataLoader(medicalIR_val_dataset, shuffle=False, batch_size=args.batch_size)

    # step-5: 模型推理
    test_score_list = IR_pair_train_with_test.test(test_IRModel, val_dataloader, args)
    # 对所有数据执行完预测之后
    results = pd.DataFrame(columns=['Hypothesis', 'PMID', 'score'])
    results['Hypothesis'] = val_df['Hypothesis']
    results['PMID'] = val_df['PMID']
    results['score'] = np.array(test_score_list)
    #results['label'] = val_df['Entails']

    # 计算评估指标
    recall_result, mrr = IR_evaluation.score_calculate('test', results)
    print(recall_result)
