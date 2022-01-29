#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2021/9/7 15:16
# @Author :
# @File : evaluation.py
# @Function: 评估指标

import json
import os

project_dir = '/data/0shared/lidongdong/WSS/SIGIR2022-20220113/attentionRegularization'
def read_results(results):
    claim_dict = dict()
    for index, row in results.iterrows():
        claim = row['Hypothesis']
        claim_dict.setdefault(claim, {'PMID': [], 'score': []}) #
        claim_dict[claim]['PMID'].append(str(row['PMID']))  # 注意：这里需要转换成str, 因为rank_result中PMID都是str类型
        claim_dict[claim]['score'].append(float(row['score']))
        #claim_dict[claim]['groundTruth'].append(row['label'])
    return claim_dict


def format_score(score):
    score_str = '{0:.4f}'.format(score)
    return '{:<11}'.format(score_str)


def score_calculate(mode, result_dataframe):
    if mode == 'val':
        # pc_oc_dict = json.load(open('pc_oc.json','r',encoding='utf-8'))
        evl_dict = json.load(open(os.path.join(project_dir, 'data/IR/dev_data_claim_pmid_dict_processed.json'), 'r', encoding='utf-8'))
    elif mode == 'test':
        evl_dict = json.load(open(os.path.join(project_dir, 'data/IR/test_data_claim_pmid_dict_processed.json'), 'r', encoding='utf-8'))
    elif mode == 'train':
        evl_dict = json.load(open(os.path.join(project_dir, 'data/IR/train_data_claim_pmid_dict_processed.json'), 'r', encoding='utf-8'))

    result_dict = read_results(result_dataframe)
    recall_result = [0 for i in range(4)]  # 共8个指标，R@1, R@3, R@5, R@10, R@20, R@50, R@100, R@200
    mrr_result = []
    successNum = 0
    max_positive_num = 0
    # result_dict:{claim:{'score':[],'PMID':[]}}
    for claim in result_dict:
        # print(claim)
        rank_results = [x for _, x in
                        sorted(zip(result_dict[claim]['score'], result_dict[claim]['PMID']), key=lambda pair: pair[0],
                               reverse=True)]
        # rank_results是排好序的PMID
        #如果没有找见该查询对应的ground truth
        if claim not in evl_dict:
            continue
        else:
            checkFlag = False
            pmid_set = [str(item) for item in evl_dict[claim]]
            pmid_set = set(pmid_set)# pmid_set中保存着该query对应的结果ID
            for pmid in pmid_set:
                if str(pmid) in rank_results:
                    checkFlag = True
            if checkFlag == False:
                continue
        # 确保排序的groud true在排序列表中（否则永远为0）
        successNum += 1
        max_positive_num = max(len(pmid_set), max_positive_num)
        mrr_check = False

        # 计算MRR的指标

        for index in range(len(rank_results)):
            if rank_results[index] in pmid_set:
                mrr_result.append(1.0/(index+1))
                mrr_check = True
                break

        if mrr_check == False:
            mrr_result.append(0.0)


        # 计算Recall值
        for i, index in zip([1, 3, 5, 20], range(8)):
            count_pmid = 0
            for pmid in pmid_set:
                if pmid in rank_results[:i]:
                    count_pmid += 1
            recall_result[index] += 1.0 * count_pmid / len(pmid_set)  # 分母为真正相关的个数（与K取值无关）
    print('max_positive_num:', max_positive_num)
    print('successNum:', successNum)
    if successNum ==0:
        result_score = [0 for s in recall_result]
        mrr = 0
    else:
        result_score = [s / successNum for s in recall_result]
        mrr = sum(mrr_result) / successNum

    MAIN_THRESHOLDS = [1, 3, 5, 20]
    metric_names = ['R@' + str(threshold) for threshold in MAIN_THRESHOLDS]
    metric_names.insert(0, 'MRR')
    metrics_header_items = ['{:<11}'.format(metric) for metric in metric_names]
    print('{:<26} '.format('') + ' '.join(metrics_header_items))

    formatted_scores = [format_score(score) for score in result_score]
    formatted_scores.insert(0, format_score(mrr))
    print('  {:<25}'.format('Score') + ' '.join(formatted_scores))

    # formatted_scores主要是为了方便展示
    return result_score, mrr
