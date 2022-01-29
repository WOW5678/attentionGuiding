#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2021/9/16 15:45
# @Author :
# @File : multiLabel_evaluation.py
# @Function: muLti-class评估

import numpy as np

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, accuracy_score

def get_f1(y_pred, y_true):

    # y_pred = (y_pred >= threshold)
    # # 再将布尔型矩阵转换成int型
    # y_pred = y_pred + 0

    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    microF1 = f1_score(y_true, y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)


    return accuracy, precision, recall, microF1




