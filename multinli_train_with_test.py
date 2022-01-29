#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2021/9/16 14:52
# @Author :
# @File : multiLabel_train_with_test.py
# @Function: 训练并测试多标签分类器

import torch
import time
import torch.nn.functional as F
from evaluations import mednli_evaluation
import numpy as np
from tqdm import tqdm
import os
import alignment

def train(classifier, train_loader, val_loader, optimizer, args):
    start = time.time()
    total_step = len(train_loader)
    best_acc, best_precision, best_recall, best_f1 = 0.0, 0.0, 0.0, 0.0

    for epoch in range(args.epochs):
        print('learning rate:', args.learning_rate)
        classifier.train()
        total_train_loss = 0.0
        batches_pred, batches_true = [], []


        # token_ids, mask_ids, seg_ids, y
        for batch_idx, batch in enumerate(tqdm(train_loader)):

            dataset = torch.tensor(batch, device=args.device)

            ehr_token_ids = dataset[:, :args.max_len]
            ehr_mask_ids = dataset[:, args.max_len:args.max_len+args.max_len]
            ehr_seg_ids = dataset[:, args.max_len*2: args.max_len*3]
            b_labels = dataset[:, -1].long()

            if args.loss_type == 'task':
                # 输入参数的顺序：token_ids, mask_ids, seg_ids
                prediction = classifier(
                    ehr_token_ids,
                    ehr_mask_ids,
                    ehr_seg_ids
                )
                loss = F.cross_entropy(prediction, b_labels)  # cross_entropy中包含着对prediction的log_softmax操作
            elif args.loss_type == 'task+pdg':
                # 输入参数的顺序：token_ids, mask_ids, seg_ids
                prediction, ehr_attentions = classifier(
                    ehr_token_ids,
                    ehr_mask_ids,
                    ehr_seg_ids
                )
                loss = F.cross_entropy(prediction, b_labels)  # cross_entropy中包含着对prediction的log_softmax操作
                pd_loss = alignment.pattern_decorrelation(ehr_attentions)
                print('{}--{}'.format(loss.item(), pd_loss.item()))
                loss = loss + args.pd_factor * pd_loss

            elif args.loss_type == 'task+adg':
                # 输入参数的顺序：token_ids, mask_ids, seg_ids
                prediction, ehr_attentions = classifier(
                    ehr_token_ids,
                    ehr_mask_ids,
                    ehr_seg_ids
                )

                loss = F.cross_entropy(prediction, b_labels)  # cross_entropy中包含着对prediction的log_softmax操作

                ad_loss = alignment.attention_diversity(ehr_attentions)
                print('{}--{}'.format(loss.item(), ad_loss.item()))
                loss = loss + args.ad_factor * ad_loss

            elif args.loss_type == 'task+both':
                # 输入参数的顺序：token_ids, mask_ids, seg_ids
                prediction, ehr_attentions = classifier(
                    ehr_token_ids,
                    ehr_mask_ids,
                    ehr_seg_ids
                )
                loss = F.cross_entropy(prediction, b_labels)  # cross_entropy中包含着对prediction的log_softmax操作

                pd_loss = alignment.pattern_decorrelation(ehr_attentions)
                ad_loss = alignment.attention_diversity(ehr_attentions)

                print('{}--{}--{}'.format(loss.item(), pd_loss.item(), ad_loss.item()))
                loss = loss + args.pd_factor * pd_loss + args.ad_factor * ad_loss

            # 更新模型参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            # 将该batch的预测结果保存起来
            probs = F.softmax(prediction, dim=1)
            y_pred = np.argmax(probs.detach().cpu().numpy(), axis=1)
            batches_pred.append(y_pred)
            batches_true.append(b_labels.detach().cpu().numpy())

        # 每一轮训练结束之后进行统计
        train_loss = total_train_loss / total_step
        batches_pred = np.concatenate(batches_pred, axis=0)
        batches_true = np.concatenate(batches_true, axis=0)
        acc, precision, recall, f1 = mednli_evaluation.get_f1(batches_pred, batches_true)


        print('EPOCH: {}---train_loss:{:.4f}---train_acc:{:.4f}----train_precision:{:.4f}---train_recall:{:.4f}----train_F1:{:.4f}'.format(epoch, train_loss, acc, precision, recall, f1))


        # 每隔eval_epochs测试一次模型的效果
        if epoch % args.eval_epochs == 0 or epoch == args.epochs-1:
            total_val_loss = 0
            val_steps = len(val_loader)
            classifier.eval()
            val_batches_pred, val_batches_true = [], []

            with torch.no_grad():
                # token_ids, mask_ids, seg_ids, y
                for batch_idx, batch in enumerate(val_loader):
                    dataset = torch.tensor(batch, device=args.device)
                    ehr_token_ids = dataset[:, :args.max_len]
                    ehr_mask_ids = dataset[:, args.max_len:args.max_len + args.max_len]
                    ehr_seg_ids = dataset[:, args.max_len * 2: args.max_len * 3]
                    b_labels = dataset[:, -1].long()
                    if args.loss_type == 'task':
                        # 输入参数的顺序：token_ids, mask_ids, seg_ids
                        prediction = classifier(
                            ehr_token_ids,
                            ehr_mask_ids,
                            ehr_seg_ids
                        )
                    else:
                        # 输入参数的顺序：token_ids, mask_ids, seg_ids
                        prediction, _ = classifier(
                            ehr_token_ids,
                            ehr_mask_ids,
                            ehr_seg_ids
                        )

                    loss = F.cross_entropy(prediction, b_labels)  # cross_entropy中包含着对prediction的log_softmax操作

                    total_val_loss += loss.item()
                    # 将该batch的预测结果保存起来
                    probs = F.softmax(prediction, dim=1)
                    y_pred = np.argmax(probs.detach().cpu().numpy(), axis=1)
                    val_batches_pred.append(y_pred)
                    val_batches_true.append(b_labels.detach().cpu().numpy())

            # 每一轮测试结束之后进行统计
            val_loss = total_val_loss / val_steps
            val_batches_pred = np.concatenate(val_batches_pred, axis=0)
            val_batches_true = np.concatenate(val_batches_true, axis=0)
            val_acc, val_precision, val_recall, val_f1 = mednli_evaluation.get_f1(val_batches_pred, val_batches_true)

            print('EPOCH: {}---val_loss:{:.4f}---val_acc:{:.4f}----val_precision:{:.4f}---val_recall:{:.4f}----val_F1:{:.4f}'.format(epoch, val_loss, val_acc, val_precision, val_recall, val_f1))

            # 保存模型
            if val_acc > best_acc:
                if '/' in args.model_name:
                    args.model_name = args.model_name.replace('/', '-')
                if not os.path.exists(args.save_path):
                    os.makedirs(args.save_path)

                torch.save(classifier.state_dict(), os.path.join(args.save_path, 'model_%s_%s.pt' % (args.model_name, args.loss_type)), _use_new_zipfile_serialization=False)
                best_acc = val_acc
                best_precision = val_precision
                best_recall = val_recall
                best_f1 = val_f1
    print('best_val_acc:{:.4f}----best_val_precision:{:.4f}---best_val_recall:{:.4f}----best_val_F1:{:.4f}'.format(best_acc, best_precision, best_recall, best_f1))
    end = time.time()
    # hours, rem = divmod(end - start, 3600)
    # minutes, seconds = divmod(rem, 60)
    # print('{:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds))
    seconds = end - start
    print('time cost per-epoch:', seconds/args.epochs)



def test(classifier, test_loader, args):

    total_step = len(test_loader)
    classifier.eval()
    start = time.time()
    total_test_loss = 0.0

    test_batches_pred, test_batches_true = [], []
    test_batches_score = []
    with torch.no_grad():
        # token_ids, mask_ids, seg_ids, y
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            dataset = torch.tensor(batch, device=args.device)
            ehr_token_ids = dataset[:, :args.max_len]
            ehr_mask_ids = dataset[:, args.max_len:args.max_len + args.max_len]
            ehr_seg_ids = dataset[:, args.max_len * 2: args.max_len * 3]
            b_labels = dataset[:, -1].long()

            if args.loss_type == 'task':
                # 输入参数的顺序：token_ids, mask_ids, seg_ids
                prediction = classifier(
                    ehr_token_ids,
                    ehr_mask_ids,
                    ehr_seg_ids
                )
            else:
                # 输入参数的顺序：token_ids, mask_ids, seg_ids
                prediction, _ = classifier(
                    ehr_token_ids,
                    ehr_mask_ids,
                    ehr_seg_ids
                )

            loss = F.cross_entropy(prediction, b_labels)
            total_test_loss += loss.item()

            # 将该batch的预测结果保存起来
            probs = F.softmax(prediction, dim=1)
            y_pred = np.argmax(probs.detach().cpu().numpy(), axis=1)
            test_batches_pred.append(y_pred)
            test_batches_true.append(b_labels.detach().cpu().numpy())

            for i in range(len(prediction)):
                score = prediction[i][b_labels[i]].item()
                test_batches_score.append(score)

            # 随机选择一个样本
            # random_idx = random.choice(args.batch_size)
            # utils.plot_attention_head(ehr_attentions[random_idx])

        # 每一轮训练结束之后进行统计
        test_loss = total_test_loss / total_step
        test_batches_pred = np.concatenate(test_batches_pred, axis=0)
        test_batches_true = np.concatenate(test_batches_true, axis=0)
        test_acc, test_precision, test_recall, test_f1 = mednli_evaluation.get_f1(test_batches_pred, test_batches_true)
        end = time.time()
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)
        print('test_loss:{:.4f}---test_acc:{:.4f}----test_precision:{:.4f}---test_recall:{:.4f}----test_F1:{:.4f}'.format(test_loss, test_acc, test_precision, test_recall, test_f1))
        print('{:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds))

        return test_batches_score