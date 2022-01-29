#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2021/9/3 15:59
# @Author :
# @File : train_test.py
# @Function:

import torch
import time
import torch.nn.functional as F
import os
import alignment

def train(classifier, train_loader, optimizer, args):
    start = time.time()
    total_step = len(train_loader)

    for epoch in range(args.epochs):
        classifier.train()

        total_train_loss = 0.0

        #token_ids, mask_ids, seg_ids, y
        for batch_idx, batch in enumerate(train_loader):

            dataset = torch.tensor(batch, device=args.device)
            token_ids = dataset[:, :args.max_len].long()
            mask_ids = dataset[:, args.max_len:args.max_len + args.max_len].long()
            seg_ids = dataset[:, args.max_len * 2: args.max_len * 3].long()
            b_labels = dataset[:, args.max_len * 3:].float()

            # 只使用分类损失
            if args.loss_type == 'task':
                # 输入参数的顺序：token_ids, mask_ids, seg_ids
                prediction = classifier(
                    token_ids,
                    mask_ids,
                    seg_ids
                )

                loss = F.binary_cross_entropy_with_logits(prediction, b_labels)

            elif args.loss_type == 'task+pdg':
                # 输入参数的顺序：token_ids, mask_ids, seg_ids
                prediction, ehr_attentions = classifier(
                    token_ids,
                    mask_ids,
                    seg_ids
                )

                loss = F.binary_cross_entropy_with_logits(prediction, b_labels)

                pd_loss = alignment.pattern_decorrelation(ehr_attentions)
                print('{}--{}'.format(loss.item(), pd_loss.item()))
                loss = loss + args.pd_factor * pd_loss

            elif args.loss_type == 'task+adg':
                # 输入参数的顺序：token_ids, mask_ids, seg_ids
                prediction, ehr_attentions = classifier(
                    token_ids,
                    mask_ids,
                    seg_ids
                )

                loss = F.binary_cross_entropy_with_logits(prediction, b_labels)

                ad_loss = alignment.attention_diversity(ehr_attentions)
                print('{}--{}'.format(loss.item(), ad_loss.item()))
                loss = loss + args.ad_factor * ad_loss

            elif args.loss_type == 'task+both':
                # 输入参数的顺序：token_ids, mask_ids, seg_ids
                prediction, ehr_attentions = classifier(
                    token_ids,
                    mask_ids,
                    seg_ids
                )

                loss = F.binary_cross_entropy_with_logits(prediction, b_labels)

                pd_loss = alignment.pattern_decorrelation(ehr_attentions)
                print('{}--{}'.format(loss.item(), pd_loss.item()))
                loss = loss + args.pd_factor * pd_loss

                ad_loss = alignment.attention_diversity(ehr_attentions)
                print('{}--{}--{}'.format(loss.item(), pd_loss.item(), ad_loss.item()))
                loss = loss + args.pd_factor * pd_loss + args.ad_factor * ad_loss


            #更新模型参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        #每一轮训练结束之后进行统计
        train_loss = total_train_loss/total_step
        print('EPOCH: {}  train_loss:{:.4f}'.format(epoch, train_loss))

        # 每一轮之后都保存模型
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        if '/' in args.model_name:
            args.model_name = args.model_name.replace('/', '-')
        torch.save(classifier.state_dict(), os.path.join(args.save_path, 'model_%s_%s_%d.pt' % (args.model_name, args.loss_type, epoch)), _use_new_zipfile_serialization=False)

    end = time.time()
    # hours, rem = divmod(end-start, 3600)
    # minutes, seconds = divmod(rem, 60)
    # print('{:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds))
    seconds = end - start
    print('time cost per-epoch:', seconds/args.epochs)


def test(test_classifier, test_loader, args):
    total_step = len(test_loader)

    test_classifier.eval()
    start = time.time()
    total_test_loss = 0
    test_score_list = []

    with torch.no_grad():
        #token_ids, mask_ids, seg_ids, y
        for batch_idx, batch in enumerate(test_loader):

            dataset = torch.tensor(batch, device=args.device)
            token_ids = dataset[:, :args.max_len].long()
            mask_ids = dataset[:, args.max_len:args.max_len + args.max_len].long()
            seg_ids = dataset[:, args.max_len * 2: args.max_len * 3].long()
            b_labels = dataset[:, args.max_len * 3:].float()

            if args.loss_type == 'task':
                # 输入参数的顺序：token_ids, mask_ids, seg_ids
                prediction = test_classifier(
                    token_ids,
                    mask_ids,
                    seg_ids
                )
            else:
                # 输入参数的顺序：token_ids, mask_ids, seg_ids
                prediction, _ = test_classifier(
                    token_ids,
                    mask_ids,
                    seg_ids
                )

            loss = F.binary_cross_entropy_with_logits(prediction, b_labels)
            total_test_loss += loss.item()

            # 将预测结果保存起来
            test_score_list += prediction.detach().cpu().numpy().tolist()

        test_loss = total_test_loss * 1.0 / total_step
        print('test_loss:{:.4f}'.format(test_loss))
        return test_score_list


