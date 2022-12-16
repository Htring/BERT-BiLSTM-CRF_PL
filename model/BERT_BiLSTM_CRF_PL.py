#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: BERT_BiLSTM_CRF_PL.py
@time:2022/05/16
@description:
"""

import torch
from argparse import ArgumentParser
from typing import Union, Dict, List, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
from seqeval.metrics import classification_report, f1_score
from pytorch_lightning import LightningModule
from torch import nn, Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR
from .BERT_BiLSTM_CRF import BERTBiLSTMCRF
from transformers import AdamW


class BERTBiLSTMCRF_PL(LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--lr", type=float, default=3e-05)
        parser.add_argument('--hidden_size', type=int, default=128)
        parser.add_argument('--lstm_num_layers', type=int, default=1)
        parser.add_argument("--dropout", type=float, default=0.5)
        parser.add_argument("--weight_decay", type=float, default=9e-3)
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.hyper_params = hparams
        self.lr = hparams.lr
        self.bert_bilstm_crf = BERTBiLSTMCRF(hparams)
        self.idx2tag = hparams.idx2tag

    def configure_optimizers(self):
        """
        配置优化器
        :return:
        """
        # 初始化模型参数优化器
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.lr,
                          weight_decay=self.hyper_params.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, 5, eta_min=0.0001)
        optimizer_dict = {'optimizer': optimizer, 'lr_scheduler': scheduler}
        return optimizer_dict

    def forward_train(self,  input_ids, attention_mask, segment_ids, label_ids):
        """
        model train
        :return:
        """
        return self.bert_bilstm_crf(input_ids, attention_mask, segment_ids, label_ids)

    def forward(self, sentences_idx, attention_mask, segment_ids):
        """
        模型落地推理
        :param sentences_idx:
        :return:
        """
        return self.bert_bilstm_crf.decode(sentences_idx, attention_mask, segment_ids)

    def training_step(self, batch, batch_idx, optimizer_idx=None) -> Union[int,
                                                                           Dict[str, Union[Tensor, Dict[str, Tensor]]]]:
        """
        模型训练的前向传播过程
        :param batch:批次数据
        :param batch_idx:
        :param optimizer_idx:
        :return:
        """
        input_ids, attention_mask, segment_ids, label_ids = batch
        loss = self.forward_train(input_ids, attention_mask, segment_ids, label_ids)
        res = {"log": {"loss": loss}, "loss": loss}
        return res

    def validation_step(self, batch, batch_idx) -> Union[int, Dict[str, Union[Tensor, Dict[str, Tensor]]]]:
        """
        开发集数据验证过程
        :param batch: 批次数据
        :param batch_idx:
        :return:
        """
        input_ids, attention_mask, segment_ids, label_ids = batch
        loss = self.forward_train( input_ids, attention_mask, segment_ids, label_ids)
        loss = loss.mean()
        return {"target": label_ids, "pred": self.bert_bilstm_crf.decode(input_ids, attention_mask, segment_ids),
                "loss": loss}

    def validation_epoch_end(self, outputs: Union[List[Dict[str, Tensor]],
                                                  List[List[Dict[str, Tensor]]]]) -> Dict[str, Dict[str, Tensor]]:
        """
        验证数据集
        :param outputs: 所有batch预测结果 validation_step的返回值构成的一个list
        :return:
        """
        return self._decode_epoch_end(outputs)

    def _decode_epoch_end(self, outputs: Union[List[Dict[str, Tensor]],
                                               List[List[Dict[str, Tensor]]]]) -> Dict[str, Dict[str, Tensor]]:
        """
        对批次预测的结果进行整理，评估对应的结果
        :return:
        """
        gold_list, pred_list = [], []  # 原始标签以及模型预测结果
        for batch_result in outputs:
            batch_size = batch_result['target'].shape[0]
            for i in range(batch_size):
                sentence_gold, sentence_pred = [], []
                for j in range(1, len(batch_result['target'][i])):
                    gold = self.idx2tag.get(batch_result['target'][i][j].item())
                    pred = self.idx2tag.get(batch_result['pred'][i][j].item())
                    if gold == "<pad>":
                        break
                    sentence_gold.append(gold)
                    sentence_pred.append(pred)
                gold_list.append(sentence_gold)
                pred_list.append(sentence_pred)
        print("\n", classification_report(gold_list, pred_list))
        f1 = torch.tensor(f1_score(gold_list, pred_list))
        tqdm_dict = {'val_f1': f1}
        results = {"progress_bar": tqdm_dict, "log": {'val_f1': f1, "step": self.current_epoch}}
        self.log("val_f1", f1)
        return results

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        """
        程序测试模块
        :param batch:
        :param batch_idx:
        :return:
        """
        input_ids, attention_mask, segment_ids, label_ids = batch
        loss = self.forward_train(input_ids, attention_mask, segment_ids, label_ids)
        loss = loss.mean()
        return {"target": label_ids,
                "pred": self.bert_bilstm_crf.decode(input_ids, attention_mask, segment_ids), "loss": loss}

    def test_epoch_end(self, outputs: Union[List[Dict[str, Tensor]],
                                            List[List[Dict[str, Tensor]]]]) -> Dict[str, Dict[str, Tensor]]:
        """
        测试集的评估
        :param outputs:测试集batch预测完成结果
        :return:
        """
        return self._decode_epoch_end(outputs)
