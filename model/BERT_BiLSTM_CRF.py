#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: BERT_BiLSTM_CRF.py
@time:2022/05/16
@description:
"""
import torch
from torch import nn
from transformers import BertModel
from torchcrf import CRF


class BERTBiLSTMCRF(nn.Module):
    __doc__ = """ bert bilstm crf """

    def __init__(self, param):
        super().__init__()
        self.bert = BertModel.from_pretrained(param.pre_train_path)
        self.lstm = nn.LSTM(input_size=self.bert.config.hidden_size,
                            bidirectional=True,
                            num_layers=param.lstm_num_layers,
                            hidden_size=param.hidden_size,
                            batch_first=True)
        self.fc = nn.Linear(in_features=param.hidden_size * 2, out_features=param.num_labels)
        self.crf = CRF(num_tags=param.num_labels, batch_first=True)
        self.dropout = nn.Dropout(param.dropout)

    def forward(self, input_ids, attention_mask, segment_ids, tags_idx):
        embeds = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)[0]
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)
        lstm_out = self.fc(lstm_out)
        mask = tags_idx != 1
        loss = self.crf(lstm_out, tags_idx, mask)
        return -loss

    def decode(self, input_ids, attention_mask, segment_ids):
        embeds = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)[0]
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)
        lstm_out = self.fc(lstm_out)
        results = self.crf.decode(lstm_out)
        result_tensor = []
        for result in results:
            result_tensor.append(torch.tensor(result))
        return torch.stack(result_tensor)