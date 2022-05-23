#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: trainer.py
@time:2021/11/21
@description:
"""
import json
import os
from argparse import ArgumentParser
import torch
from pytorch_lightning import Trainer
from model.BERT_BiLSTM_CRF_PL import BERTBiLSTMCRF_PL
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import pytorch_lightning as pl
from dataloader import NERDataModule
from transformers import AutoTokenizer


pl.seed_everything(2022)


def train(args):
    path_prefix = "model_save"
    os.makedirs(path_prefix, exist_ok=True)
    ner_dm = NERDataModule(args, pad_token_label_id=torch.nn.CrossEntropyLoss().ignore_index)
    args.idx2tag = ner_dm.idx2tag
    args.num_labels = len(args.idx2tag)
    if args.load_pre:
        model = BERTBiLSTMCRF_PL.load_from_checkpoint(args.ckpt_path, hparams=args)
    else:
        model = BERTBiLSTMCRF_PL(args)
    lr_logger = LearningRateMonitor()
    checkpoint_callback = ModelCheckpoint(save_top_k=3,
                                          monitor="val_f1",
                                          mode="max",
                                          dirpath=path_prefix,
                                          filename="ner-{epoch:03d}-{val_f1:.3f}", )
    trainer = Trainer.from_argparse_args(args, callbacks=[lr_logger,
                                                          checkpoint_callback],
                                         gpus=1,
                                         max_epochs=args.epoch)
    if args.save_state_dict:
        if len(os.name) > 0:
            ner_dm.save_dict(path_prefix)

    if args.train:
        trainer.fit(model=model, datamodule=ner_dm)

    if args.test:
        trainer.test(model, ner_dm)


def model_use(param):
    model_dir = os.path.dirname(param.ckpt_path)
    tokenizer = AutoTokenizer.from_pretrained(param.pre_train_path)

    def _load_dict():
        with open(os.path.join(model_dir, 'index2tag.txt'), 'r', encoding='utf8') as reader:
            i2t_dict: dict = json.load(reader)
        i2t_dict = {int(index): tag for index, tag in i2t_dict.items()}
        return i2t_dict

    def num_data(content: str, max_length):
        res = tokenizer.encode_plus(text=list(content),
                                    return_tensors='pt',
                                    max_length=max_length,
                                    is_split_into_words=True,
                                    )
        return res

    index2tag = _load_dict()
    param.num_labels = len(index2tag)
    param.idx2tag = index2tag
    model = BERTBiLSTMCRF_PL.load_from_checkpoint(param.ckpt_path, hparams=param)

    test_data = "常建良，男，"
    # encode
    input_data = num_data(test_data, param.max_seq_length)
    # predict
    predict = model(input_data["input_ids"], input_data['attention_mask'], input_data['token_type_ids'])[0][1: -1]
    result = []
    # decode
    for predict_id in predict:
        result.append(index2tag.get(predict_id.item()))
    print(predict)
    print(result)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--load_pre", default=True, action="store_true")
    parser.add_argument("--ckpt_path", type=str, default="model_save/ner-epoch=019-val_f1=0.936.ckpt")
    parser.add_argument("--test", action="store_true", default=True)
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--save_state_dict", default=True, action="store_true")
    parser.add_argument("--data_dir", default="data/corpus", type=str, help="train data dir")
    parser.add_argument("--max_seq_length", default=128, type=int, help="sentence max length")
    parser.add_argument("--pre_train_path", default="pre_model/bert-base-chinese", type=str)
    parser.add_argument("--model_name_or_path", default="bert_bilstm_crf", type=str)
    parser.add_argument("--overwrite_cache", default=False, type=bool)
    parser.add_argument("--epoch", default=50, type=int)
    parser = BERTBiLSTMCRF_PL.add_model_specific_args(parser)
    params = parser.parse_args()
    # train(params)
    model_use(params)
