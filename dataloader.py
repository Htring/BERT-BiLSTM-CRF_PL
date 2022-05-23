#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: dataloader.py
@time:2022/03/20
@description:
"""
import argparse
import json
from typing import Optional, List
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from model.util import read_tsv
import os
from loguru import logger
from transformers import AutoTokenizer, PreTrainedTokenizer



class InputExample(object):
    __doc__ = """ 样本数据 """

    def __init__(self, guid, text, label=None):
        """
        construct a input example
        :param guid: unique id for the example
        :param text: the tokenized text of the sequence, for single sequence tasks, only this sequence must be specified
        """
        self.guid = guid
        self.text = text
        self.label = label

    def __str__(self):
        return f"guid:{self.guid}, text:{self.text}, label:{self.label}"


class InputFeatures(object):
    __doc__ = """ a single set of features of data """

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


def convert_example2features(examples,
                             label_list: list,
                             max_seq_length,
                             tokenizer: PreTrainedTokenizer,
                             pad_token_label_id=-1,):
    label_map = {label: index for index, label in enumerate(label_list)}
    label_pad_id = label_map.get("O")
    features = []
    for ex_index, example in enumerate(examples):
        label_ids, tokens, origin_tokens = [], [], []
        for word, label in zip(example.text, example.label):
            word_tokens = tokenizer.tokenize(word)
            origin_tokens.append(word)
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
        labels_len = len(label_ids)
        tokens_len = len(tokens)
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[: (max_seq_length - 2)]
            label_ids = label_ids[: (max_seq_length - 2)]

        tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
        origin_tokens = [tokenizer.cls_token] + origin_tokens + [tokenizer.sep_token]
        label_ids = [label_pad_id] + label_ids + [label_pad_id]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_len = len(input_ids)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)
        segment_len = len(segment_ids)
        # pad to max length
        pad_length = max_seq_length - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * pad_length
        input_mask += [0] * pad_length
        label_ids += [label_pad_id] * pad_length
        segment_ids += [0] * pad_length
        if len(label_ids) != max_seq_length:
            logger.info("*** Error ***")
            logger.info("guid: %s" % example.guid)
            logger.info("origin tokens: %s" % " ".join([str(x) for x in origin_tokens]))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("labels_len: %s" % str(labels_len))
            logger.info("tokens_len: %s" % str(tokens_len))
            logger.info("input_len: %s" % str(input_len))
            logger.info("segment_ids: %s" % str(segment_len))
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        features.append(InputFeatures(input_ids=input_ids,
                                      input_mask=input_mask,
                                      segment_ids=segment_ids,
                                      label_ids=label_ids))
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

    return features


def load_and_cache_examples(args, examples, tokenizer, labels, pad_token_label_id, mode):
    """
    加载或缓存样本数据
    :param args: 通用参数
    :param examples: examples list
    :param tokenizer: 分词器
    :param labels: 使用的标签
    :param pad_token_label_id: label pad 的id
    :param mode: 数据类型
    :return:
    """
    # load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, "cache_{}_{}_{}".format(mode,
                                                                               list(filter(None,
                                                                                           args.model_name_or_path.split(
                                                                                               "/"))).pop(),
                                                                               str(args.max_seq_length)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cache file %s" % cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s" % args.data_dir)
        features = convert_example2features(examples,
                                            labels,
                                            args.max_seq_length,
                                            tokenizer,
                                            pad_token_label_id=pad_token_label_id, )
        logger.info("Saving features into cached file %s" % cached_features_file)
        torch.save(features, cached_features_file)
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset


class NERDataProcessor(object):

    def __init__(self, corpus_dir):
        self.corpus_dir = corpus_dir
        self.train_examples, self.dev_examples, self.test_examples = None, None, None
        self.labels_list = None

    def get_train_example(self):
        if self.train_examples is None:
            self.train_examples = self._create_examples(read_tsv(os.path.join(self.corpus_dir, "train.txt")),
                                                        "train")
        return self.train_examples

    def get_dev_example(self):
        if self.dev_examples is None:
            self.dev_examples = self._create_examples(read_tsv(os.path.join(self.corpus_dir, "dev.txt")),
                                                      "dev")
        return self.dev_examples

    def get_test_example(self):
        if self.test_examples is None:
            self.test_examples = self._create_examples(read_tsv(os.path.join(self.corpus_dir, "test.txt")),
                                                       "test")
        return self.test_examples

    def get_labels(self):
        if self.labels_list is None:
            labels = set()
            for example in self.get_train_example():
                labels.update(example.label)
            labels = list(labels)
            labels.sort()
            self.labels_list = labels
        return self.labels_list

    @staticmethod
    def _create_examples(lines: list, set_type: str) -> List[InputExample]:
        """
        create examples
        :param lines: token sentence
        :param set_type: examples type
        :return:
        """
        examples = []
        for index, (sentence, label) in enumerate(lines, start=1):
            guid = "%s-%s" % (set_type, index)
            text = sentence
            label = label
            examples.append(InputExample(guid=guid,
                                         text=text,
                                         label=label))
        return examples


class NERDataModule(pl.LightningDataModule):

    def __init__(self, args, pad_token_label_id):
        super().__init__()
        self.data_path = args.data_dir
        self.batch_size = args.batch_size
        self.pad_token_label_id = pad_token_label_id
        self.idx2tag = None
        self.args = args
        self.ner_data_processor = NERDataProcessor(self.data_path)
        self.tokenizer = AutoTokenizer.from_pretrained(args.pre_train_path)
        self.train_iter, self.val_iter, self.test_iter = None, None, None
        self.setup()

    def setup(self, stage: Optional[str] = None) -> None:
        labels_list = self.ner_data_processor.get_labels()
        train_examples = self.ner_data_processor.get_train_example()
        dev_examples = self.ner_data_processor.get_dev_example()
        test_examples = self.ner_data_processor.get_test_example()
        train_features = load_and_cache_examples(self.args, train_examples, self.tokenizer, labels_list,
                                                 self.pad_token_label_id, mode="train")
        dev_features = load_and_cache_examples(self.args, dev_examples, self.tokenizer, labels_list,
                                               self.pad_token_label_id, mode="dev")
        test_features = load_and_cache_examples(self.args, test_examples, self.tokenizer, labels_list,
                                                self.pad_token_label_id, mode="test")
        train_sampler = RandomSampler(train_features)
        self.train_iter = DataLoader(train_features, sampler=train_sampler, batch_size=self.batch_size)
        self.val_iter = DataLoader(dev_features, batch_size=self.batch_size)
        self.test_iter = DataLoader(test_features, batch_size=self.batch_size)
        self.idx2tag = {index: tag for index, tag in enumerate(labels_list)}

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.train_iter

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.test_iter

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.val_iter

    def save_dict(self, data_dir):
        with open(os.path.join(data_dir, "index2tag.txt"), 'w', encoding='utf8') as writer:
            json.dump(self.idx2tag, writer, ensure_ascii=False)
