#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: util.py
@time:2022/05/18
@description:
"""


def read_tsv(file_path, split_seg="\t"):
    """
    read tsv style data
    :param file_path: file path
    :param split_seg: seg
    :return: [(sentence, label), ...]
    """
    data = []
    sentence = []
    label = []
    with open(file_path, 'r', encoding='utf8') as reader:
        for line in reader:
            line = line.strip()
            if not line:
                if sentence:
                    data.append((sentence, label))
                    sentence = []
                    label = []
                continue
            if split_seg not in line:
                split_seg = " "
            splits = line.split(split_seg)
            sentence.append(splits[0])
            label.append(splits[-1])
    if sentence:
        data.append((sentence, label))
    return data
