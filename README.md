
## 背景
NER任务毋庸多言，之前也是从HMM，BiLSTM-CRF，IDCNN-CRF一路实现，也看到各个模型的效果和性能。在BERT大行其道的时期，不用BERT做一下BERT那就有点out了，毕竟基于BERT的衍生语言模型也变得更加强悍。不过当前使用BERT+softmax既可以做到非常好的效果，接上BiLSTM以及再使用CRF解码，主要是为了充分理解各层直接的衔接关系等。除此之外，模型在训练过程中需要一些小tricks，如：lr_scheduler，warmup等都需要我们慢慢理解其在背后使用的意义和效果等。

当然，如果你对之前NER实现的代码感兴趣的话，可以看看这些文章：[【NLP】基于隐马尔可夫模型（HMM）的命名实体识别（NER）实现](https://blog.csdn.net/meiqi0538/article/details/124065834?spm=1001.2014.3001.5501)、[【NLP】基于Pytorch lightning与BiLSTM-CRF的NER实现](https://blog.csdn.net/meiqi0538/article/details/124209678?spm=1001.2014.3001.5501)、[【NLP】基于Pytorch的IDCNN-CRF命名实体识别(NER)实现](https://blog.csdn.net/meiqi0538/article/details/124644060?spm=1001.2014.3001.5501)。

当然本程序在实现完相关模型后，也将源码上传到了GitHub上了，有兴趣看源码的可以自拿：[https://github.com/Htring/BERT-BiLSTM-CRF_PL](https://github.com/Htring/BERT-BiLSTM-CRF_PL)。欢迎各位小可爱留言交流。除此之外还需要说的是，一个模型在训练的通常实现后能够达到一个差不多的效果，有时还可以通过一些“炼丹”小技巧有些许的提升。

本文源码讲解参见博客：[【NLP】基于BERT-BiLSTM-CRF的NER实现](https://piqiandong.blog.csdn.net/article/details/124920451)

## 主要程序包依赖

```text
pytorch-crf             0.7.2
pytorch-lightning       1.5.0
torch                   1.10.0
torchmetrics            0.10.3
torchtext               0.11.0
transformers            4.24.0
```

预训练模型需要放在：`pre_model/bert-base-chinese`，也可以根据自己的情况进行调整。

## 数据来源
本程序数据来源与之前NER任务相同，地址：[https://github.com/luopeixiang/named_entity_recognition](https://github.com/luopeixiang/named_entity_recognition).

为了能够使用seqeval工具评估模型效果，将原始数据中“M-”,"E-"开头的标签处理为“I-”.

## 程序结构
程序设计结构依然像以往的形式，包括如下三个模块：

数据处理模块：dataloader.py
模型实现模块： BERT_BiLSTM_CRF.py
模型训练封装模块：BERT_BiLSTM_CRF_PL.py
模型训练和模型使用模块：trainner.py

## 模型效果
选择最好的模型（ner-epoch=019-val_f1=0.936.ckpt）在测试集效果如下：
```text
Testing: 100%|██████████| 15/15 [00:04<00:00,  3.12it/s]
               precision    recall  f1-score   support

        CONT       1.00      1.00      1.00        28
         EDU       0.90      0.92      0.91       112
         LOC       1.00      1.00      1.00         6
        NAME       0.98      0.99      0.99       112
         ORG       0.91      0.95      0.93       551
         PRO       0.66      0.76      0.70        33
        RACE       1.00      1.00      1.00        14
       TITLE       0.94      0.95      0.95       762

   micro avg       0.92      0.95      0.94      1618
   macro avg       0.92      0.95      0.93      1618
weighted avg       0.93      0.95      0.94      1618

--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'val_f1': 0.9365466833114624}
--------------------------------------------------------------------------------
Testing: 100%|██████████| 15/15 [00:07<00:00,  2.13it/s]

```

## 总结
万事开头难，先开始去实现一个模块可能因为对于一些理论不是很熟悉，不知道从何下手。在这种情况下可以参考一些开源代码，根据开源代码、自己的理解、理论去实现程序，在不懂一些理论通过在百度，bing等搜索去了解自己的知识盲区，耐住性子，其实了解和搞懂一路走下来就是一个时间问题。当然本文也看了网上的一些源码，例如：[https://github.com/lonePatient/BERT-NER-Pytorch](https://github.com/lonePatient/BERT-NER-Pytorch)、[https://github.com/taishan1994/pytorch_bert_bilstm_crf_ner](https://github.com/taishan1994/pytorch_bert_bilstm_crf_ner)等。看看别人的代码，取长补短应该是学习的一个好方法。

当然本文在改写的过程中难免在一些细节上处理不好，欢迎留言交流哦。


## 联系我

1. 我的github：[https://github.com/Htring](https://github.com/Htring)
2. 我的csdn：[科皮子菊](https://piqiandong.blog.csdn.net/)
3. 我订阅号：AIAS编程有道
   ![AIAS编程有道](https://s2.loli.net/2022/05/05/DS37LjhBQz2xyUJ.png)
4. 知乎：[皮乾东](https://www.zhihu.com/people/piqiandong)
