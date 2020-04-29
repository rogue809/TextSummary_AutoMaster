# TextSummary_AutoMaster
## 关键词
NLP文本摘要模型    中文客服对话生成报告    word2vec预训练词向量    seq2seq模型    attention机制    BeamSearch、PGN优化


## 数据集
训练集（82943条记录）建立模型，基于汽车品牌、车系、车况的问题与对话的文本，输出建议的报告文本。如下：
| Brand | Model | Question | Dialogue | Report |
| --- | --- | --- | --- | ---|
| 奔驰 | 奔驰GL级 |  变速箱旁边漏机油 | 技师说：具体是哪个部位呢？是发动机和变速器正中间位置吗？车主说... | 随时联系 |

测试集（20000条记录）使用训练好的模型，输出建议报告的结果文件。


## 项目文件说明
seq2seq模型的baseline版本， 使用tensorflow2.0搭建。

### 后续
paddlepaddle版本;
transformer优化模型;
用BERT做预训练;
pytorch版


## 日志
2020/4/27 分词和清洗数据，

2020/4/29 建立vocab词汇表

2020// word2vec预训练词向量

2020// 搭建seq2seq模型(encoder,decoder,attention层）

2020// 搭建模型，训练
效果：

2020// beam search加入
效果：

2020// PGN加入
效果：



## 代码部分
1.preprocess.py
完成原始数据的解析与存储


2.data_reader.py
读取数据，并建立vocab


3.utils/build_w2v.py
以vocab中的index为key值构建embedding_matrix，构建embedding_matrix
利用word2vector方法预训练词向量，
补充：gensim中Word2Vec或Fasttext两种方式训练词向量


4.main.py
完成模型的训练和预测
- 构建Seq2seq模型中的Encoder层和Decoder层
- 构建Seq2seq模型中的Attention
- loss函数采用    ，优化器采用   


5.beam search


6.PGN








