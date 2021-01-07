# *- coding: utf-8 -*-

# =================================
# time: 2020.9.09
# author: @tangzhilin(Resigned)
# function: 由该文件将数据库内的原始数据获取, 并在拆分为训练数据, 测试数据, 验证数据之后
# 分发给模型, 模型具体的数据格式依旧在模型文件内部处理
# =================================
import pandas as pd
import numpy as np

from MysqlLink.sql_link import get_sql_fit_datas, get_sql_labels_keywords

from Transformer.XLnet_model import fit_model as xlnet_fit
from Transformer.AlBert_model import fit_model as albert_fit
from Transformer.Bert_model import fit_model as bert_fit
from Transformer.Robert_model import fit_model as robert_fit



def datas_base():
    datas = get_sql_fit_datas(0, 0)

    train = datas.sample(frac=0.05)
    train = train[['solve', 'text']]

    test = train.sample(frac=0.15)
    train = train.loc[list(set(train.index) - set(test.index))]
    valid = train.sample(frac=0.1)
    train = train.loc[list(set(train.index) - set(valid.index))]
    # 生成训练集和测试集

    return train, test, valid



def bert_fit_model():
    # train, test, valid = datas_base()
    from datetime import datetime
    from sklearn.model_selection import train_test_split
    train = pd.read_excel('../test_data/train.xlsx')
    test = pd.read_excel('../test_data/test.xlsx')
    train = train.sample(frac=0.09)
    train, valid = train_test_split(train, test_size=0.05)
    time_start = datetime.now()
    bert_fit(train, test, valid)
    end_time = datetime.now()
    print("cost time: ", end_time - time_start)


bert_fit_model()


def xlnet_fit_model():
    # train, test, valid = datas_base()
    from sklearn.model_selection import train_test_split
    from datetime import datetime
    time_start = datetime.now()
    # datas = pd.read_csv('../test_data/待训练数据_汇总_第三版.csv', error_bad_lines=True)
    train = pd.read_excel('../test_data/train.xlsx')
    test = pd.read_excel('../test_data/test.xlsx')
    # train, valid = train_test_split(train, test_size=0.15)
    train = train.sample(frac=0.999)
    xlnet_fit(train, test, None)
    print("cost time: ", datetime.now() - time_start)


def albert_fit_model():
    train, test, valid = datas_base()

    def link_text_solve(x, y):
        # 将数据拼接成("xxxx, yyy, zz", 1)的格式
        return (x, y)

    train['link_text'] = train.apply(
        lambda x: link_text_solve(x['text'], x['solve']), axis=1
    )
    test['link_text'] = train.apply(
        lambda x: link_text_solve(x['text'], x['solve']), axis=1
    )
    valid['link_text'] = train.apply(
        lambda x: link_text_solve(x['text'], x['solve']), axis=1
    )
    train = train['link_text'].values
    test = test['link_text'].values
    valid = valid['link_text'].values

    albert_fit(train, test, valid)


def data_base_robert():
    datas = get_sql_fit_datas(0, 0)
    labels_keywords = get_sql_labels_keywords()
    train = datas.sample(frac=0.05)
    train.columns = ['labels', 'text']

    return train, labels_keywords


def robert_fit_model():
    """
    funcation: 找到标签对应的关键词在文本中的位置
    用于 进行文本->输出标签

    """
    datas, word_labels = data_base_robert()

    def find_size(text, label):
        label_key = word_labels.loc[word_labels['labels'] == label]
        size_list = []
        for kw in label_key['keyword'].values:
            if kw in text:
                start = text.find(kw)
                end = start + 1 + len(kw)
                size_list.append(str(start) + '|' + str(end))
        return size_list if len(size_list) >= 1 else np.nan

    datas['start_end'] = datas.apply(
        lambda x: find_size(x['text'], x['labels']), axis=1
    )

    df = datas.copy()
    datas.dropna(subset=['start_end'], inplace=True)

    df.drop(index=datas.index.values, inplace=True)
    # 目的是将文本中有多个关键词的拆分, 然后展开
    datas = pd.DataFrame({
        'text': datas.text.repeat(datas.start_end.str.len()),
        'labels': datas.labels.repeat(datas.start_end.str.len()),
        'start_end': np.concatenate(datas.start_end.values)
    })

    sample_index = pd.DataFrame(datas.index.value_counts(), columns=['counts'])

    sample_index = sample_index.loc[sample_index['counts'] >= 2].index
    ds_1 = datas.drop(index=sample_index)
    ds_2 = datas.loc[sample_index]

    def shuffle_text_keyword(text):
        # 将部分含多个关键词的句子顺序打乱
        text = text.split(' ')
        index = np.arange(len(text))

        np.random.shuffle(index)

        text = " ".join(text[i] for i in index).strip()

        return text

    ds_2['text'] = ds_2.apply(
        lambda x: shuffle_text_keyword(x['text']), axis=1
    )
    ds = pd.concat([ds_1, ds_2], ignore_index=True)

    df['start_end'] = df['start_end'].apply(
        lambda x: [0, 0]
    )
    ds['start_end'] = ds['start_end'].apply(
        lambda x: [int(i) + 1 for i in str(x).split('|')]
    )
    # +1的原因是token编码后会在开头添加一个101也就是空的意思

    df = pd.concat([df, ds], ignore_index=True)

    def link_text_solve(x, y):
        # 将数据拼接成("xxxx, yyy, zz", 1)的格式
        return (x, y)

    df['link_text'] = df.apply(
        lambda x: link_text_solve(x['text'], x['start_end']), axis=1
    )

    df = df['link_text'].values

    robert_fit(df)

