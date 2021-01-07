# *- coding: utf-8 -*-

# =================================
# time: 2020.7.25
# author: @tangzhilin(Resigned)
# function: 数据初步清洗
# update: 8.14 --- 重新设计了标签与是否解决的数据格式
# update: 8.20 --- 取消对该文件的使用
# =================================

import pandas as pd
import numpy as np
import re
from datetime import datetime
import string


pd.set_option('display.max_columns', None)


def get_labels(datas):
    """
    :param datas: datas_all

    function: 获取文本id与标签/是否解决的对应的关系
    """
    get_labels_datas = datas[['id', 'labels', 'solve']].copy()
    get_labels_datas.dropna(subset=['labels'], inplace=True)
    get_labels_datas.dropna(subset=['solve'], inplace=True)
    # 将多标签的labels的数据和单一标签labels数据拆分
    df = get_labels_datas.copy()
    df['get_multiple_labels'] = df['labels'].apply(
        lambda x: 1 if ',' in x else 2
    )

    one_labels_data = df.loc[df['get_multiple_labels'] == 2].copy()

    one_labels_data.drop(columns=['get_multiple_labels'], inplace=True)
    df['get_multiple_labels'] = df['get_multiple_labels'].apply(
        lambda x: np.nan if x == 2 else x
    )
    df.dropna(subset=['get_multiple_labels'], inplace=True)
    df.drop(columns=['get_multiple_labels'], inplace=True)

    # 多标签和多是否解决均已逗号为间隔
    df['labels'] = df['labels'].apply(
        lambda x: x.split(',') if ',' in x else x
    )
    df['solve'] = df['solve'].apply(
        lambda x: x.split(',') if ',' in x else x
    )
    r"""
        将
        id  labels       solve
        123  [a, b, c]    [1, 0, 1]
        拆分成
        id     labels    solve
        123      a         1
        123      b         0
        123      c         1
    """

    df = pd.DataFrame({'id': df.id.repeat(df.labels.str.len()),
                       'labels': np.concatenate(df.labels.values),
                       'solve': np.concatenate(df.solve.values)})

    df = pd.concat([df, one_labels_data])

    df['solve'] = df['solve'].apply(
        lambda x: '-1' if x == '0' else x
    )

    def labels_solve(label, solve):
        if ',' in solve:
            solve = solve.split(',')
            solve = solve[0]
        return [label, int(solve)]

    df['label_solve'] = df.apply(
        lambda x: labels_solve(x['labels'], x['solve']), axis=1
    )
    labels = df.labels.values
    labels = np.unique(",".join(labels).split(','))
    for label in labels:
        df[label] = df['label_solve'].apply(
            lambda x: x[1] if x[0] == label else None
        )

    df.drop(columns=['labels', 'solve', 'label_solve'], inplace=True, axis=1)
    columns = ['价格', '优惠政策', '保险', '内饰', '加装改装', '动力', '外观', '操控', '空间', '竞品对比',
               '经销商试驾', '置换', '蜘蛛智选', '质保', '购车流程', '配置', '金融']
    df = df[columns].groupby(df['id']).sum()
    df.columns = columns
    df.reset_index(inplace=True)
    # 大于等于1的值是存在此标签并且已经解决
    # 小于0的值说明存在多个此标签但是未解决的次数大于解决的次数
    # 等于0的值说明不存在此标签
    for column in columns:
        df[column] = df[column].apply(
            lambda x: 0 if int(x) < 0
            else 1 if int(x) >= 1
            else np.nan
        )
    return df
