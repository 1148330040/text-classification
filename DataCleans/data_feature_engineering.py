# *- coding: utf-8 -*-

# =================================
# time: 2020.7.28
# author: @tangzhilin(Resigned)
# function: 数据特征工程
# update: 8.18日 更改了数据的连接方式
# update: 9.08日 依照新的标签数据标准更改了代码
# =================================

import pandas as pd
import numpy as np
import re
import jieba
import jieba.analyse
import warnings
warnings.filterwarnings('ignore')


def feature_engineering(datas, fit):
    """
    :param fit: 处理的是训练数据的话fit=True 默认为False
    :param datas: all_datas

    function: 针对数据进行特征工程

    """

    # 文本缩减
    datas.drop_duplicates(subset=['id', 'text'], keep='first', inplace=True)

    pattern = "\s*[\u4e00-\u9fa5]{1,4}[哥|姐|先生|总|老板]\s{1}"
    datas['text'] = datas['text'].apply(
        lambda x: re.sub(pattern, '', x.strip())
    )

    datas['text'] = datas['text'].apply(
        lambda x: np.nan if x == ''
        else x
    )

    datas.dropna(subset=['text'], inplace=True)

    # 使用jieba对文本进行分词处理
    # 对没有对应标签的文本, 进行删减处理
    jieba.load_userdict("../config/jieba_thesaurus.txt")

    # jieba 分词
    def remove_space(text):
        text = np.array(list(jieba.cut(text)))
        return text

    datas['text'] = datas['text'].apply(
        lambda x: remove_space(x)
    )

    datas['count_positive_word'] = datas['text'].apply(
        lambda x: list(x).count('好的') + list(x).count('是的') + list(x).count('对的') + list(x).count('没问题')
    )

    # 关键词统计
    datas['text'] = datas['text'].apply(
        lambda x: "".join(x) + " "
    )

    id_count_words = datas.groupby(['id']).agg({
        'count_positive_word': ['sum']
        # 以文本id为维度, 获取文本中出了几次 正向词
    })

    id_count_words.columns = ['count_pos_word']
    id_count_words.reset_index(inplace=True)
    id_count_words = pd.DataFrame(id_count_words)
    datas = pd.merge(datas, id_count_words[['id', 'count_pos_word']],
                     on=['id'], how='left')

    # 连接 针对原始训练文本处理
    # datas['labels'].fillna("remove", inplace=True)

    if fit:
        datas['solve'].fillna(-1, inplace=True)
        datas_labels = datas.dropna(subset=['labels'])
        datas_labels['index'] = datas_labels.index.values
        datas_labels = datas_labels[['id', 'chat_id', 'labels', 'index', 'count_pos_word']]

        def link_text(indexs):
            start_index = -1.0
            end_index = -1.0
            solve = -1.0
            index = indexs

            while start_index == -1.0:
                if datas.iloc[index].solve != -1.0:
                    solve = datas.iloc[index].solve
                    start_index = index
                index = index - 1

            index = indexs + 1
            while end_index == -1.0:
                if datas.iloc[index].solve != -1.0:
                    end_index = index
                index = index + 1

            text = ''

            for i in range(start_index, end_index + 1):
                text = text + " " + datas.iloc[i].text

            return text + str(solve)

        datas_labels['text'] = datas_labels['index'].apply(
            lambda x: link_text(x)
        )

    else:
        error_labels_index = list(datas.dropna(subset=['labels']).index.values)
        labels_keywords = pd.read_excel('../test_data/label_keywords.xlsx')
        mark_words = ['没问题', '不客气', '仅限', '有的', '是的', '好的', '可以',
                      '不了', '不到', '谢谢', '对的', '没空', '不错', '没车',
                      '到店', '没呢', '清楚', '明白', '确认', '没法', '不到',
                      '了解', '都是', '还没', '比较', '地址', '不多', '没有',
                      '放心', '嗯', '恩', '行', '没', '有']

        # 针对标签制定逻辑(错标)

        def bad_mark(index):
            # 针对错误标记
            chart_id = datas.loc[index]['id']
            labels = datas.loc[index]['labels']
            if type(labels) != str:
                # 会出现NAN的原因是下面的逻辑设置
                return
            # 如果下文聊天框ID不同则删除该标签
            try:
                if str(chart_id) != str(datas.loc[index + 5]['id']):
                    datas.loc[datas.index == index, 'labels'] = np.NaN
                    return
            except:
                return

            # 如果标签对应文本长度过长则删除该标签
            if len(datas.loc[index]['text']) > 26:
                datas.loc[datas.index == index, 'labels'] = np.NaN
                return

            # 若连续出现标签，则放弃连续出现的标签
            if (index + 1) in error_labels_index:
                datas.loc[datas.index == index, 'labels'] = np.NaN
                datas.loc[datas.index == index + 1, 'labels'] = np.NaN
                return

            # 如果文本对应的标签数目大于三个以上则放弃该标签
            if len(datas.loc[index]['labels'].split(',')) > 3:
                datas.loc[datas.index == index, 'labels'] = np.NaN
                return

            # 三个标签文本则保留下文中有对应关键词出现的文本
            if len(datas.loc[index]['labels'].split(',')) >= 2:
                labels = labels.split(',')
                # 将下文四行文本连接
                text = datas.loc[index:(index + 4), 'text'].values.sum()
                new_labels = []
                for label in labels:
                    if label == '蜘蛛智选':
                        continue
                    label_keywords = labels_keywords.loc[labels_keywords['tag_lv2'] == label, '关键词'].values
                    for word in label_keywords:
                        if word in text:
                            new_labels.append(label)
                            break
                new_labels = ','.join(i for i in new_labels)

                datas.loc[datas.index == index, 'labels'] = new_labels

                return

            # 如果标签下文中出现了标签对应关键词则保留该标签
            label_keywords = labels_keywords.loc[labels_keywords['tag_lv2'] == labels, '关键词'].values

            for i in range(1, 6):
                text = datas.loc[index + i]['text']
                lk_num = 0
                mk_num = 0
                if type(text) != str:
                    continue
                if lk_num == 0:
                    for word in label_keywords:
                        if word in text:
                            lk_num = 1
                if mk_num == 0:
                    for word in mark_words:
                        if word in text:
                            mk_num = 1
                if lk_num + mk_num == 2:
                    return

            datas.loc[datas.index == index, 'labels'] = np.NaN

            return

        for index in error_labels_index:
            bad_mark(index)

        def omit_mark():
            # 针对遗漏标记做处理
            # 去除文本中的标签文本(包含标签对应的下文五行内容)
            omit_label_index = list(datas.dropna(subset=['labels'])['labels'].index.values)
            drop_indexs = [i + j for i in omit_label_index for j in range(5)]

            drop_indexs = sorted(list(set(drop_indexs)))
            drop_datas = datas.drop(index=drop_indexs[:-5])
            drop_ids = drop_datas['id'].unique()
            groups_datas = pd.DataFrame({})

            def add_labels(texts):

                if len(texts) > 50 or len(texts) <= 4:
                    return np.NaN

                for lk_indexs, word in enumerate(labels_keywords['关键词']):
                    # index的作用是找到word对应的label也就是tag_lv2
                    if word in str(texts):
                        labels = labels_keywords.loc[lk_indexs]['tag_lv2']
                        return labels

                return np.NaN

            def drop_labels(labels, drop_index):
                # labels为空直接返回无需操作
                if type(labels) != str:
                    return np.NaN

                # 确保不要出现连续的标签以第一个出现标签的文本为准
                try:
                    if drop_index > 1 and type(group_datas.loc[drop_index - 1, 'labels']) == str:
                        return np.NaN
                except:
                    return labels
                role = group_datas.loc[drop_index]['role']

                # role为MEMBER且label不为空
                # 则判断后续文本行的role是否还是MEMBER
                try:
                    # 这里会报错的原因是drop_index是最后一个
                    # 加1的话则超出了group_datas的界限
                    if role != 'CUSTOMER':
                        if group_datas.loc[drop_index + 1]['role'] != 'CUSTOMER':
                            return np.NaN
                    else:
                        role_num = 1
                        for i in range(1, 3):
                            if group_datas.loc[drop_index + i]['role'] == role:
                                role_num += 1
                        # 如果标签文本对应的role连续三行都一致则返回NaN
                        if role_num == 3:
                            return np.NaN
                except:
                    return np.NaN

                return labels

            # 将聊天框ID为一组数据进行处理
            # 首先解决的是为包含关键词的 行文本 添加对应关键词标签
            # 其次解决的是有标签的行文本是否值得保留标签
            for drop_id in drop_ids:
                group_datas = drop_datas.loc[drop_datas['id'] == drop_id]
                group_datas['index'] = group_datas.index.values
                # 初选label
                group_datas['labels'] = group_datas.apply(
                    lambda x: add_labels(x['text']), axis=1
                )

                group_datas['drop_labels'] = group_datas['labels']
                # 现将group_datas.index按照长度设置之后需要将他的index变为原样

                group_datas['labels'] = group_datas.apply(
                    lambda x: drop_labels(x['drop_labels'], x['index']), axis=1
                )

                group_datas.drop(['drop_labels'], axis=1, inplace=True)

                # 其次下文三行内没有相关关键词和结束词的不要
                drop_index_3 = list(group_datas.dropna(subset=['labels']).index.values)
                for d_index3 in drop_index_3:
                    mk_num = 0
                    lk_num = 0
                    label = group_datas.loc[d_index3]['labels']
                    label_keywords = labels_keywords.loc[labels_keywords['tag_lv2'] == label]['关键词'].values
                    try:
                        for d_num in range(1, 4):
                            text = str(group_datas.loc[d_index3 + d_num]['text'])
                            if lk_num <= 1:
                                for lk in label_keywords:
                                    if lk in text:
                                        lk_num += 1
                            if mk_num <= 1:
                                for mw in mark_words:
                                    if mw in text:
                                        mk_num += 1
                            if (lk_num + mk_num) == 2:
                                break
                        if (lk_num + mk_num) != 2:
                            group_datas.loc[group_datas.index == d_index3, 'labels'] = np.NaN
                    except:

                        group_datas.loc[group_datas.index == d_index3, 'labels'] = np.NaN
                # 将以聊天框ID为维度的数据新添加的标签合并到数据中
                group_datas = group_datas[['chat_id', 'labels']]
                group_datas.columns = ['chat_id', 'new_labels']
                group_datas.dropna(subset=['new_labels'], inplace=True)
                groups_datas = pd.concat([groups_datas, group_datas], sort=True)

            return groups_datas

        merge_datas = omit_mark()
        datas = pd.merge(datas, merge_datas, on=['chat_id'], how='left')
        datas.to_excel('C:\\Users\\tzl17\\Desktop\\show.xlsx')

        def choose_labels(labels, new_labels):
            if type(labels) == str:
                return labels
            if type(new_labels) == str:
                return new_labels
            return np.NaN

        datas['labels'] = datas.apply(
            lambda x: choose_labels(x['labels'], x['new_labels']), axis=1
        )

        datas.drop('new_labels', axis=1, inplace=True)

        datas_labels = datas.dropna(subset=['labels'])
        datas_labels['index'] = datas_labels.index.values
        datas_labels = datas_labels[['id', 'chat_id', 'labels', 'index', 'count_pos_word']]

        def link_text(label_index):
            mark_num = 1
            text = str(datas.loc[label_index]['text'])
            chat_id = datas.loc[label_index]['id']
            solve = str(datas.loc[label_index]['solve'])
            index = label_index
            while mark_num <= 5:
                # 限制长度, 最硬性的标准如果超过了则直接反回
                if len(text) > 130:
                    text = text[:126]
                    return text + str(solve)
                index = index + 1
                # 如果标签对应文本的下行文本不属于同一个id则直接返回None
                # 如果不是下一行则返回当前连接的text
                if datas.loc[index]['id'] != chat_id:
                    if index == label_index + 1:
                        return None
                    else:
                        return text + solve
                text_pro = datas.loc[index]['text']
                try:
                    text_pro2 = datas.loc[index + 1]['text']
                    # 判断文本中是否出现mark_words做出相应的处理

                    text = text + ' ' + text_pro
                    for mark_word in mark_words:
                        if mark_word in text_pro:
                            # 如果文本中出现了mark_words则直接连接到后面
                            mark_word2_num = 0
                            for mark_word2 in mark_words:
                                if mark_word2 in text_pro2:
                                    mark_word2_num = 1
                                    break
                            if mark_word2_num == 0:
                                return text + solve
                            else:
                                break
                    mark_num = mark_num + 1
                except:
                    return None

            return text + solve

        datas_labels['text'] = datas_labels['index'].apply(
            lambda x: link_text(x)
        )
    # 由于部分文本的solve并不是对应的标签所处的列因此需要特殊处理
    # solve所处的位置为text的最后三位
    datas_labels.dropna(subset=['text'], inplace=True)

    datas_labels['solve'] = datas_labels['text'].apply(
        lambda x: x[-3:]
    )
    datas_labels['text'] = datas_labels['text'].apply(
        lambda x: x[:-3]
    )
    # 拆分, 分配, 连接(多标签)
    # 找到并将多标签文本单独拎出来
    more_labels_df = datas_labels.loc[datas_labels['labels'].str.contains(','), :]
    # 剔除多标签文本保留单标签文本
    datas_labels.loc[datas_labels['labels'].str.contains(','), 'text'] = np.nan
    datas_labels.dropna(subset=['text'], inplace=True)

    more_labels_df['labels'] = more_labels_df['labels'].apply(
        lambda x: x.split(',') if ',' in x
        else x
    )
    df = pd.DataFrame({'text': more_labels_df.text.repeat(more_labels_df.labels.str.len()),
                       'count_pos_word': more_labels_df.count_pos_word.repeat(more_labels_df.labels.str.len()),
                       'id': more_labels_df.solve.repeat(more_labels_df.labels.str.len()),
                       'chat_id': more_labels_df.chat_id.repeat(more_labels_df.labels.str.len()),
                       'solve': more_labels_df.solve.repeat(more_labels_df.labels.str.len()),
                       'labels': np.concatenate(more_labels_df.labels.values)})

    df_one_label = datas_labels
    # 标签只有一个的文本

    df_more_labels = df
    # 标签有多个的文本

    def shuffle_text(text):
        # 将对应多个标签的文本内容打乱顺序
        text = text.split(' ')
        index = np.arange(len(text))

        np.random.shuffle(index)
        text = "".join(text[i] for i in index).strip()

        return text

    df_more_labels['text'] = df_more_labels.apply(
        lambda x: shuffle_text(x['text']), axis=1
    )
    df = pd.concat([df_one_label, df_more_labels], ignore_index=True)
    datas_labels = pd.concat([datas_labels, df], ignore_index=True)

    datas_labels.drop_duplicates(subset=['text', 'labels'], keep='first', inplace=True)
    datas_labels.drop(columns='index', inplace=True)

    datas_labels = datas_labels.sample(frac=1.0)

    datas_labels['count_pos_word'].fillna(0, inplace=True)

    def end_link(text, labels, pos_count):

        text = text + ' ' + str(labels) + ' ' + str(pos_count)

        return text

    datas_labels['text'] = datas_labels.apply(
        lambda x: end_link(x['text'], x['labels'], x['count_pos_word']), axis=1)

    return datas_labels


def engineer_datas(datas, fit=False):
    datas = feature_engineering(datas, fit)
    datas['text_size'] = datas['text'].apply(
        lambda x: np.nan if len(x) > 130 else len(x)
    )
    datas.dropna(subset=['text_size'], inplace=True)
    return datas



