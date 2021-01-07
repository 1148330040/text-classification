# *- coding: utf-8 -*-

# =================================
# time: 2020.7.21
# author: @XXX
# function: 数据初步清洗
# update: 9.08日 依照新的数据标注标准修改了部分代码
# =================================

import pandas as pd
import numpy as np
import re

pd.set_option('display.max_columns', None)


def pre_clean_all_datas(datas):
    """
    :param datas: datas_all(包含三个发送角色)

    function: 针对text, 删除空文本和无效文本, 处理部分特殊文本

    variable:
        length_threshold_text: 文本对话长度是否过短的限定阈值, 小于该值则删除这一组文本
        length_threshold_text_count: 一组文本对话中customer发送的文本数目限定阈值, 小于该值则删除这一组文本

    constant:
        pattern: 正则表达式匹配模式(一个函数内统一用此命名匹配模式)
        special_id:  不相关的文本ID(10万行数据)
        china_punc: 中文标点\特殊符号
        english_punc: 英文标点\特殊符号

    """
    try:
        datas['sender_sex'] = datas['sender_sex'].apply(
            lambda x: 1 if x == 2.0 else 0
        )
    except:
        pass

    # 属于内部交流的聊天框ID
    try:
        special_id = [1471, 1872, 1905, 1108, 590, 719, 1965, 2127, 2352, 2371, 2381, 2447, 1484]
        datas.drop(index=special_id, inplace=True)
    except:
        pass

    # 文本中会出现一些不规范的空格' ', 需要用常用空格将其替代
    # 否则后续的符号处理, 以及部分正则表达式设置都会有问题

    datas['text'] = datas['text'].apply(
        lambda x: re.sub("[' ']", ' ', x)
    )
    datas['text'] = datas['text'].apply(
        lambda x: re.sub("\[OK\]", '好的', x)
    )
    datas['text'] = datas['text'].apply(
        lambda x: re.sub("\[抱拳\]", '谢谢', x)
    )
    datas['text'] = datas['text'].apply(
        lambda x: re.sub("\[强\]", '好的', x)
    )
    datas['text'] = datas['text'].apply(
        lambda x: re.sub("ok", '好的', x)
    )
    # 去除文本中 @xxx 的内容
    pattern = "@[\u4e00-\u9fa5]*\w*\s*\w*\s{0,1}\w{0,1}\s"
    datas['text'] = datas['text'].apply(
        lambda x: np.nan if len(re.sub(pattern, '', x)) < 1
        else re.sub(pattern, '', x)
    )
    datas.dropna(subset=['text'], inplace=True)

    # 去除文本中 pro xxxxx 的 内容
    pattern = "Pro[\u4e00-\u9fa5]+\s{1}[\u4e00-\u9fa5]+\s*"
    datas['text'] = datas['text'].apply(
        lambda x: np.nan if len(re.sub(pattern, '', x)) < 1
        else re.sub(pattern, '', x)
    )
    datas.dropna(subset=['text'], inplace=True)

    # 部分文本内容只有一些数字('1', '2', '452'), 清除
    datas['text'] = datas['text'].apply(
        lambda x: np.nan if x.isdigit()
        else x
    )
    datas.dropna(subset=['text'], inplace=True)

    # 部分文本内容只有表情符号的([握手], [抱拳]), 清除
    pattern = "\\[[\u4e00-\u9fa5\\]]+\\]"
    datas['text'] = datas['text'].apply(
        lambda x: np.nan if len(re.sub(pattern, '', x)) < 1
        else re.sub(pattern, '', x)
    )
    datas.dropna(subset=['text'], inplace=True)

    pattern = "https?://\S+|www\.\s+"
    datas['text'] = datas['text'].apply(
        lambda x: np.nan if len(re.sub(pattern, '', x)) < 1
        else re.sub(pattern, '', x)
    )
    datas.dropna(subset=['text'], inplace=True)

    # 去除中文标点符号和特殊符号
    china_punc = "！？。?？＂＃＄％＆＇（）＊＋，－／：；℡₀₁₂₃₄₅₆₇₈₉คิดถึงคน＜＝＞＠" \
                 "［＼］＾＿｛｜｝～、〃》「」『』〔〕〝〞–—‘'“”…º¹²³⁴⁵⁶⁷⁸⁹Ππ�"
    datas['text'] = datas['text'].apply(
        lambda x: np.nan if len(re.sub("[%s]+" % re.escape(china_punc), '', str(x))) < 1
        else re.sub("[%s]+" % re.escape(china_punc), ' ', str(x))
    )
    datas.dropna(subset=['text'], inplace=True)

    # 去除英文标点符号
    english_punc = "!#$%&'()*,/:;<=>?@[\]^_`{|}~-"
    datas['text'] = datas['text'].apply(
        lambda x: np.nan if len(re.sub("[%s]+" % re.escape(english_punc), '', str(x))) < 1
        else re.sub("[%s]+" % re.escape(english_punc), ' ', str(x))
    )
    datas.dropna(subset=['text'], inplace=True)

    # 去除D90 Pro VIP订单专员|官方顾问|官方助理
    pattern = "D90 Pro\s{0,1}(VIP订单专员|官方顾问|官方助理)"
    datas['text'] = datas['text'].apply(
        lambda x: re.sub(pattern, '', x)
    )
    datas.dropna(subset=['text'], inplace=True)

    # 去除文本中的电话号码
    pattern = '[\d]{3}[\s|-]{0,1}[\d]{3,4}[\s|-]{0,1}[\d]{4,5}\s{0,1}'
    datas['text'] = datas['text'].apply(
        lambda x: re.sub(pattern, '', x)
    )
    datas.dropna(subset=['text'], inplace=True)

    # 将长空格"    "变为一个长度的空格 " "
    pattern = "[\s]+"
    datas['text'] = datas['text'].apply(
        lambda x: np.nan if x.isspace()
        else re.sub(pattern, ' ', x)
    )
    datas.dropna(subset=['text'], inplace=True)

    return datas


def pre_clean_datas_customer(datas):
    """
    :param datas: datas_customer

    function: 针对datas_customer 做出的一些相关数据处理

    :def labels_solve(): 用于将文本相关的标签与对话id对应的处理函数

    constant:
        special_mark: name带有该关键词标记的说明其实际身份为MEMBER
        special_customer: name为该内容的说明其实际身份为MEMBER
        pattern: 正则表达式匹配模式(一个函数内统一用此命名匹配模式)
    """
    special_mark = ['大通', '上汽', 'CV', 'DV', 'D90']
    special_customer = ['宋美林', '黄铭', 'Ponder Lu', 'lechen',
                        'kkk', '木易M', 'GiNo', '昕芸', '方也十兑酋犬',
                        '朱文勋', '魅华']
    r'''
        : 聊天框ID: 2447 是元兵和商家内部人员交流的文本.
        里面所涉及的name应该为MEMBER或删除相关文本.
        yb_name =
            ['方也十兑酋犬', 'D就是这么调', '张倩雯', '吴洋', '拾', '木易M',
            'Fantasy卿', '昕芸', '施悦猷', 'paranoidkk', '刘强', 'chenyang',
            '陈晓翔', '徐超', '石慧', '叶晓春', 'Zz', '张博伦', '用户大当家',
            '小丁', '没空', '浪迹天涯', '晓翔']
    '''
    sub_marks = special_mark + special_customer

    # 显示角色为customer实则为manager
    datas['name'].fillna('just_moment', inplace=True)
    for mark in sub_marks:
        datas.loc[datas['name'].str.contains(mark), 'role'] = 'MEMBER'

    r'''
        关于'xxx 小乔', 部分名称是 '@D90 Pro官方助理-小乔'
        还有部分名称是 '@小乔 D90 Pro官方助理'
        因此需要在pre_clean_all_datas()后面在做一次处理
    '''
    pattern = 'Pro官方助理 小乔'
    datas['text'] = datas['text'].apply(
        lambda x: np.nan if len(re.sub(pattern, '', x)) < 1
        else re.sub(pattern, '', x)
    )
    datas.dropna(subset=['text'], inplace=True)

    pattern = '\w*官方助理\s*'
    datas['text'] = datas['text'].apply(
        lambda x: np.nan if len(re.sub(pattern, '', x)) < 1
        else re.sub(pattern, '', x)
    )
    datas.dropna(subset=['text'], inplace=True)
    r'''
        发送角色为customer的, 如果text带有'【', '】'那么通常指向
        活动链接\银行信息\上汽财务相关信息(贷款成功与否)
        通常活动链接包含的内容有'好消息'和'礼'
        考虑到很少一部分用户会说'礼拜天' 因此需要额外加一个逻辑

        ps:(在针对DataFrame内的逻辑运算时, 不能直接使用and和or)
    '''
    datas.loc[datas['text'].str.contains('【') &
              datas['text'].str.contains('】'),
              'text'] = '反馈'

    member_names = ['官方顾问', '陈阳', '昕芸', '孙云赫', '贾明野', '傅嘉卿',
                    '吴赛', '小傅', '薛金刚', '安继平', '小刘', '邓斌', '程林枫']
    for m_name in member_names:
        datas['text'] = datas['text'].apply(
            lambda x: x.replace(m_name, '')
            if m_name in x else x
        )

    return datas


def pre_clean_datas_member(datas):
    """
    :param datas: datas_member

    function: 针对datas_member 做出的一些相关数据处理

    constant:
        start_words: 用于标记开头文本
        activate_names: 用于标记九重礼活动内容文本
        special_words: 用于标记相关活动内容文本
    """

    datas.loc[datas['role'] == 'CC', 'role'] = 'MEMBER'

    # 开头的话不仅长且无用
    start_words = ['非常感谢您关注', '专属客户', '合法权益', '专属顾问',
                   '咨询', '工程师', '勇征荒野']
    for s_word in start_words:
        datas.loc[datas['text'].str.contains(s_word), 'text'] = ''

    datas.dropna(subset=['text'], inplace=True)

    activate_names = ['上市礼', '严选礼', '金融礼', '置换礼', '重礼',
                      '交车礼', '质保礼', '畅玩礼', '无忧礼', '流量礼', '豪礼']
    for a_name in activate_names:
        datas.loc[datas['text'].str.contains(a_name), 'text'] = '优惠政策'

    datas.loc[datas['text'].str.contains('【') &
              datas['text'].str.contains('】'), 'text'] = '活动'

    return datas


def pre_replace_keyword(datas):
    """
    :param datas

    function: 将text文本中出现的一些关键词替换(4s->经销商, d90,h9->汽车, xx万,xx元->价格等)

    constant:
        pattern: 正则表达式匹配模式(一个函数内统一用此命名匹配模式)
    """
    # 将 '90pro, d90pro, d90 pro, D90pro, D90 pro'车型替换为在售车
    pattern = "(D|d)*90\s{0,1}([p|P]ro)*"
    datas['text'] = datas['text'].apply(
        lambda x: re.sub(pattern, '在售车', x)
    )

    # 将 'h9, q5, Cs95, R8' 替换为竞品车
    pattern = "[a-zA-Z]{1,2}[\d]{1,2}"
    datas['text'] = datas['text'].apply(
        lambda x: re.sub(pattern, '竞品车', x)
    )

    # 将 '4000元, 3000, 10万, 5万, 196000'替换为价格
    pattern = "[\d]{1,7}[元|万]+"
    datas['text'] = datas['text'].apply(
        lambda x: re.sub(pattern, '价格', x)
    )

    # 为了区别年份(2017, 2019)因此还需要对类似于(1000, 2000, 3000, 4000, 190000等)再做一次处理
    pattern = "[\d]{5,8}|[3-9][\d]{3}"
    datas['text'] = datas['text'].apply(
        lambda x: re.sub(pattern, '价格', x)
    )

    # 针对 (19.98, 19.8)类似的价格 再做一次处理
    pattern = "[\d]{2}\\.+[\d]*"
    datas['text'] = datas['text'].apply(
        lambda x: re.sub(pattern, '价格', x)
    )

    pattern = "1000|2000"
    datas['text'] = datas['text'].apply(
        lambda x: re.sub(pattern, '价格', x)
    )

    # 将 'x+x+x, x-x-x' 替换为座位外观
    pattern = "[\d]{1}[\\+|\\-][\d]{1}[\\+|\\-][\d]{1}"
    datas['text'] = datas['text'].apply(
        lambda x: re.sub(pattern, '座位外观', x)
    )

    # 将 'xT, xL, x升' 替换为动力
    pattern = "[\d]{1}\\.[\d]*[升|L|T]"
    datas['text'] = datas['text'].apply(
        lambda x: re.sub(pattern, '动力', x)
    )

    # 将 'x吨, x宽, x米' 替换为配置
    pattern = "[\d]{1}\\.[\d]*[吨|宽|m|米]"
    datas['text'] = datas['text'].apply(
        lambda x: re.sub(pattern, '配置', x)
    )

    # 将 'xx寸' 替换为外观
    pattern = "[\d]{1,2}\s*[寸]"
    datas['text'] = datas['text'].apply(
        lambda x: re.sub(pattern, '外观', x)
    )

    # 将(AT和HT两种轮胎型号)替换为外观
    pattern = "[H|h|A|a][T|t]"
    datas['text'] = datas['text'].apply(
        lambda x: re.sub(pattern, '外观', x)
    )

    # 将余下数字都替换为空
    # datas['text'] = datas['text'].apply(
    #     lambda x: re.sub("\d", '', x)
    # )

    # 将'.'(因为要处理2.0t, x.x高等数据, 因此保留了.)替换为空
    datas['text'] = datas['text'].apply(
        lambda x: re.sub("\\.", '', x)
    )

    # 将(年, 月, 日)替换为空
    datas['text'] = datas['text'].apply(
        lambda x: re.sub("年月日", '', x)
    )

    datas.loc[datas['text'].str.contains('地址'), 'text'] = "试驾点"

    datas.loc[datas['text'].str.contains('车型'), 'text'] = "试驾点"

    datas.loc[datas['text'].str.contains('重磅预定特惠'), 'text'] = "新车发布"

    datas.loc[datas['text'].str.contains('置换是只能是国四及以下的'), 'text'] = '置换政策说明'

    datas.loc[datas['text'].str.contains('落实您试乘试驾的服务'), 'text'] = '试驾安排'

    datas.loc[datas['text'].str.contains('上面会直接显示价格哒'), 'text'] = '定制车型价格'

    datas.loc[datas['text'].str.contains('置换材料'), 'text'] = '置换材料'

    datas.loc[datas['text'].str.contains('GPS'), 'text'] = '两成首付政策'

    datas.loc[datas['text'].str.contains('全地形智能四驱'), 'text'] = '操控相关介绍'

    datas.loc[datas['text'].str.contains('已购买不计免赔'), 'text'] = '保险相关'

    datas.loc[datas['text'].str.contains('贷款购车'), 'text'] = '贷款介绍'

    datas.loc[datas['text'].str.contains('免息贷款'), 'text'] = '贷款优惠政策'

    datas.loc[datas['text'].str.contains('金融手续费'), 'text'] = '贷款优惠政策'

    datas.loc[datas['text'].str.contains('定制您专属的车型'), 'text'] = '车型定制'

    datas.loc[datas['text'].str.contains('终身质保赠送'), 'text'] = '质保赠送'

    datas.loc[datas['text'].str.contains('置换规则') | datas['text'].str.contains('置换范围'), 'text'] = '置换规则和流程'

    datas.loc[datas['text'].str.contains('在线签署'), 'text'] = '购车流程订单确认'

    datas.loc[datas['text'].str.contains('这位是我们的'), 'text'] = '试驾金融相关介绍'

    datas.loc[datas['text'].str.contains('国四及以下'), 'text'] = '置换政策'

    datas.loc[datas['text'].str.contains('初审通过'), 'text'] = '贷款审核'

    datas.loc[datas['text'].str.contains('三种驱动模式'), 'text'] = '分时四驱和智能四驱的差别介绍'

    datas.loc[datas['text'].str.contains('柴油发动机对比'), 'text'] = '柴油发动机介绍'

    datas.loc[datas['text'].str.contains('现在订车'), 'text'] = '购买流程'

    datas.loc[datas['text'].str.contains('十月十日'), 'text'] = '介绍词'

    return datas


def pre_clean_all_datas_end(datas):
    """
    :param datas: datas_all

    function: 数据清洗结束后的完善与整理
    """

    # 将所有的字母去除
    pattern = '[A-Za-z]+'
    datas['text'] = datas['text'].apply(
        lambda x: re.sub(pattern, '', x)
    )
    datas.dropna(subset=['text'], inplace=True)

    # 将特殊空格去除
    special_space = ['\u202c', '\u202d', '\ue41d', '\ue412']
    for s_space in special_space:
        datas['text'] = datas['text'].apply(
            lambda x: '' if s_space in x
            else x
        )

    # 将''变为np.nan
    datas['text'] = datas['text'].apply(
        lambda x: np.nan if x == ''
        else x
    )

    datas.dropna(subset=['text'], inplace=True)

    return datas


def clean_datas(datas):

    """
    function: 传递预处理后的数据
    """
    print(datas.columns)
    try:
        datas = datas[['聊天框ID', '聊天ID', '发送角色', '发送人',
                       '发送内容', '标签', '是否解决问题']]
    except:
        datas['是否解决问题'] = None
        datas = datas[['聊天框ID', '聊天ID', '发送角色', '发送人',
                       '发送内容', '标签', '是否解决问题']]

    datas.rename(columns={'聊天框ID': 'id', '聊天ID': 'chat_id',
                          '发送角色': 'role', '发送内容': 'text',
                          '标签': 'labels', '是否解决问题': 'solve',
                          '发送人': 'name'}, inplace=True)

    datas['id'] = datas['id'].astype(np.int64)
    datas['chat_id'] = datas['chat_id'].astype(np.int64)
    datas['role'] = datas['role'].astype(np.str)
    datas['name'] = datas['name'].astype(np.str)
    datas['text'] = datas['text'].astype(np.str)
    datas['solve'] = datas['solve'].apply(
        lambda x: np.nan if str(x).isspace() else x
    )
    datas['solve'] = datas['solve'].astype(np.float)

    # role为空的数据中多数均为CUSTOMER(90%)少部分为MEMBER
    # 这一部分很难通过代码直接判断因此将其统一填充为CUSTOMER
    datas['role'].fillna(value='CUSTOMER', inplace=True)

    datas.loc[datas['role'] != 'CUSTOMER', 'role'] = 'MEMBER'

    datas = pre_clean_all_datas(datas)

    datas_customer = datas.loc[datas['role'] == 'CUSTOMER']

    datas_member = datas.loc[datas['role'] == 'MEMBER']

    # 分批次处理可以有效提高效率
    datas_customer = pre_clean_datas_customer(datas_customer)

    datas_member = pre_clean_datas_member(datas_member)

    datas = pd.concat([datas_customer, datas_member], sort=True)

    datas.sort_index(inplace=True)

    datas = pre_replace_keyword(datas)
    datas = pre_clean_all_datas_end(datas)
    datas.drop(columns=['name'], inplace=True)

    return datas

