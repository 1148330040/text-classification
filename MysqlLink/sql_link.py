# *- coding: utf-8 -*-

# =================================
# time: 2020.9.7
# author: @tangzhilin(Resigned)
# function: sql DB information
# =================================

import pymysql
import pandas as pd
import yaml
# 加载配置文件
from sqlalchemy import create_engine


def link_mysql_db():
    """
    :return: params
    function: 连接参数
    """
    with open('../config/config.yml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    params = {
        'host': config['db_sc'].get('host'),
        'port': 3306,
        'db': config['db_sc'].get('data_base'),

        'user': config['db_sc'].get('user_name'),
        'password': config['db_sc'].get('pass_word'),

        'charset': 'UTF8MB4',
        'cursorclass': pymysql.cursors.DictCursor
    }

    return params


def test():
    engine = create_engine("mysql+pymysql://root:123456@localhost:3306/mysql?charset=utf8")

    datas = pd.read_csv("../../NLPdatas/test_train.csv", nrows=1000,
                        error_bad_lines=False, engine='python')
    datas.columns = ['solve', 'text']
    datas.to_sql(name='nlp_engineer_datas', con=engine, if_exists='replace',
                 index=False, index_label=False, chunksize=50)


def get_sql_train_datas(time_start, time_end):
    """
    :param time_start: 起止时间
    :param time_end: 结束时间

    function: 获取指定时间段的原始数据数据(用于后期更新训练模型)

    :return: sql_train_data
    """

    params = link_mysql_db()

    db = pymysql.connect(**params)
    try:
        with db.cursor() as cursor:
            sql = "select  " \
                  "a.id  聊天框ID , " \
                  "c.id 聊天ID , " \
                  "a.room_name 群名称 , " \
                  "a.unique_customer_wework_id  群内主客户ID , " \
                  "b.role_type  发送角色 , " \
                  "b.from_role_name 发送人, " \
                  "b.from_role_id 发送人ID, " \
                  "case when d.gender is not null then d.gender else e.gender end   '发送人-性别(1男 2女)', " \
                  "f.phone 通过元兵添加验证的客户号码 , " \
                  "c.tolist  接收对象ID , " \
                  "c.msgtype 发送格式, " \
                  "case when f.id is not null then 1 else 0 end 通过元兵添加验证的客户 , " \
                  "c.content  发送内容 , " \
                  "c.created_time 信息发送时间  " \
                  "from  wework_service_db.wework_chat_session a  " \
                  "left join wework_service_db.wework_chat_session_r_data b on  a.id = b.chat_session_id   " \
                  "left join wework_service_db.wework_chat_data c on b.msgid = c.msgid and c.enterprise_corpid =  'wwdbb24b8fe45409c0'   " \
                  "left join wework_service_db.wework_customer d on c.from = d.external_userid  and  b.role_type = 'CUSTOMER' " \
                  "left join wework_service_db.wework_member e on c.from = e.userid and   b.role_type <> 'CUSTOMER' " \
                  "left join call_service_db.wework_add_record f on f.customer_wechat = b.from_role_id and f.status = 'CONFIRMED' " \
                  "where a.corpid = 'wwdbb24b8fe45409c0' " \
                  "group by a.id , c.id  LIMIT 100"
            try:
                cursor.execute(sql)
                result = cursor.fetchall()
                datas = pd.DataFrame(result)
                datas.to_excel("C:\\Users\\tzl17\\Desktop\\第五版_datas.xlsx")
                return datas
            except:
                raise ValueError(
                    "Please check the database for database: nlp_datas"
                )
    finally:
        db.close()


def get_sql_clean_datas(time_start, time_end):
    params = link_mysql_db()

    db = pymysql.connect(**params)
    try:
        with db.cursor() as cursor:
            sql = "SELECT * " \
                  "FROM mysql.nlp_clean_datas;"
            try:
                cursor.execute(sql)
                result = cursor.fetchall()
                datas = pd.DataFrame(result)
                return datas
            except:
                raise ValueError(
                    "Please check the database for database: nlp_clean_datas"
                )
    finally:
        db.close()


def get_sql_fit_datas(time_start, time_end):
    """
    :param time_start: 起止时间
    :param time_end: 结束时间

    function: 获取处理好的带训练数据

    :return: sql_fit_datas
    """
    params = link_mysql_db()

    db = pymysql.connect(**params)
    try:
        with db.cursor() as cursor:
            sql = "SELECT * " \
                  "FROM mysql.nlp_engineer_datas;"
            try:
                cursor.execute(sql)
                result = cursor.fetchall()
                datas = pd.DataFrame(result)
                return datas
            except():
                raise ValueError(
                    "Please check the database for database: nlp_engineer_datas"
                )
    finally:
        db.close()


def get_sql_labels_keywords():
    params = link_mysql_db()

    db = pymysql.connect(**params)
    try:
        with db.cursor() as cursor:
            sql = "SELECT * " \
                  "FROM mysql.nlp_labels_keywords;"
            try:
                cursor.execute(sql)
                result = cursor.fetchall()
                datas = pd.DataFrame(result)
                return datas
            except():
                raise ValueError(
                    "Please check the database for database: nlp_labels_keywords"
                )
    finally:
        db.close()


