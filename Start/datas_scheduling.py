# *- coding: utf-8 -*-

# =================================
# time: 2020.9.11
# author: @tangzhilin(Resigned)
# function: 由该文件将数据库内的原始数据获取, 并控制数据预清洗和特征工程
# =================================

from sqlalchemy import create_engine

from DataCleans.data_pre_clean import clean_datas
from DataCleans.data_feature_engineering import engineer_datas

from MysqlLink.sql_link import get_sql_train_datas, get_sql_clean_datas


def datas_clean():
    datas = get_sql_train_datas(None, None)

    datas = clean_datas(datas=datas)

    try:
        engine = create_engine("mysql+pymysql://root:123456@localhost:3306/mysql?charset=utf8")
        datas.to_sql(name='nlp_clean_datas', con=engine, if_exists='replace',
                     index=False, index_label=False, chunksize=5000)
    except:
        raise ValueError(
            "clean datas error"
        )


def datas_engineer():
    datas = get_sql_clean_datas(None, None)
    datas = engineer_datas(datas)
    try:
        engine = create_engine("mysql+pymysql://root:123456@localhost:3306/mysql?charset=utf8")
        datas.to_sql(name='nlp_engineer_datas', con=engine, if_exists='replace',
                     index=False, index_label=False, chunksize=50)
    except:
        raise ValueError(
            "engineer datas error"
        )

datas_clean()
datas_engineer()
