import random
import pandas as pd
import numpy as np
import sys
import os
import requests
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta, datetime
import csv
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, BayesianRidge, ElasticNet
import joblib
import matplotlib.pyplot as plt
import json
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from main_RFR import get_today, headers

before_days = 10
code_data_length_limit = 2400

# 爬取数据
def get_data(*args):
    code, start_date, end_date = args[0]
    code_download_path_urls = (
        "http://quotes.money.163.com/service/chddata.html?code=1{code}&start={start}&end={end}&fields={fields}",
        "http://quotes.money.163.com/service/chddata.html?code=0{code}&start={start}&end={end}&fields={fields}"
    )
    fields = 'TCLOSE;HIGH;LOW;TOPEN;LCLOSE;CHG;PCHG;TURNOVER;VOTURNOVER;VATURNOVER;TCAP;MCAP'
    for url_template in code_download_path_urls:
        u = url_template.format(code=code, start=start_date, end=end_date, fields=fields)
        print("爬取：" + code)
        response = requests.get(u, headers=headers).text
        # 转成数组
        text = response.split("\r\n")

        # file = open('data.txt', 'r', encoding='UTF8')
        # response = file.read()
        # text = response.split("\n")

        if len(text) >= code_data_length_limit:
            for i in range(len(text), 3, -1):
                now = text[i - 1 - 1].split(",")[9]
                nextZhangFu = text[i - 1 - 1 - 1].split(",")[9]
                nextHigh = text[i - 1 - 1 - 1].split(",")[3]
                nextLow = text[i - 1 - 1 - 1].split(",")[5]
                if nextZhangFu.lower() == "none" or now.lower() == "none":
                    text[i - 1 - 1] = ""
                    continue

                # 处理日期
                text[i - 1 - 1] = text[i - 1 - 1] + "," + nextZhangFu + "," + nextHigh + "," + nextLow
            # 获取今天的数据
            text[0] = ""
            text[1] = ""

            with open(os.path.join('data_all.csv'), 'a', encoding='utf-8') as f:
                file_str = ""
                for data in text:
                    if data != "":
                        file_str = file_str + data + "\n"
                f.write(file_str)

            print("股票编号：" + code + " 数据爬取成功!\n")
            return True
    return False


if __name__ == '__main__':
    if len(sys.argv) > 1:
        all_code = sys.argv[1:]
        # print(all_code)
    else:
        # 换手
        all_code_url = "http://44.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=2500&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid=f8&fs=m:0+t:6,m:0+t:13,m:0+t:80,m:1+t:2,m:1+t:23&fields=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152&_=1579615221139"
        # 成交金额
        # all_code_url = "http://44.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=500&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid=f6&fs=m:0+t:6,m:0+t:13,m:0+t:80,m:1+t:2,m:1+t:23&fields=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152&_=1579615221139"
        # 涨跌幅
        # all_code_url = "http://44.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=4000&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid=f3&fs=m:0+t:6,m:0+t:13,m:0+t:80,m:1+t:2,m:1+t:23&fields=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152&_=1579615221139"
        # 成交量
        # all_code_url = "http://44.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=1000&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid=f5&fs=m:0+t:6,m:0+t:13,m:0+t:80,m:1+t:2,m:1+t:23&fields=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152&_=1579615221139"
        r = requests.get(all_code_url, timeout=5).json()
        all_code = [data['f12'] for data in r['data']['diff']]
        all_name = [data['f14'] for data in r['data']['diff']]
        print(all_code)
    start_date = (date.today() - timedelta(days=code_data_length_limit * 1.6 + before_days)).strftime("%Y%m%d")
    end_date = (date.today()).strftime("%Y%m%d")
    end_date = (date.today() - timedelta(days=before_days)).strftime("%Y%m%d")

    for code in list(all_code):
        if code.startswith("688") or code.startswith("30"):
            all_code.remove(code)

    random.shuffle(all_code)
    # print(start_date, end_date)

    # 多线程
    with ThreadPoolExecutor(max_workers=20) as tpe:
        tpe.map(get_data, [(code, start_date, end_date) for code in all_code])

    # for code in all_code:
    #     get_data([code, start_date, end_date, data_path, predict_path, grid_search])
