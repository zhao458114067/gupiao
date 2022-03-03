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

is_my_code = False
# 前几天的数据，用于测试，计算明天是0
before_days = 0
code_data_length_limit = 20


# 训练生成模型以及预测
def download_code_data(*args):
    code, start_date, end_date, data_path, predict_path, grid_search = args[0]
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

            today_data, price, today_huan = get_today(code)
            if float(price) > 60 or float(today_huan) < 1:
                continue

            with open(os.path.join(data_path, '{code}.csv'.format(code=code)), 'w', encoding='utf-8') as f:
                file_str = ""
                for data in text:
                    if data != "":
                        file_str = file_str + data + "\n"
                f.write(file_str)

            print("股票编号：" + code + " 数据爬取成功!\n")
            train_and_predict(code, today_data, grid_search)
            return True
    return False


def get_score_result(x, y, today_data, grid_search):
    if os.path.exists("model\\model.pkl"):
        # 预测明天的情况
        x_transfer = joblib.load("model\\x_transfer")
        y_transfer = joblib.load("model\\y_transfer")
        score = 0
        # 预测明天的情况
        today_data = x_transfer.transform(today_data)
        t_predict = grid_search.predict(today_data)
        t_predict = y_transfer.inverse_transform(DataFrame(t_predict))
    else:
        x_transfer = MinMaxScaler()
        y_transfer = MinMaxScaler()
        # 划分训练集
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
        # 特征工程，标准化
        x_train = x_transfer.fit_transform(x_train)
        y_train = y_transfer.fit_transform(DataFrame(y_train))
        x_test = x_transfer.transform(x_test)
        y_test = y_transfer.transform(DataFrame(y_test))
        grid_search = RandomForestRegressor(bootstrap=True, n_estimators=185, max_features=2,
                                            random_state=42, n_jobs=-1)

        # grid_search = GradientBoostingRegressor(random_state=42)
        param_grid = [
            # 12 (3×4) 种超参数组合
            {'learning_rate': [0.05], 'n_estimators': [100], 'max_depth': [1],
             'min_samples_leaf': [27], 'min_samples_split': [2], 'alpha': [0.1]}
        ]
        # grid_search = GridSearchCV(grid_search, param_grid, cv=5,
        #                            scoring='neg_mean_squared_error', verbose=100)
        grid_search.fit(x_train, y_train)

        # best_params_ = grid_search.best_params_
        # best_score_ = grid_search.best_score_
        # print("best_params_:", best_params_)
        # print("best_score_:", best_score_)

        score = grid_search.score(x_test, y_test)
        y_predict = grid_search.predict(x_test)
        y_predict = y_transfer.inverse_transform(DataFrame(y_predict))
        y_test = y_transfer.inverse_transform(y_test)
        # print("y_predict:", y_predict)
        # print("直接比对：\n", y_test, y_predict)

        # print("精准率：\n", score)
        # 保存模型
        joblib.dump(grid_search, "model\\model.pkl")
        joblib.dump(x_transfer, "model\\x_transfer")
        joblib.dump(y_transfer, "model\\y_transfer")
        # 预测明天的情况
        today_data = x_transfer.transform(today_data)
        t_predict = grid_search.predict(today_data)
        t_predict = y_transfer.inverse_transform(DataFrame(t_predict))

    # 绘图
    # y_test = y_test.sort_index()
    # plt.plot(y_test.values, color='red', label='Original')
    # plt.plot(y_predict, color='green', label='Predict')
    # plt.xlabel('the number of test data')
    # plt.ylabel('earn_rate')
    # plt.title('')
    # plt.legend()
    # plt.show()
    return score, t_predict


def train_and_predict(code, today_data, grid_search):
    # 读取文件
    column_names = ['日期', '股票代码', '名称', '收盘价', '最高价', '最低价', '开盘价', '前收盘',
                    '涨跌额', '涨跌幅', '换手率', '成交量', '成交金额', '流通市值', '总市值', 'tomorrowZhangFu',
                    'afterTomorrowHigh', 'tomorrowLow']

    gupiao_name = today_data.split(",")[2]
    if gupiao_name.__contains__("ST") or gupiao_name.__contains__("退市"):
        return

    if os.path.exists("model\\model.pkl"):
        data = pd.read_csv(end_date + "\\data\\" + code + ".csv", names=column_names)
    else:
        data = pd.read_csv("data_all.csv", names=column_names)
    # 如果日期相同，取数据里边的日期
    today_riqi = data.get('日期')
    data.drop(['日期', '股票代码', '名称', '涨跌额', '成交金额'], axis=1, inplace=True)

    real_data_today_riqi = today_data.split(",")[0]
    data_today_riqi = list(today_riqi).__getitem__(0)

    # if str(data_today_riqi) == str(real_data_today_riqi):
    today_data = DataFrame([DataFrame(data.iloc[0].values.tolist())[0]]).iloc[:, :len(data.columns) - 3]
    today_price = list(today_data.get(0)).__getitem__(0)
    today_huan = list(today_data.get(6)).__getitem__(0)
    # else:
    #     today_price = float(today_data.split(",")[3])
    #     today_data = DataFrame([today_data.split(",")]).iloc[:, 3: 13]
    #     today_huan = float(today_data.split(",")[10])

    print(code + " today_price", today_price, " 换手：", today_huan)

    data.drop(0, axis=0, inplace=True)

    # 第二天的必须删除，不知道后天的高点
    # data.drop(1, axis=0, inplace=True)

    # 数据处理，去除？转换为nan
    data.replace(to_replace="?", value=np.nan)
    # 删除nan的数据
    data.dropna(inplace=True)
    data_length = len(data)
    if data_length <= 0:
        return

    # 取特征值、目标值
    x = data.iloc[:, : len(data.columns) - 3]

    # 预测明天的情况
    # 明天的涨幅
    score_result = get_score_result(x, data.iloc[:, len(data.columns) - 3:len(data.columns)], today_data,
                                    grid_search)
    score = score_result[0]
    t_predict_zhangfu = score_result[1][0][0]
    t_predict_high = score_result[1][0][1]
    t_predict_low = score_result[1][0][2]

    # 绘图
    # y_test = y_test.sort_index()
    # plt.plot(y_test.values, color='red', label='Original')
    # plt.plot(y_predict, color='green', label='Predict')
    # plt.xlabel('the number of test data')
    # plt.ylabel('earn_rate')
    # plt.title('')
    # plt.legend()
    # plt.show()
    if t_predict_zhangfu > 0:
        result_type = "涨"
        file_type = "red"
    else:
        result_type = "跌"
        file_type = "green"

    fudong = ((t_predict_high - t_predict_low) / t_predict_low) * 100
    # result = "股票编号：" + code + "\r\n股票名称：" + gupiao_name + "\r\n下个交易日会" + result_type + \
    #          str(t_predict_zhangfu) + ",精确率：" + str(score_zhangfu) + "\r\n" + "高：" + \
    #          str(t_predict_high) + ",精确率：" + str(score_high) + "\r\n" + "低：" + str(t_predict_low) + ",精确率：" + str(
    #     score_low) + "\r\n浮动：" + str(fudong)
    #
    # if fudong > 7:
    #     with open(os.path.join(end_date + "\\predict_linear", file_type + code + '.csv'.format(code=code)), 'w',
    #               encoding='utf-8') as f:
    #         f.write(result)
    t_predict_low = t_predict_low + today_price * 0.005
    zhangfu = ((t_predict_high - today_price) / today_price) * 100
    diefu = -((t_predict_low - today_price) / today_price) * 100

    # 保留两位小数
    t_predict_low = round(t_predict_low, 2)
    t_predict_high = round(t_predict_high, 2)

    result = "股票编号：" + code + ",下个交易日会" + result_type + str(t_predict_zhangfu) + "\n股票名称：" + gupiao_name + "\n收：" + \
             str(t_predict_high) + ",精确率：" + str(score) + "\n" + "低：" + str(t_predict_low) + "\n浮动：" + str(
        fudong) + "\n跌幅：" + str(diefu) + "\n涨幅：" + str(zhangfu)

    # if (2.2 > zhangfu > 1.8 or 4 > zhangfu > 3.2) or (is_my_code):
    if (3 > t_predict_zhangfu > 2 and diefu < 0.9) or (is_my_code):
        with open(os.path.join(end_date + "\\predict_linear", gupiao_name + '.csv'.format(code=code)), 'w',
                  encoding='utf-8') as f:
            f.write(result)

        all_result = gupiao_name + "[" + str(t_predict_low) + "-" + str(t_predict_high) + "]\n"

        with open(os.path.join(end_date + "\\predict_linear", 'all_result.csv'.format(code=code)), 'a',
                  encoding='utf-8') as f:
            f.write(all_result)
    print(result)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        all_code = sys.argv[1:]
        # print(all_code)
    else:
        # 换手
        all_code_url = "http://44.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=3000&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid=f8&fs=m:0+t:6,m:0+t:13,m:0+t:80,m:1+t:2,m:1+t:23&fields=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152&_=1579615221139"
        # 成交金额
        # all_code_url = "http://44.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=1000&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid=f6&fs=m:0+t:6,m:0+t:13,m:0+t:80,m:1+t:2,m:1+t:23&fields=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152&_=1579615221139"
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

    data_path = os.path.join(end_date, "data")
    os.makedirs(data_path, exist_ok=True)
    predict_path = os.path.join(end_date, "predict_linear")
    os.makedirs(predict_path, exist_ok=True)
    for code in list(all_code):
        if code.startswith("688") or code.startswith("30"):
            all_code.remove(code)

    random.shuffle(all_code)
    # print(start_date, end_date)

    # all_code = ["600703"]
    if len(all_code) < 20:
        is_my_code = True

    grid_search = "1"
    if os.path.exists("model\\model.pkl"):
        grid_search = joblib.load("model\\model.pkl")
        print("加载模型成功")
        worker_num = 20
    else:
        worker_num = 1

    # 多线程
    with ThreadPoolExecutor(max_workers=worker_num) as tpe:
        tpe.map(download_code_data,
                [(code, start_date, end_date, data_path, predict_path, grid_search) for code in all_code])

    # for code in all_code:
    #     get_data([code, start_date, end_date, data_path, predict_path, grid_search])
