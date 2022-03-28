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

is_my_code = False

# 前几天的数据，用于测试，计算明天是0
before_days = 1
code_data_length_limit = 3420

predit_days_number = 1

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'Accept-Encoding': 'gzip, deflate'
}


def get_today(code):
    url1 = (
        "https://59.push2his.eastmoney.com/api/qt/stock/kline/get?cb=&secid=1.{code}&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5%2Cf6&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58%2Cf59%2Cf60%2Cf61&klt=101&fqt=0&beg=20220101&end=20500101",
        "https://59.push2his.eastmoney.com/api/qt/stock/kline/get?cb=&secid=0.{code}&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5%2Cf6&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58%2Cf59%2Cf60%2Cf61&klt=101&fqt=0&beg=20220101&end=20500101")

    for u in url1:
        u = u.format(code=code)
        response = requests.get(u, headers=headers).text
        if (len(response) > 100):
            today_data = json.loads(response)["data"]
            get_code = today_data["code"]
            if code == get_code:
                klines = today_data["klines"]
                today_kline = klines[len(klines) - 1]
                today_kline = today_kline.split(",")
                today_huan = today_kline[len(today_kline) - 1]
                break

    url = (
        "http://api.money.126.net/data/feed/1{code}",
        "http://api.money.126.net/data/feed/0{code}"
    )
    for u in url:
        u = u.format(code=code)
        response = requests.get(u, headers=headers).text
        if (len(response) > 200):
            response = response.split("({")[1].split(" });")[0]
            c = code + "\":"
            response = response.split(c)[1]
            today_data = json.loads(response)
            if today_data["symbol"] == code:
                price = str(today_data["price"])
                today_data = date.today().strftime("%Y-%m-%d") + "," + today_data["symbol"] + "," + today_data["name"] \
                             + "," + str(today_data["price"]) + "," + str(today_data["high"]) \
                             + "," + str(today_data["low"]) + "," + str(today_data["open"]) \
                             + "," + str(today_data["yestclose"]) + "," + str(today_data["updown"]) \
                             + "," + str(today_huan) + "," + str(today_data["volume"])

                return today_data, price, today_huan


def download_code_data(*args):
    code, start_date, end_date, data_path, predict_path = args[0]
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
            for i in range(len(text) - 2, predit_days_number, -1):
                now = text[i].split(",")[9]
                price = text[i].split(",")[3]
                gupiao_name = text[i].split(",")[2]
                if gupiao_name.__contains__("ST") or gupiao_name.__contains__("退市") or float(price) > 50:
                    return False
                nextZhangFu = text[i - 1].split(",")[9]
                nextShou = text[i - predit_days_number].split(",")[3]
                nextLow = text[i - 1].split(",")[5]
                if nextZhangFu.lower() == "none" or now.lower() == "none":
                    text[i] = ""
                    continue

                # 处理日期
                text[i] = text[i] + "," + nextZhangFu + "," + nextShou + "," + nextLow
            # 获取今天的数据
            text[0] = ""

            try:
                today_data, price, today_huan = get_today(code)
            except:
                today_data = "1,1,1,1,1,1,1,1,1,1"

            with open(os.path.join(data_path, '{code}.csv'.format(code=code)), 'w', encoding='utf-8') as f:
                file_str = ""
                for data in text:
                    if data != "":
                        file_str = file_str + data + "\n"
                f.write(file_str)

            print("股票编号：" + code + " 数据爬取成功!\n")
            train_and_predict(code, today_data)
            return True
    return False


def get_score_result(x, y, today_data):
    # if os.path.exists("model.pkl"):
    #     grid_search = joblib.load("model.pkl")
    #     x_transfer = joblib.load("x_transfer")
    #     y_transfer = joblib.load("y_transfer")
    #     # 预测明天的情况
    #     score = 0
    #     x_today = x_transfer.transform(today_data)
    #     t_predict = grid_search.predict(x_today)
    #     t_predict = y_transfer.inverse_transform(t_predict)
    # else:
    x_transfer = MinMaxScaler()
    y_transfer = MinMaxScaler()
    # 划分训练集
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.1)
    # 特征工程，标准化
    x_train = x_transfer.fit_transform(x_train)
    y_train = y_transfer.fit_transform(DataFrame(y_train))
    x_test = x_transfer.transform(x_test)
    y_test = y_transfer.transform(DataFrame(y_test))
    # zx
    # grid_search = GradientBoostingRegressor(n_estimators=2000, learning_rate=0.05, max_depth=1, max_features='sqrt',
    #                                        min_samples_leaf=27, min_samples_split=2, loss='ls', random_state=42,
    #                                        alpha=0.1)
    # grid_search = GradientBoostingRegressor(n_estimators=2000, learning_rate=0.01, max_depth=15, max_features='sqrt',
    #                                        min_samples_leaf=10, min_samples_split=10, loss='ls', random_state=42)
    # if type == 'zhangfu':
    #     grid_search = RandomForestRegressor(bootstrap=True, n_estimators=245, max_features=1,
    #                                         random_state=42, n_jobs=-1)
    # if type == 'shou':
    grid_search = RandomForestRegressor(random_state=42, n_jobs=-1)
    # if type == 'di':
    #     grid_search = RandomForestRegressor(bootstrap=True, n_estimators=65, max_features=6,
    #                                         random_state=42, n_jobs=-1)

    # grid_search = GradientBoostingRegressor(random_state=42)
    param_grid = [
        # 12 (3×4) 种超参数组合
        {'learning_rate': [0.05], 'n_estimators': [100], 'max_depth': [1],
         'min_samples_leaf': [27], 'min_samples_split': [2], 'alpha': [0.1]}
    ]
    # grid_search = GridSearchCV(grid_search, param_grid, cv=5,
    #                            scoring='neg_mean_squared_error', verbose=100)

    grid_search.fit(x_train, y_train)

    # print("best_params_:", grid_search.best_params_)
    # print("best_score_", grid_search.best_score_)

    score = grid_search.score(x_test, y_test)
    y_predict = grid_search.predict(x_test)
    y_predict = y_transfer.inverse_transform(DataFrame(y_predict))
    y_test = y_transfer.inverse_transform(y_test)
    # print("y_predict:", y_predict)
    # print("直接比对：\n", y_test, y_predict)

    # print("精准率：\n", score)
    # 保存模型
    # joblib.dump(grid_search, "model.pkl")
    # joblib.dump(x_transfer, "x_transfer")
    # joblib.dump(y_transfer, "y_transfer")
    # 预测明天的情况
    today_data = x_transfer.transform(today_data)
    t_predict = grid_search.predict(today_data)
    t_predict = y_transfer.inverse_transform(t_predict.reshape(1, -1))

    # 绘图
    # plt.plot(y_test, color='red', label='Original')
    # plt.plot(y_predict, color='green', label='Predict')
    # plt.xlabel('the number of test data')
    # plt.ylabel('earn_rate')
    # plt.title('')
    # plt.legend()
    # plt.show()
    return score, t_predict


def train_and_predict(code, today_data):
    # 读取文件
    column_names = ['日期', '股票代码', '名称', '收盘价', '最高价', '最低价', '开盘价', '前收盘',
                    '涨跌额', '涨跌幅', '换手率', '成交量', '成交金额', '流通市值', '总市值', 'tomorrowZhangFu',
                    'afterTomorrowHigh', 'tomorrowLow']

    gupiao_name = today_data.split(",")[2]
    if gupiao_name.__contains__("ST") or gupiao_name.__contains__("退市"):
        return

    # if os.path.exists("model.pkl"):
    data = pd.read_csv(end_date + "\\data\\" + code + ".csv", names=column_names)
    # else:
    #     data = pd.read_csv("data.csv", names=column_names)
    # 如果日期相同，取数据里边的日期
    today_riqi = data.get('日期')
    data.drop(['日期', '股票代码', '名称', '收盘价', '最高价', '最低价', '开盘价', '前收盘',
               '涨跌额', '成交金额', '流通市值'], axis=1, inplace=True)

    real_data_today_riqi = today_data.split(",")[0]
    data_today_riqi = list(today_riqi).__getitem__(0)

    # if str(data_today_riqi) == str(real_data_today_riqi):
    today_data = DataFrame([DataFrame(data.iloc[0].values.tolist())[0]]).iloc[:, :len(data.columns) - 3]
    today_price = list(today_data.get(0)).__getitem__(0)
    # today_huan = list(today_data.get(7)).__getitem__(0)
    # else:
    #     today_price = float(today_data.split(",")[3])
    #     today_data = DataFrame([today_data.split(",")]).iloc[:, 3: 13]
    #     today_huan = float(today_data.split(",")[10])

    print(code + " today_zhangfu", today_price)

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
    score_result = get_score_result(x, data.iloc[:, len(data.columns) - 3:len(data.columns)], today_data)
    score = score_result[0]
    t_predict_zhangfu = score_result[1][0][0]
    t_predict_high = score_result[1][0][1]
    t_predict_low = score_result[1][0][2]
    # t_predict_zhangfu = score_result[1][0][0]
    # t_predict_high = score_result[1][0][1]
    # t_predict_low = score_result[1][0][2]

    # 绘图
    # y_test = y_test.sort_index()
    # plt.plot(y_test.values, color='red', label='Original')
    # plt.plot(y_predict, color='green', label='Predict')
    # plt.xlabel('the number of test data')
    # plt.ylabel('earn_rate')
    # plt.title('')
    # plt.legend()
    # plt.show()

    zhangfu = ((t_predict_high - today_price) / today_price) * 100
    t_predict_low = t_predict_low + today_price * 0.005
    diefu = ((t_predict_low - today_price) / today_price) * 100
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

    # 保留两位小数
    t_predict_low = round(t_predict_low, 2)
    t_predict_high = round(t_predict_high, 2)

    result = "股票编号：" + code + ",下个交易日会" + result_type + str(t_predict_zhangfu) + "\n股票名称：" + gupiao_name + "\n收盘：" + \
             str(t_predict_high) + ",精确率：" + str(score) + "\n" + "低点：" + str(t_predict_low) + "\n浮动：" + str(
        fudong) + "\n跌幅：" + str(diefu) + "\n涨幅：" + str(zhangfu)

    if (t_predict_zhangfu > 3) or (is_my_code):
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
        all_code_url = "http://44.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=500&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid=f8&fs=m:0+t:6,m:0+t:13,m:0+t:80,m:1+t:2,m:1+t:23&fields=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152&_=1579615221139"
        # 成交量
        # all_code_url = "http://44.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=500&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid=f6&fs=m:0+t:6,m:0+t:13,m:0+t:80,m:1+t:2,m:1+t:23&fields=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152&_=1579615221139"
        # 涨跌幅
        # all_code_url = "http://44.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=4000&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid=f3&fs=m:0+t:6,m:0+t:13,m:0+t:80,m:1+t:2,m:1+t:23&fields=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152&_=1579615221139"

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

    # random.shuffle(all_code)
    # print(start_date, end_date)

    # all_code = ["600111"]
    if len(all_code) < 20:
        is_my_code = True

    # 多线程
    with ThreadPoolExecutor(max_workers=20) as tpe:
        tpe.map(download_code_data, [(code, start_date, end_date, data_path, predict_path) for code in all_code])

    # for code in all_code:
    #     download_code_data([code, start_date, end_date, data_path, predict_path])
