import random

import pandas as pd
import numpy as np
import sys
import os
import requests
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta, datetime

from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
import joblib
import matplotlib.pyplot as plt
import json

is_my_code = False

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'Accept-Encoding': 'gzip, deflate'
}


def get_today(code):
    url = (
        "http://api.money.126.net/data/feed/0{code}",
        "http://api.money.126.net/data/feed/1{code}"
    )
    for u in url:
        u = u.format(code=code)
        response = requests.get(u, headers=headers).text
        if (len(response) > 200):
            response = response.split("({")[1].split(" });")[0]
            c = code + "\":"
            response = response.split(c)[1]
            today_data = json.loads(response)
            price = str(today_data["price"])
            today_data = date.today().strftime("%Y-%m-%d") + "," + today_data["symbol"] + "," + today_data["name"] \
                         + "," + str(today_data["price"]) + "," + str(today_data["high"]) \
                         + "," + str(today_data["low"]) + "," + str(today_data["open"]) \
                         + "," + str(today_data["yestclose"]) + "," + str(today_data["updown"]) \
                         + "," + str(today_data["percent"]) + "," + str(2.23) + "," + str(today_data["volume"]) \
                         + "," + str(today_data["turnover"])

            return today_data, price


def download_code_data(*args):
    code, start_date, end_date, data_path, predict_path = args
    code_data_length_limit = 4396
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
        if len(text) >= code_data_length_limit:
            for i in range(len(text), 3, -1):
                now = text[i - 1 - 1].split(",")[9]
                nextZhangFu = text[i - 1 - 1 - 1].split(",")[9]
                nextHigh = text[i - 1 - 1 - 1 - 1].split(",")[4]
                nextLow = text[i - 1 - 1 - 1].split(",")[5]
                if nextZhangFu.lower() == "none" or now.lower() == "none":
                    text[i - 1 - 1] = ""
                    continue

                # 处理日期
                text[i - 1 - 1] = text[i - 1 - 1] + "," + nextZhangFu + "," + nextHigh + "," + nextLow
            # 获取今天的数据
            text[0] = ""
            try:
                today_data, price = get_today(code)
            except:
                continue
            if today_data is not None and float(price) < 100:
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


def start_train(x, y, modelType):
    # 划分训练集
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    # 特征工程，标准化
    transfer = MinMaxScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # if os.path.exists("model1.pkl"):
    #     esetimator = joblib.load("model1.pkl")
    # else:
    esetimator = LinearRegression()

    esetimator.fit(x_train, y_train)
    score = esetimator.score(x_test, y_test)
    y_predict = esetimator.predict(x_test)

    # print("y_predict:", y_predict)
    # print("直接比对：\n", y_test, y_predict)

    # print("精准率：\n", score)
    # 保存模型
    joblib.dump(esetimator, modelType + ".pkl")

    return score, y_predict


def train_and_predict(code, today_data):
    # 读取文件
    column_names = ['日期', '股票代码', '名称', '收盘价', '最高价', '最低价', '开盘价', '前收盘',
                    '涨跌额', '涨跌幅', '换手率', '成交量', '成交金额', '流通市值', '总市值', 'tomorrowZhangFu',
                    'afterTomorrowHigh', 'tomorrowLow']

    gupiao_name = today_data.split(",")[2]
    #
    data = pd.read_csv(end_date + "\\data\\" + code + ".csv", names=column_names)
    # 如果日期相同，取数据里边的日期
    today_riqi = data.get('日期')
    data.drop(
        ['日期', '股票代码', '名称', '流通市值', '总市值'], axis=1, inplace=True)

    real_data_today_riqi = today_data.split(",")[0]
    data_today_riqi = list(today_riqi).__getitem__(0)
    # if data_today_riqi == real_data_today_riqi:
    today_data = DataFrame([DataFrame(data.iloc[0].values.tolist())[0]]).iloc[:, :len(data.columns) - 3]
    data.drop(0, axis=0, inplace=True)

    # 第二天的必须删除，不知道后天的高点
    data.drop(1, axis=0, inplace=True)

    # 数据处理，去除？转换为nan
    data.replace(to_replace="?", value=np.nan)
    # 删除nan的数据
    data.dropna(inplace=True)
    data_length = len(data)
    if data_length <= 0:
        return;

    # 取特征值、目标值
    x = data.iloc[:, : len(data.columns) - 3]

    # 预测明天的情况
    # 明天的涨幅
    score_result = get_score_result(x, data["tomorrowZhangFu"], today_data)
    score_zhangfu = score_result[0]
    t_predict_zhangfu = score_result[1]

    # 后天的高点
    score_result = get_score_result(x, data["afterTomorrowHigh"], today_data)
    score_high = score_result[0]
    t_predict_high = score_result[1]

    # 明天的低点
    score_result = get_score_result(x, data["tomorrowLow"], today_data)
    score_low = score_result[0]
    t_predict_low = score_result[1]

    # 绘图
    # y_test = y_test.sort_index()
    # plt.plot(y_test.values, color='red', label='Original')
    # plt.plot(y_predict, color='green', label='Predict')
    # plt.xlabel('the number of test data')
    # plt.ylabel('earn_rate')
    # plt.title('')
    # plt.legend()
    # plt.show()
    if t_predict_zhangfu[0] > 0:
        result_type = "涨"
        file_type = "red"
    else:
        result_type = "跌"
        file_type = "green"

    fudong = ((t_predict_high[0] - t_predict_low[0]) / t_predict_low[0]) * 100
    # result = "股票编号：" + code + "\r\n股票名称：" + gupiao_name + "\r\n下个交易日会" + result_type + \
    #          str(t_predict_zhangfu) + ",精确率：" + str(score_zhangfu) + "\r\n" + "高：" + \
    #          str(t_predict_high) + ",精确率：" + str(score_high) + "\r\n" + "低：" + str(t_predict_low) + ",精确率：" + str(
    #     score_low) + "\r\n浮动：" + str(fudong)
    #
    # if fudong > 7:
    #     with open(os.path.join(end_date + "\\predict_linear", file_type + code + '.csv'.format(code=code)), 'w',
    #               encoding='utf-8') as f:
    #         f.write(result)

    result = "股票编号：" + code + "\n股票名称：" + gupiao_name + "\n高：" + \
             str(t_predict_high) + ",精确率：" + str(score_high) + "\n" + "低：" + str(t_predict_low) + ",精确率：" + str(
        score_low) + "\n浮动：" + str(fudong)

    if (score_high > 0.991 and score_low > 0.998 and fudong > 4) or (is_my_code):
        with open(os.path.join(end_date + "\\predict_linear", code + '.csv'.format(code=code)), 'w',
                  encoding='utf-8') as f:
            f.write(result)

        all_result = gupiao_name + str(t_predict_low) + "-" + str(t_predict_high) + "\n"

        with open(os.path.join(end_date + "\\predict_linear", 'all_result.csv'.format(code=code)), 'a',
                  encoding='utf-8') as f:
            f.write(all_result)
    print(result)


def get_score_result(x, y, today_data):
    # 划分训练集
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    # 特征工程，标准化
    transfer = MinMaxScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # if os.path.exists("model1.pkl"):
    #     esetimator = joblib.load("model1.pkl")
    # else:
    esetimator = LinearRegression()

    esetimator.fit(x_train, y_train)
    score = esetimator.score(x_test, y_test)
    y_predict = esetimator.predict(x_test)

    # print("y_predict:", y_predict)
    # print("直接比对：\n", y_test, y_predict)

    # print("精准率：\n", score)
    # 保存模型
    # joblib.dump(esetimator, "model1.pkl")
    # 预测明天的情况
    # x_today = DataFrame([today_data.split(",")])
    x_today = transfer.transform(today_data)
    t_predict = esetimator.predict(x_today)
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


if __name__ == '__main__':
    if len(sys.argv) > 1:
        all_code = sys.argv[1:]
        # print(all_code)
    else:
        all_code_url = "http://44.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=1000&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid=f6&fs=m:0+t:6,m:0+t:13,m:0+t:80,m:1+t:2,m:1+t:23&fields=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152&_=1579615221139"
        r = requests.get(all_code_url, timeout=5).json()
        all_code = [data['f12'] for data in r['data']['diff']]
        all_name = [data['f14'] for data in r['data']['diff']]
        print(all_code)
    start_date = (date.today() - timedelta(days=30000)).strftime("%Y%m%d")

    end_date = (date.today()).strftime("%Y%m%d")
    # end_date = (date.today() - timedelta(days=3)).strftime("%Y%m%d")

    data_path = os.path.join(end_date, "data")
    os.makedirs(data_path, exist_ok=True)
    predict_path = os.path.join(end_date, "predict_linear")
    os.makedirs(predict_path, exist_ok=True)
    for code in list(all_code):
        if ("688" in code or code.startswith("30")):
            all_code.remove(code)
    random.shuffle(all_code)
    # print(start_date, end_date)
    # 下载数据
    # with ThreadPoolExecutor(max_workers=80) as tpe:
    #     tpe.map(download_code_data, [(code, start_date, end_date, data_path, predict_path) for code in all_code])
    # start()

    # all_code = ["600571", "000629", "600330", "600367", "600699", "600872"]
    if len(all_code) < 20:
        is_my_code = True

    for code in all_code:
        try:
            download_code_data(code, start_date, end_date, data_path, predict_path)
        except:
            print()
