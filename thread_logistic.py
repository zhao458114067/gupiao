import pandas as pd
import numpy as np
import sys
import os
import requests
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta, datetime

from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
import joblib
import matplotlib.pyplot as plt
from main_linear import get_today

symcode = "002387"
huanshou = 0

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'Accept-Encoding': 'gzip, deflate'
}


def download_code_data(*args):
    code, start_date, end_date, data_path, predict_path = args[0]
    code_data_length_limit = 2000
    code_download_path_urls = (
        "http://quotes.money.163.com/service/chddata.html?code=1{code}&start={start}&end={end}&fields={fields}",
        "http://quotes.money.163.com/service/chddata.html?code=0{code}&start={start}&end={end}&fields={fields}"
    )
    fields = 'TCLOSE;HIGH;LOW;TOPEN;LCLOSE;CHG;PCHG;TURNOVER;VOTURNOVER;VATURNOVER;TCAP;MCAP'
    for url_template in code_download_path_urls:
        u = url_template.format(code=code, start=start_date, end=end_date, fields=fields)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36',
            'Accept-Language': 'zh-CN,zh;q=0.9',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'Accept-Encoding': 'gzip, deflate'
        }
        response = requests.get(u, headers=headers).text
        # 转成数组
        if len(response) >= code_data_length_limit:
            text = requests.get(u, headers=headers).text.split("\r\n")
            for i in range(len(text), 3, -1):
                now = text[i - 1 - 1].split(",")[8]
                next = text[i - 1 - 1 - 1].split(",")[8]

                if next.lower() == "none" or now.lower() == "none":
                    continue
                if float(next) > 0:
                    text[i - 1 - 1] = text[i - 1 - 1] + "," + "1"
                else:
                    text[i - 1 - 1] = text[i - 1 - 1] + "," + "0"

                # 处理日期
                today = text[i - 1 - 1].split(",\'")[0]
                yueri = today.split("-")[1] + today.split("-")[2]
                text[i - 1 - 1] = yueri + text[i - 1 - 1].split(today)[1]
            # 获取今天的数据
            today_data = text[1]
            today_year = today_data.split(",\'")[0]
            today_month = today_year.split("-")[1] + today_year.split("-")[2]
            today_data = today_month + today_data.split(today_year)[1]
            text[0] = ""
            text[1] = ""
            with open(os.path.join(data_path, '{code}.csv'.format(code=code)), 'w', encoding='utf-8') as f:
                file_str = ""
                for data in text:
                    file_str = file_str + data + "\r\n"
                f.write(file_str)
            print("股票编号：" + code + " 数据爬取成功!\n")
            tomorrow_num = []
            for i in range(10):
                today_data = text[2]
                # today_data = get_today(code)
                train_and_predict(code, today_data, tomorrow_num)

            if len(tomorrow_num) > 7:
                result = "股票编号：" + code + "\r\n股票名称：" + today_data.split(",")[2] + "\r\n下个交易日会涨,概率为" + str(
                    tomorrow_num / 10)
                with open(os.path.join(end_date + "\\predict_logistic", "red" + code + '.csv'.format(code=code)), 'w',
                          encoding='utf-8') as f:
                    f.write(result)
            if len(tomorrow_num) < 3:
                result = "股票编号：" + code + "\r\n股票名称：" + today_data.split(",")[2] + "\r\n下个交易日会跌,概率为" + str(
                    tomorrow_num / 10)
                with open(os.path.join(end_date + "\\predict_logistic", "green" + code + '.csv'.format(code=code)), 'w',
                          encoding='utf-8') as f:
                    f.write(result)
            return True
    return False


# path = r'C:\Users\Dell\Desktop\cal\cml\3\xinan\xinan'
# p = read_csv(path, sep='\s+', encoding='ANSI', header=None)
# c = pd.concat(p)  # 将p里面全部dataframe合成一个


def train_and_predict(code, today_data, tomorrow_num):
    # 读取文件
    column_names = ['日期', '股票代码', '名称', '收盘价', '最高价', '最低价', '开盘价', '前收盘',
                    '涨跌额', '涨跌幅', '换手率', '成交量', '成交金额', '总市值', '流通市值', 'Class']

    data = pd.read_csv(end_date + "\\data\\" + code + ".csv", names=column_names)
    # 数据处理，去除？转换为nan
    data.replace(to_replace="?", value=np.nan)
    # 删除nan的数据
    data.dropna(inplace=True)
    # 取特征值、目标值
    x = data.iloc[:, [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
    y = data["Class"]
    # 划分训练集
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    # 特征工程，标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    if os.path.exists("model.pkl"):
        esetimator = joblib.load("model.pkl")
    else:
        esetimator = LogisticRegression()
    esetimator.fit(x_train, y_train)
    y_predict = esetimator.predict(x_test)

    print("y_predict:", y_predict)
    print("直接比对：\n", y_test == y_predict)
    score = esetimator.score(x_test, y_test)
    print("精准率：\n", score)
    # 保存模型
    joblib.dump(esetimator, "model.pkl")
    # 预测明天的情况
    x_today = DataFrame([today_data.split(",")])
    x_today = x_today.iloc[:, [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
    x_today = transfer.transform(x_today)
    t_predict = esetimator.predict(x_today)
    # 绘图
    plt.plot(y_test, color='red', label='Original')
    plt.plot(y_train, color='green', label='Predict')
    plt.xlabel('the number of test data')
    plt.ylabel('earn_rate')
    plt.title(start_date + '-' + end_date)
    plt.legend()
    # plt.show()
    if t_predict[0] > 0:
        tomorrow_num.append(1)
        # result = "股票编号：" + code + "\r\n股票名称：" + today_data.split(",")[2] + "\r\n下个交易日会涨" + str(
        #     t_predict) + "，精确率：" + str(
        #     score)
        # if score > 0.5:
        #     with open(os.path.join(end_date + "\\predict_logistic", "red" + code + '.csv'.format(code=code)), 'w',
        #               encoding='utf-8') as f:
        #         f.write(result)
    else:
        result = "股票编号：" + code + "\r\n股票名称：" + today_data.split(",")[2] + "\r\n下个交易日会跌" + str(
            t_predict) + "，精确率：" + str(
            score)
        # if score > 0.5:
        #     with open(os.path.join(end_date + "\\predict_logistic", "green" + code + '.csv'.format(code=code)), 'w',
        #               encoding='utf-8') as f:
        #         f.write(result)
    # print(result)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        all_code = sys.argv[1:]
        print(all_code)
    else:
        all_code_url = "http://44.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=10000&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid=f3&fs=m:0+t:6,m:0+t:13,m:0+t:80,m:1+t:2,m:1+t:23&fields=f12&_=1579615221139"
        r = requests.get(all_code_url, timeout=5).json()
        all_code = [data['f12'] for data in r['data']['diff']]
        print(all_code)
    start_date = (date.today() - timedelta(days=100)).strftime("%Y%m%d")
    end_date = date.today().strftime("%Y%m%d")
    data_path = os.path.join(end_date, "data")
    os.makedirs(data_path, exist_ok=True)
    predict_path = os.path.join(end_date, "predict_logistic")
    os.makedirs(predict_path, exist_ok=True)
    print(start_date, end_date)

    for code in list(all_code):
        if ("300" in code or "688" in code):
            all_code.remove(code)
            
    # 下载数据
    with ThreadPoolExecutor(max_workers=30) as tpe:
        tpe.map(download_code_data, [(code, start_date, end_date, data_path, predict_path) for code in all_code])
    # download_code_data(symcode, start_date, end_date, data_path, predict_path)
