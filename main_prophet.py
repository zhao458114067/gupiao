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
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric
import warnings
warnings.filterwarnings('ignore')

# 定义模型
def FB(data: pd.DataFrame) -> pd.DataFrame:
    df = data

    # df['cap'] = data.total_purchase_amt.values.max()
    # df['floor'] = data.total_purchase_amt.values.min()

    m = Prophet(
        changepoint_prior_scale=0.05,
        daily_seasonality=False,
        yearly_seasonality=True,  # 年周期性
        weekly_seasonality=True,  # 周周期性
        growth="logistic",
    )

    #     m.add_seasonality(name='monthly', period=30.5, fourier_order=5, prior_scale=0.1)#月周期性
    m.add_country_holidays(country_name='CN')  # 中国所有的节假日

    m.fit(df)

    future = m.make_future_dataframe(periods=30, freq='D')  # 预测时长
    future['cap'] = data.total_purchase_amt.values.max()

    forecast = m.predict(future)

    fig = m.plot_components(forecast)
    fig1 = m.plot(forecast)

    return forecast


def download_code_data(*args):
    code, start_date, end_date, data_path, predict_path = args
    code_data_length_limit = 20000
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
                now = text[i - 1 - 1].split(",")[9]
                next = text[i - 1 - 1 - 1].split(",")[9]

                if next.lower() == "none" or now.lower() == "none":
                    text[i - 1 - 1] = ""
                    continue

                # 处理日期
                today = text[i - 1 - 1].split(",\'")[0]
                yueri = today.split("-")[1] + today.split("-")[2]
                text[i - 1 - 1] = text[i - 1 - 1] + "," + next
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
            train_and_predict(code, today_data)
            return True
    return False


def train_and_predict(code, today_data):
    # 读取文件
    column_names = ['日期', '股票代码', '名称', '收盘价', '最高价', '最低价', '开盘价', '前收盘',
                    '涨跌额', '涨跌幅', '换手率', '成交量', '成交金额', '总市值', '流通市值', 'Class']

    data = pd.read_csv(end_date + "\\data\\" + code + ".csv", names=column_names)
    # 数据处理，去除？转换为nan
    data.replace(to_replace="?", value=np.nan)
    # 删除nan的数据
    data.dropna(inplace=True)
    # 取特征值、目标值
    x = data.iloc[:, [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]
    y = data["Class"]
    # 划分训练集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    # 特征工程，标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 模型
    result_purchase = FB(data)

    # 预测明天的情况
    # today_data = "0422,'002120,韵达股份,12.93,13.15,12.86,13.13,13.1,-0.17,-1.2977,0.3596,10407365,135159006.66,37485649502.9,37420030528.7"
    # x_today = DataFrame([today_data.split(",")])
    # x_today = x_today.iloc[:, [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]
    # x_today = transfer.transform(x_today)
    # t_predict = esetimator.predict(x_today)
    #
    # plt.plot(y_test, color='red', label='Original')
    # plt.plot(y_train, color='green', label='Predict')
    # plt.xlabel('the number of test data')
    # plt.ylabel('earn_rate')
    # plt.title('2016.3—2017.12')
    # plt.legend()
    # plt.show()
    # if t_predict[0] > 0:
    #     result = "股票编号：" + code + "\r\n股票名称：" + today_data.split(",")[2] + "\r\n下个交易日会涨" + str(
    #         t_predict) + "精确率：" + str(score)
    #     if score > 0.99:
    #         with open(os.path.join(end_date + "\\predict_linear", "red" + code + '.csv'.format(code=code)), 'w',
    #                   encoding='utf-8') as f:
    #             f.write(result)
    # else:
    #     result = "股票编号：" + code + "\r\n股票名称：" + today_data.split(",")[2] + "\r\n下个交易日会跌" + str(
    #         t_predict) + "精确率：" + str(score)
    #     if score > 0.99:
    #         with open(os.path.join(end_date + "\\predict_linear", "green" + code + '.csv'.format(code=code)), 'w',
    #                   encoding='utf-8') as f:
    #             f.write(result)
    # print(result)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        all_code = sys.argv[1:]
        # print(all_code)
    else:
        all_code_url = "http://44.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=10000&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid=f3&fs=m:0+t:6,m:0+t:13,m:0+t:80,m:1+t:2,m:1+t:23&fields=f12&_=1579615221139"
        r = requests.get(all_code_url, timeout=5).json()
        all_code = [data['f12'] for data in r['data']['diff']]
        print(all_code)
    start_date = (date.today() - timedelta(days=360)).strftime("%Y%m%d")
    end_date = date.today().strftime("%Y%m%d")
    data_path = os.path.join(end_date, "data")
    os.makedirs(data_path, exist_ok=True)
    predict_path = os.path.join(end_date, "predict_linear")
    os.makedirs(predict_path, exist_ok=True)
    # print(start_date, end_date)
    # 下载数据
    # with ThreadPoolExecutor(max_workers=80) as tpe:
    #     tpe.map(download_code_data, [(code, start_date, end_date, data_path, predict_path) for code in all_code])
    # start()
    # for code in all_code:
    download_code_data("002120", start_date, end_date, data_path, predict_path)
