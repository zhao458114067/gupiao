import random
import sys
import os
import requests
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta, datetime

from main_RFR_all import before_days, headers

code_data_length_limit = 60
predit_days_number = 2


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
            for i in range(len(text) - 2, predit_days_number, -1):
                now = text[i].split(",")[9]
                price = text[i].split(",")[3]
                gupiao_name = text[i].split(",")[2]
                if gupiao_name.__contains__("ST") or gupiao_name.__contains__("退市") or float(price) > 50:
                    return False
                nextZhangFu = text[i - 1].split(",")[9]
                nextHigh = text[i - predit_days_number].split(",")[4]
                nextLow = text[i - 1].split(",")[5]
                if nextZhangFu.lower() == "none" or now.lower() == "none":
                    text[i] = ""
                    continue

                # 处理日期
                text[i] = text[i] + "," + nextZhangFu + "," + nextHigh + "," + nextLow
            # 获取今天的数据
            for i in range(predit_days_number + 1):
                text[i] = ""

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
        all_code_url = "http://44.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=3000&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid=f8&fs=m:0+t:6,m:0+t:13,m:0+t:80,m:1+t:2,m:1+t:23&fields=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152&_=1579615221139"
        # 成交金额
        # all_code_url = "http://44.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=500&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid=f6&fs=m:0+t:6,m:0+t:13,m:0+t:80,m:1+t:2,m:1+t:23&fields=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152&_=1579615221139"
        # 涨跌幅
        # all_code_url = "http://44.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=1000&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid=f3&fs=m:0+t:6,m:0+t:13,m:0+t:80,m:1+t:2,m:1+t:23&fields=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152&_=1579615221139"
        # 成交量
        # all_code_url = "http://44.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=1000&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid=f5&fs=m:0+t:6,m:0+t:13,m:0+t:80,m:1+t:2,m:1+t:23&fields=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152&_=1579615221139"
        r = requests.get(all_code_url, timeout=5).json()
        all_code = [data['f12'] for data in r['data']['diff']]
        all_name = [data['f14'] for data in r['data']['diff']]
        print(all_code)
    start_date = (date.today() - timedelta(days=code_data_length_limit * 1.6 + before_days)).strftime("%Y%m%d")

    end_date = (date.today() - timedelta(days=before_days)).strftime("%Y%m%d")

    for code in list(all_code):
        if code.startswith("688") or code.startswith("30"):
            all_code.remove(code)

    random.shuffle(all_code)
    # print(start_date, end_date)

    # 多线程
    with ThreadPoolExecutor(max_workers=100) as tpe:
        tpe.map(get_data, [(code, start_date, end_date) for code in all_code])

    # for code in all_code:
    #     get_data([code, start_date, end_date])
