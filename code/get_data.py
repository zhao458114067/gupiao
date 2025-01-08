import json
import random
import sys
import os
import requests
from concurrent.futures import ThreadPoolExecutor

gettrace = getattr(sys, 'gettrace', None)
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'Accept-Encoding': 'gzip, deflate'
}
PROXY = {
    "http": None,
    "https": None,
}


# 爬取数据
def get_price_and_save(code):
    range = 'm1'
    count = 640
    file_str = get_price_data(code, range, count)
    with open(os.path.join('../data', range + 'data_all.csv'), 'a', encoding='utf-8') as f:
        print("股票编号：" + code + " 数据爬取成功!\n")
        f.write(file_str)


def get_price_data(code, range, count):
    code_download_path_urls = (
        "http://ifzq.gtimg.cn/appstock/app/kline/mkline?param=sh{code},{range},,{count}",
        "http://ifzq.gtimg.cn/appstock/app/kline/mkline?param=sz{code},{range},,{count}",
    )
    file_str = ""
    for index, url_template in enumerate(code_download_path_urls):
        try:
            temp_code = 'sh' + str(code)
            if index == 1:
                temp_code = 'sz' + str(code)
            print("爬取：" + temp_code)

            request_url = url_template.format(code=code, range=range, count=count)
            response = requests.get(request_url, headers=headers, proxies=PROXY).text
            st = json.loads(response)
            text = st['data'][temp_code][range]

            if not text:
                continue

            for data in text:
                if data != "":
                    file_str = file_str + temp_code + "," + ",".join(map(str, data)) + "\n"
            break
        except:
            continue
    return file_str


def get_all_code():
    global all_code
    if len(sys.argv) > 1:
        all_code = sys.argv[1:]
        # print(all_code)
    else:
        # 换手
        all_code_url = "http://44.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=50000&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid=f8&fs=m:0+t:6,m:0+t:13,m:0+t:80,m:1+t:2,m:1+t:23&fields=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152&_=1579615221139"
        # 成交金额
        # all_code_url = "http://44.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=500&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid=f6&fs=m:0+t:6,m:0+t:13,m:0+t:80,m:1+t:2,m:1+t:23&fields=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152&_=1579615221139"
        # 涨跌幅
        # all_code_url = "http://44.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=1000&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid=f3&fs=m:0+t:6,m:0+t:13,m:0+t:80,m:1+t:2,m:1+t:23&fields=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152&_=1579615221139"
        # 成交量
        # all_code_url = "http://44.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=1000&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid=f5&fs=m:0+t:6,m:0+t:13,m:0+t:80,m:1+t:2,m:1+t:23&fields=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152&_=1579615221139"
        r = requests.get(all_code_url, timeout=15, proxies=PROXY).json()
        all_code = [data['f12'] for data in r['data']['diff']]
        all_name = [data['f14'] for data in r['data']['diff']]
        print(all_code)
    # start_date = (date.today() - timedelta(days=code_data_length_limit * 1.6 + before_days)).strftime("%Y%m%d")
    start_date = '20230914'
    # end_date = (date.today() - timedelta(days=before_days)).strftime("%Y%m%d")
    end_date = '20230914'
    for code in list(all_code):
        if code.startswith("688"):
            all_code.remove(code)
    random.shuffle(all_code)
    return all_code


if __name__ == '__main__':
    all_code = get_all_code()
    # print(start_date, end_date)

    worker_num = 20
    # debug下线程一个用于调试
    if gettrace():
        worker_num = 1

    # print(worker_num)

    # 多线程
    with ThreadPoolExecutor(max_workers=worker_num) as tpe:
        tpe.map(get_price_and_save, [code for code in all_code])

    # for code in all_code:
    #     get_data([code, start_date, end_date])
