from concurrent.futures import ThreadPoolExecutor
from io import StringIO

import pandas as pd

from practice import model_filename, scaler_filename, days_ahead
from get_data import get_price_data, get_all_code
from joblib import load

model = load(model_filename)
scaler = load(scaler_filename)


# 使用特定股票的最近数据点进行预测
def predict_future(code):
    recent_data = get_price_data(code, 'm5', 640 - days_ahead)
    if recent_data:
        recent_data = pd.read_csv(StringIO(recent_data), header=None)
        recent_data.columns = ['stock_code', 'date', 'open', 'close', 'high', 'low', 'volume', 't1', 't2']
        recent_data.drop(columns=['stock_code', 'date', 't1', 't2'], inplace=True)
        # 对输入数据标准化
        recent_data_scaled = scaler.transform([recent_data.values.flatten()])

        # 预测未来三天的涨幅
        predictions = model.predict(recent_data_scaled)
        if predictions[0] > 0.08:
            print(f'go to buy {code}, prediction {predictions[0]}')


if __name__ == '__main__':
    all_code = get_all_code()
    with ThreadPoolExecutor(max_workers=50) as tpe:
        tpe.map(predict_future, [code for code in all_code])
