import pandas as pd
from joblib import dump
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

days_ahead = 144
range = 'm5'
model_filename = f'../model/stock_prediction_model_{range}.joblib'
scaler_filename = f'../model/scaler_{range}.joblib'


def model_practice():
    # 读取数据，没有标题
    file_path = f'../data/{range}data_all.csv'
    df = pd.read_csv(file_path, header=None)
    # 手动指定列名
    column_names = ['stock_code', 'date', 'open', 'close', 'high', 'low', 'volume', 't1', 't2']
    df.columns = column_names
    # 排序数据，确保时间顺序
    df = df.sort_values(by=['stock_code', 'date'], ascending=True)
    df.drop(columns=['date', 't1', 't2'], inplace=True)

    # 定义函数以创建特征和目标
    def create_features_targets(df, window_size=640):
        features, targets = [], []
        grouped = df.groupby('stock_code')

        for stock_code, group in grouped:
            group = group.drop(columns=['stock_code']).reset_index(drop=True)

            if group.shape[0] >= window_size:
                try:
                    feature = group.iloc[0:window_size - days_ahead].values  # 保持数据的二维结构

                    target = group.iloc[window_size - 48:window_size - 1]['close'].max() / group['close'].iloc[
                        window_size - days_ahead] - 1
                    features.append(feature.flatten())  # 如果需要将数据转成一维，应用 flatten()
                    targets.append(target)
                except Exception as e:
                    print(e)

        return features, targets

    # 创建特征和目标
    X, y = create_features_targets(df)

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 拆分训练和测试数据
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.03, random_state=42)
    # 通过网格参数训练模型
    param_grid = {
        'n_estimators': [180, 200, 220, 240, 260],
        'learning_rate': [0.04, 0.05, 0.06],
        'max_depth': [5]
    }

    # 创建模型

    # 创建 GridSearchCV 对象,使用gpu训练
    model = XGBRegressor(n_estimators=2000, learning_rate=0.05, max_depth=5, n_jobs=-1, device='cuda:0')
    # model = GridSearchCV(estimator=model, param_grid=param_grid, cv=4, scoring='r2', verbose=3)

    model.fit(X_train, y_train)
    # print("Best parameters:", model.best_params_)
    # 保存模型和标准化器
    dump(model, model_filename)
    dump(scaler, scaler_filename)

    # 绘制预测曲线
    plot_prediction_curve(X_test, y_test, model, scaler)


def plot_prediction_curve(X_test, y_test, model, scaler):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'Mean Absolute Error: {mae}')
    print(f'R^2 Score: {r2}')
    # 使用训练好的模型预测
    y_pred = model.predict(X_test)

    # 绘制真实值与预测值的曲线
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='True values', color='blue', linestyle='-', marker='o', alpha=0.7)  # 真实值带有透明度
    plt.plot(y_pred, label='Predicted values', color='red', linestyle='--', marker='x', alpha=0.7)  # 预测值带有透明度
    plt.title('True vs Predicted Values')
    plt.xlabel('Samples')
    plt.ylabel('Price Change')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    model_practice()
