import pandas as pd
import numpy as np
from utils.timefeatures import time_features
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from TSMixer.TSMixer import Model
from scipy.io import savemat
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
warnings.filterwarnings('ignore')

file_path = 'data\\FC\\Ocean.csv'
window = 12  # 模型输入序列长度
length_size = 1  # 预测结果的序列长度
epochs = 50  # 迭代次数
batch_size = 64
# 读取数据
df_raw = pd.read_csv(file_path)  # 读取目标文件
df_stamp = df_raw[['date']]
df_stamp['date'] = pd.to_datetime(df_stamp.date)
data_stamp = time_features(df_stamp, timeenc=1, freq='h')
print(data_stamp)

data = df_raw.iloc[:, 1:]  # 第一列为时间，去除时间列
data_target = df_raw.iloc[:, -1:]  # 目标数据
data_dim = len(data.iloc[1, :])
scaler = preprocessing.MinMaxScaler()
if data_dim == 1:
    data_inverse = scaler.fit_transform(np.array(data).reshape(-1, 1))  # 将目标数据变成二维数组，并进行归一化
else:
    data_inverse = scaler.fit_transform(np.array(data))
data = data_inverse
data_length = len(data)
train_set = 0.7

data_train = data[:int(train_set * data_length), :]  # 训练集
data_train_mark = data_stamp[:int(train_set * data_length), :]
data_test = data[int(train_set * data_length):, :]  # 测试集
data_test_mark = data_stamp[int(train_set * data_length):, :]

n_feature = data_dim


def data_loader(window, length_size, batch_size, data, data_mark):
    # 构建LSTM输入
    seq_len = window
    sequence_length = seq_len + length_size  # 输入序列的长度+预测序列的长度
    result = []
    result_mark = []
    for index in range(len(data) - sequence_length + 1):
        result.append(data[index: index + sequence_length])  # 输入序列
        result_mark.append(data_mark[index: index + sequence_length])
    result = np.array(result)
    result_mark = np.array(result_mark)
    x_train = result[:, :-length_size]  # 输入特征
    x_train_mark = result_mark[:, :-length_size]
    y_train = result[:, -(length_size + int(window / 2)):]  # 目标数据
    y_train_mark = result_mark[:, -(length_size + int(window / 2)):]

    x_train, y_train = torch.tensor(x_train).to(torch.float32), torch.tensor(y_train).to(torch.float32)
    x_train_mark, y_train_mark = torch.tensor(x_train_mark).to(torch.float32), torch.tensor(y_train_mark).to(
        torch.float32)
    ds = torch.utils.data.TensorDataset(x_train, y_train, x_train_mark, y_train_mark)
    dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    return dataloader, x_train, y_train, x_train_mark, y_train_mark


dataloader_train, x_train, y_train, x_train_mark, y_train_mark = data_loader(window, length_size, batch_size,
                                                                             data_train, data_train_mark)
dataloader_test, x_test, y_test, x_test_mark, y_test_mark = data_loader(window, length_size, batch_size, data_test,
                                                                        data_test_mark)

# 强制使用CPU
device = torch.device("cpu")
print("Using CPU.")


class Config:
    def __init__(self):
        self.seq_len = window
        self.label_len = int(window / 2)
        self.pred_len = length_size
        self.e_layers = 2
        self.dec_in = n_feature
        self.enc_in = n_feature
        self.c_out = 1
        self.d_model = 32
        self.dropout = 0.1


config = Config()


def TSMixer_train(config):
    net = Model(config).to(device)
    criterion = nn.MSELoss().to(device)  # 损失函数
    optimizer = optim.Adam(net.parameters(), lr=0.0001)  # 优化器

    iteration = 0
    for epoch in range(epochs):
        for i, (datapoints, labels, datapoints_mark, labels_mark) in enumerate(dataloader_train):
            datapoints, labels, datapoints_mark, labels_mark = datapoints.to(device), labels.to(
                device), datapoints_mark.to(device), labels_mark.to(device)
            optimizer.zero_grad()
            preds = net(datapoints, datapoints_mark, labels, labels_mark, None)
            labels = labels[:, -length_size:].squeeze()
            preds = preds.squeeze()
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            iteration += 1
            if iteration % 100 == 0:
                print(f"Iteration: {iteration} Loss: {loss:.4f}")

    best_model_path = 'checkpoint/best_TSMixer.pt'
    torch.save(net.state_dict(), best_model_path)
    return net


def TSMixer_test(config):
    net = Model(config).to(device)
    net.load_state_dict(torch.load('checkpoint/best_TSMixer.pt'))  # 加载训练好的模型
    net.eval()

    pred = net(x_test.to(device), x_test_mark.to(device), y_test.to(device), y_test_mark.to(device))
    pred = pred.detach().cpu()
    true = y_test[:, -length_size:].detach().cpu()

    print("Shape of pred before adjustment:", pred.shape)
    print("Shape of true before adjustment:", true.shape)

    # 维度调整
    true = true[:, :, -1]
    pred = pred[:, :, -1]

    print("Shape of pred after adjustment:", pred.shape)
    print("Shape of true after adjustment:", true.shape)

    # 数据逆归一化
    y_data_test_inverse = scaler.fit_transform(np.array(data_target).reshape(-1, 1))
    pred_uninverse = scaler.inverse_transform(pred[:, -1:])
    true_uninverse = scaler.inverse_transform(true[:, -1:])

    return true_uninverse, pred_uninverse


def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100


if __name__ == "__main__":
    # 训练
    TSMixer_train(config)
    true, pred = TSMixer_test(config)

    # 保存结果
    savemat('results\\预测步长{}_TSMixer.mat'.format(length_size), {'true': true, 'pred': pred})
    result_finally = np.concatenate((true, pred), axis=1)
    print(result_finally.shape)

    df = pd.DataFrame(result_finally, columns=['real', 'pred'])
    df.to_csv('results\\预测步长{}_TSMixer.csv'.format(length_size), index=False)

    # 获取时间戳
    time_stamps = df_stamp['date'].iloc[-len(result_finally):]  # 使用最新的预测结果对应的日期

    # 绘制预测图
    plt.figure(figsize=(12, 6))
    plt.plot(time_stamps, result_finally[:, 0], c='red', linestyle='--', linewidth=1, label='True')  # 使用日期作为 x 轴
    plt.plot(time_stamps, result_finally[:, 1], c='black', linestyle='-', linewidth=1, label='Predicted')
    plt.title('ConvTSMixer Quantile Prediction Results')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 保存图像为 EPS 格式
    plt.savefig('images\\预测步长{}_TSMixer.eps'.format(length_size), dpi=500, format='eps')
    plt.show()

    # 计算评价指标
    y_test = result_finally[:, 0]
    y_test_predict = result_finally[:, 1]
    R2 = 1 - np.sum((y_test - y_test_predict) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    MAE = mean_absolute_error(y_test_predict, y_test)
    RMSE = np.sqrt(mean_squared_error(y_test_predict, y_test))
    MAPE_value = mape(y_test_predict, y_test)

    print('MAE:', MAE)
    print('RMSE:', RMSE)
    print('MAPE:', MAPE_value)
    print('R2:', R2)

    savef = pd.DataFrame()
    savef['MAE'] = [str(MAE)]
    savef['RMSE'] = [str(RMSE)]
    savef['MAPE'] = [str(MAPE_value)]
    savef['R2'] = [str(R2)]
    savef.to_csv('error\\error_预测步长{}_TSMixer.csv'.format(length_size), index=False)