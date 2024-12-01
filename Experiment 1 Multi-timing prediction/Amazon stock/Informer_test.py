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
from informer.Informer import Model
from scipy.io import savemat
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
warnings.filterwarnings('ignore')

file_path = 'data\\FC\\Amazon_modified.csv'
window = 15 # 模型输入序列长度
length_size = 1  # 预测结果的序列长度
epochs = 100 # 迭代次数
batch_size = 32

# 读取数据
df_raw = pd.read_csv(file_path)  # 读取目标文件
df_stamp = df_raw[['date']]
df_stamp['date'] = pd.to_datetime(df_stamp.date)
data_stamp = time_features(df_stamp, timeenc=1, freq='h')
print(data_stamp)

data = df_raw.iloc[:, -1:]  # 第一列为时间，去除时间列
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

data_train = data[:int(train_set * data_length), :]  # 读取目标数据，第一列记为0：1，后面以此类推, 训练集和验证集，如果是多维输入的话最后一列为目标列数据
data_train_mark = data_stamp[:int(train_set * data_length), :]
data_test = data[int(train_set * data_length):, :]  # 这里把训练集和测试集分开了，也可以换成两个csv文件
data_test_mark = data_stamp[int(train_set * data_length):, :]

n_feature = data_dim

def data_loader(window, length_size, batch_size, data, data_mark):
    # 构建lstm输入
    seq_len = window     #模型每次输入序列 输入序列长度
    sequence_length = seq_len + length_size  # 序列长度，也就是输入序列的长度+预测序列的长度
    result = []    #空列表
    result_mark = []
    for index in range(len(data) - sequence_length + 1):  # 循环次数为数据集的总长度
        result.append(data[index: index + sequence_length])  # 第i行到i+sequence_length
        result_mark.append(data_mark[index: index + sequence_length])
    result = np.array(result)  # 得到样本，样本形式为sequence_length*特征
    result_mark = np.array(result_mark)
    x_train = result[:, :-length_size]  # 训练集特征数据
    print('x_train shape:', x_train.shape)
    x_train_mark = result_mark[:, :-length_size]
    print('x_train_mark shape:', x_train_mark.shape)
    y_train = result[:, -(length_size + int(window/2)):]  # 训练集目标数据
    print('y_train shape:', y_train.shape)
    y_train_mark = result_mark[:, -(length_size + int(window/2)):]
    print('y_train_mark shape:', y_train_mark.shape)
    #X_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], data_dim))  # 重塑数据形状，保证数据顺利输入模型
    #y_train = np.reshape(y_train, (y_train.shape[0], -1))

    x_train, y_train = torch.tensor(x_train).to(torch.float32), torch.tensor(y_train).to(torch.float32)    #将数据转变为tensor张量
    x_train_mark, y_train_mark = torch.tensor(x_train_mark).to(torch.float32), torch.tensor(y_train_mark).to(torch.float32)  # 将数据转变为tensor张量
    ds = torch.utils.data.TensorDataset(x_train, y_train, x_train_mark, y_train_mark)
    dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)  # 对训练集数据进行打包，每32个数据进行打包一次，组后一组不足32的自动打包
    return dataloader, x_train, y_train, x_train_mark, y_train_mark


dataloader_train, x_train, y_train, x_train_mark, y_train_mark = data_loader(window, length_size, batch_size, data_train, data_train_mark)
dataloader_test, x_test, y_test, x_test_mark, y_test_mark = data_loader(window, length_size, batch_size, data_test, data_test_mark)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

class Config:
    def __init__(self):
        # self.features = "M"
        self.seq_len = window #
        self.label_len = int(window/2) #
        self.pred_len = length_size #
        self.e_layers = 2 #块的数量
        self.d_layers = 1
        self.dec_in = n_feature
        self.enc_in = n_feature #特征维度
        self.c_out = 1
        self.d_model = 32 #
        self.d_ff = 32  #
        self.top_k = 5
        self.num_kernels = 6
        self.embed = 'timeF'
        self.freq = 'h'
        self.dropout = 0.1
        self.factor = 5
        self.n_heads = 8
        self.output_attention = 0
        self.distil = 1
        self.activation = 'gelu'

config = Config()


def informer_train(config):
    net = Model(config).to(device)
    criterion = nn.MSELoss().to(device)  # 损失函数
    optimizer = optim.Adam(net.parameters(), lr=0.001)  # 优化算法和学习率
    """
    模型训练过程
    """
    iteration = 0
    for epoch in range(epochs):      #10
        for i, (datapoints, labels, datapoints_mark, labels_mark) in enumerate(dataloader_train):
            datapoints, labels, datapoints_mark, labels_mark = datapoints.to(device), labels.to(device), datapoints_mark.to(device), labels_mark.to(device)
            optimizer.zero_grad()

            preds = net(datapoints, datapoints_mark, labels, labels_mark, None)
            labels = labels[:,-length_size:,-1:].squeeze()
            preds = preds.squeeze()
            loss =criterion(preds, labels)
            loss.backward()
            optimizer.step()

            iteration += 1
            if iteration % 100 == 0:     #250
                print("Iteration: {} Val-loss: {:.4f}".format(str(iteration), loss))
    best_model_path = 'checkpoint/best_informer.pt'
    torch.save(net.state_dict(), best_model_path)
    return net
def informer_test(config):
    net = Model(config).to(device)
    net.load_state_dict(torch.load('checkpoint/best_informer.pt'))  # 加载训练好的模型
    net.eval()

    pred = net(x_test.to(device), x_test_mark.to(device), y_test.to(device), y_test_mark.to(device))
    pred = pred.detach().cpu()
    true = y_test[:,-length_size:,-1:].detach().cpu()
    # 检查pred和true的维度并调整
    print("Shape of pred before adjustment:", pred.shape)
    print("Shape of true before adjustment:", true.shape)

    # 可能需要调整pred和true的维度，使其变为二维数组
    true = true[:, :, -1]
    pred = pred[:, :, -1]  # 假设需要将pred调整为二维数组，去掉最后一维
    # true =np.array(true)
    # 假设需要将true调整为二维数组

    print("Shape of pred after adjustment:", pred.shape)
    print("Shape of true after adjustment:", true.shape)

    y_data_test_inverse = scaler.fit_transform(np.array(data_target).reshape(-1, 1))    #这段代码是为了重新更新scaler，因为之前定义的scaler是是十六维，这里重新根据目标数据定义一下scaler
    pred_uninverse = scaler.inverse_transform(pred[:, -1:])    #如果是多步预测， 选取最后一列
    true_uninverse = scaler.inverse_transform(true[:, -1:])

    return true_uninverse, pred_uninverse
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

if __name__ == "__main__":
    informer_train(config)
    true, pred = informer_test(config)
    savemat('results\\预测步长{}_informer.mat'.format(length_size), {'true':true, 'pred': pred})
    result_finally = np.concatenate((true, pred), axis=1)
    print(result_finally.shape)
    df = pd.DataFrame(result_finally, columns=['real', 'pred'])
    df.to_csv('results\\预测步长{}_informer.csv'.format(length_size), index=False)
    time = np.arange(len(result_finally))
    plt.figure(figsize=(12, 3))
    plt.plot(time, result_finally[:, 0], c='red', linestyle='--', linewidth=1, label='true')
    plt.plot(time, result_finally[:, 1], c='black', linestyle='-', linewidth=1, label='pred')
    plt.title('TimeNet Quantile prediction results')
    plt.legend()
    plt.savefig('images\\预测步长{}_informer.png'.format(length_size), dpi=100)
    plt.show()

    y_test = result_finally[:, 0]
    y_test_predict = result_finally[:, 1]
    R2 = 1 - np.sum((y_test - y_test_predict) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    MAE = mean_absolute_error(y_test_predict, y_test)
    RMSE = np.sqrt(mean_squared_error(y_test_predict, y_test))
    MAPE = mape(y_test_predict, y_test)

    print('MAE:', MAE)
    print('RMSE:', RMSE)
    print('MAPE:', MAPE)
    print('r2:', R2)
    savef = pd.DataFrame()
    savef['MAE'] = [str(MAE)]
    savef['RMSE'] = [str(RMSE)]
    savef['MAPE'] = [str(MAPE)]
    savef['R2'] = [str(R2)]
    savef.to_csv('error\\error_预测步长{}_informer.csv'.format(length_size), index=False)