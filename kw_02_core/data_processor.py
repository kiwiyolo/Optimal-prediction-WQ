import numpy as np


class DataLoader():

    def __init__(self, filename, split, cols):
        df = filename[cols].astype(float)
        dataframe = df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))    # 调用maxmin归一化方法进行归一化--要将‘_next_window’中的归一化注释掉！
        # dataframe = regularit(pd.read_table(filename))    # 调用rfr中的maxmin归一化方法进行归一化--要将‘_next_window’中的归一化注释掉！
        # dataframe = pd.read_table(filename)   # 用于训练模型时的数据路径输入
        # dataframe = filename                    # 用于验证时的数据输入
        i_split = int(len(dataframe) * split)   # 划分比→训练集长度
        self.data_train = dataframe.get(cols).values[:i_split]  # 划分数据集→训练集
        self.data_test  = dataframe.get(cols).values[i_split:]  # 划分数据集→测试集
        self.len_train  = len(self.data_train)  # 训练集长度
        self.len_test   = len(self.data_test)   # 测试集长度
        self.len_train_windows = None

    def get_test_data(self, seq_len, normalise):

        # 建立滑块（序列seq）数据
        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i+seq_len])    # 滑窗的时候是从下标1开始的，略去了第0个样本（1-seq_len）

        data_windows = np.array(data_windows).astype(float)
        # 只对x（特征值）进行归一化
        # x = self.normalise_windows(data_windows[:, :-1, :-1], single_window=False) if normalise else data_windows
        # 同时对x,y（特征值和目标值）均进行归一化
        # data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows # （15,8,13）
        x = data_windows[:, :, :-1]
        y = data_windows[:, -1, -1]
        return x, y

    def get_train_data(self, seq_len, normalise):

        # 建立滑块（序列seq）数据
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def generate_train_batch(self, seq_len, batch_size, normalise):
        
        i = 0
        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y = self._next_window(i, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, i, seq_len, normalise):
        window = self.data_train[i:i+seq_len]
        # window = self.normalise_windows(window, single_window=True)[0] if normalise else window
        # 将一个序列中所有的特征值作为x，最后一个目标值作为y
        x = window[:, :-1]
        y = window[-1, -1].reshape(1,1)
        return x, y

    def normalise_windows(self, window_data, single_window=False):
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                try:
                    normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]     # 此时的归一化方法和DataLoader'__init__'中的归一化中选取一个，此处是基于相对滑动序列的初始值的变幅归一的
                except:
                    print('第0行，第%s列：' % col_i, float(window[0, col_i]))
                normalised_window.append(normalised_col)
            normalised_window = np.array(normalised_window).T 
            normalised_data.append(normalised_window)
        return np.array(normalised_data)