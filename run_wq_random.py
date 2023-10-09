import os
import json
import time
import math
import matplotlib.pyplot as plt
import numpy as np
from kw_02_core.data_processor import DataLoader
from kw_02_core.model import Model
from keras.utils import plot_model
import pandas as pd
from sklearn.metrics import *
from sklearn.model_selection import train_test_split


# 绘图展示结果
def plot_results(predicted_data, true_data, title=None, eval_text=None, x_ticks=None):
    plt.rc('font', family='Microsoft YaHei')
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    # [x+1 for x in range(len(predicted_data))],
    plt.plot(true_data, color='b', label='True Data')  # true_data[8:]→序列的最开始8个数据不作为预测输出，相对应的真值应该从第九个开始（索引8）
    plt.plot(predicted_data, color='r', label='Prediction')
    if title != None:
        plt.title(title)
    # if eval_text != None:
    #     plt.text(len(predicted_data), max(true_data) + 0.11, eval_text, ha='center',
    #              fontsize=6, rotation=0, c='y', alpha=1)
    if x_ticks.all() != None:
        # 创建需要做x刻度标签的下标索引
        tic = []
        t_l = []
        print(x_ticks.shape)
        for i in range(len(x_ticks)):
            if i % 7 == 0:
                tic.append(x_ticks[i])  # 设置绘图时的横坐标标签→日期
                t_l.append(i)
        plt.xticks(t_l, t_l, rotation=30)
    plt.legend()
    plt.savefig('./kw_04_outcome/%s.png' % title.split('(')[0])
    plt.close()


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')

    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        ax.plot(padding + data, label='Prediction')
    plt.legend()
    plt.show()


# RNN时间序列
def main():
    # 读取所需参数
    configs = json.load(open('config.json', 'r', encoding='utf-8', errors='ignore'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])
    # 读取数据
    data = DataLoader(
        '.\kw_01_data/00_data_use\data_rs02_wq_all.txt',
        configs['data']['train_test_split'],
        configs['data']['columns']
    )

    # 创建RNN模型
    model_path = '.\kw_03_saved_models/'
    model_file = input('请输入选用的模型文件名/训练新模型（请输入new）：')
    if model_file == 'new':
        model = Model()
        mymodel = model.build_model(configs)
        plot_model(mymodel, to_file='model.png', show_shapes=True)
    else:
        model = Model()
        model.load_model(model_path + model_file)

    # 加载训练数据
    x, y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )
    print(x.shape)
    print(y.shape)

    # 训练模型
    model.train(
        x,
        y,
        epochs=configs['training']['epochs'],
        batch_size=configs['training']['batch_size'],
        save_dir=configs['model']['save_dir']
    )

    # TODO  增加验证集，early stopping，防止过拟合
    # # early stoppping
    # from keras.callbacks import EarlyStopping
    # early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2)
    # # 训练
    # history = model.fit(train_X, train_y, epochs=300, batch_size=20, validation_data=(test_X, test_y), verbose=2,
    #                     shuffle=False, callbacks=[early_stopping])
    # 测试结果
    x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )

    # 展示测试效果
    # 1、基于序列的预测1-50→51+2-50→52
    # predictions = model.predict_sequences_multiple(x_test, configs['data']['sequence_length'],
    #                                                configs['data']['sequence_length'], debug=False)
    # print(np.array(predictions).shape)
    #
    # plot_results_multiple(predictions, y_test, configs['data']['sequence_length'])
    # 2、基于点的预测1-50→51、2-51（实际）→52
    predictions_point = model.predict_point_by_point(x_test, debug=True)
    predictions_point = np.array(predictions_point).reshape(len(predictions_point), 1)
    print(y_test.shape)

    plot_results(predictions_point, y_test)


def all_cs(data_0, cs_names, configs):
    """
    传入原数据，按照split分割各断面训练集整合→输入
    :param data_0: 原数据
    :param cs_names: 断面名称
    :param configs: 模型配置
    :return: 训练集的x，y
    """
    x_l = []  # 从各个测站分割训练集特征值汇集
    y_l = []  # 从各个测站分割训练集目标值汇集
    for name in cs_names:
        data_cs = data_0[data_0['cs_name'] == name][configs['data']['columns']]  # 依次提取各站点数据，并筛选出各波段数值作为模型输入
        data_cs = data_cs.reset_index(drop=True)
        data = DataLoader(
            data_cs,
            configs['data']['train_test_split'],
            configs['data']['columns']
        )
        # 加载训练数据
        x, y = data.get_train_data(
            seq_len=configs['data']['sequence_length'],
            normalise=configs['data']['normalise']
        )
        x_l.append(x)
        y_l.append(y)
        print(x.shape)
        print(y.shape)
    # 将各个测站的训练集的特征值和目标值进行整合————》x,y
    for xi in range(len(x_l)):
        if xi != 0:
            x_l[0] = np.concatenate((x_l[0], x_l[xi]))
    for yi in range(len(y_l)):
        if yi != 0:
            y_l[0] = np.concatenate((y_l[0], y_l[yi]))
    x = x_l[0]
    y = y_l[0]
    return x, y


def cs_cs(data_0, cs_names, configs):
    """
    传入原数据，按照split分割各断面训练集→训练各模型并验证
    :param data_0: 原数据
    :param cs_names: 断面名称
    :param configs: 模型配置
    :return: None
    """
    x_l = []  # 从各个测站分割训练集特征值汇集
    y_l = []  # 从各个测站分割训练集目标值汇集
    for name in cs_names:
        data_cs = data_0[data_0['cs_name'] == name][configs['data']['columns']]  # 依次提取各站点数据，并筛选出各波段数值作为模型输入
        data_cs = data_cs.reset_index(drop=True)
        data = DataLoader(
            data_cs,
            configs['data']['train_test_split'],
            configs['data']['columns']
        )
        # 加载训练数据
        x, y = data.get_train_data(
            seq_len=configs['data']['sequence_length'],
            normalise=configs['data']['normalise']
        )
        # 训练模型
        model.train(
            x,
            y,
            epochs=configs['training']['epochs'],
            batch_size=configs['training']['batch_size'],
            save_dir=configs['model']['save_dir']
        )

        # 测试结果
        x_test, y_test = data.get_test_data(
            seq_len=configs['data']['sequence_length'],
            normalise=configs['data']['normalise']
        )
        # 模型模拟效果评估
        eval = model.model.evaluate(
            x_test,
            y_test,
            batch_size=None,
            verbose="auto",
            sample_weight=None,
            steps=None,
            callbacks=None,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
            return_dict=False,
        )

        # 展示测试效果
        # 2、基于点的预测1-50→51、2-51（实际）→52
        predictions_point = model.predict_point_by_point(x_test, debug=True)
        predictions_point = np.array(predictions_point).reshape(len(predictions_point), 1)
        print(y_test.shape)
        # 配置绘图参数，调用plot_results进行绘图
        eval_metrics = configs['model']['metrics'] + ':' + '{:.3f}'.format(eval[1])  # 评价结果展现mse
        # 建立时间序列的横坐标
        split = configs['data']['train_test_split']
        i_split = int(len(data_0[data_0['cs_name'] == name]) * split)  # 划分比→训练集长度
        x_name = data_0[data_0['cs_name'] == name].get('date_month').values[
                 (i_split + configs['data']['sequence_length']):]  # 划分数据集→测试集
        title = name + '_' + configs['data']['columns'][-1]  # 设置绘图标题和图名
        plot_results(predictions_point, y_test, title, eval_metrics, x_name)


def model_evaluation(pre, true, f_name):
    """
    打开/创建文件f_name，对模型进行评估
    :param pre: 模型预测值
    :param true: 对应真实值
    :param f_name: 待写入的文件路径
    :return:
    """
    with open(f_name, 'a') as f:
        f.write('mean_absolute_error:   %3f \n' % mean_absolute_error(true, pre))
        f.write('mean_squared_error:    %3f \n' % mean_squared_error(true, pre))
        f.write('median_absolute_error: %3f \n' % median_absolute_error(true, pre))
        f.write('mean_absolute_percentage_error:    %3f \n' % mean_absolute_percentage_error(true, pre))
        f.write('r2_score:  %3f \n' % r2_score(true, pre))
        f.write('Nash-Sutcliffe Efficiency:  %3f \n' % nse(true, pre))
        f.write('Relative error:  %3f \n' % re(true, pre))


def nse(targets, predictions):
    predictions = np.array(predictions)
    targets = np.array(targets)
    return 1 - (np.sum((predictions - targets) ** 2) / np.sum((targets - np.mean(targets)) ** 2))


def re(targets, predictions):
    predictions = np.array(predictions)
    targets = np.array(targets)
    return (np.sum((predictions - targets) / targets)) / len(predictions)




if __name__ == '__main__':
    # main()
    # for wq in ['溶解氧(mg/L)', '化学需氧量(mg/L) ', '高锰酸盐指数(mg/L)', '生化需氧量(mg/L)', '氨氮(mg/L)', '总磷(mg/L)']:
    # 分站点训练
    # 打开数据，分别提取各站点波段数据进行模拟
    configs = json.load(open('config_BP.json', 'r', encoding='utf-8', errors='ignore'))  # 加载模拟配置json
    # 设置要进行模拟的站名
    #'温榆河北关闸', '北运河东关大桥','玉带河玉带河入凉水河口','凉水河许各庄','北运河榆林庄'
    cs_names = ['凉水河许各庄']
    # 设置采用的模型文件名
    model_use = '2_1.h5'
    # 打开数据，以config_verif_cs.json为配置→dataframe
    data_0 = pd.read_table('.\kw_01_data/00_data_use\data_rs02_wq_all_25D.txt')
    # 创建RNN模型
    model_path = '.\kw_03_saved_models/'
    model_file = input('请输入选用的模型文件名/训练新模型（请输入 new）：')
    if model_file == 'new':
        model = Model()
        mymodel = model.build_model(configs)
        plot_model(mymodel, to_file='model.png', show_shapes=True)
    else:
        model = Model()
        model.load_model(model_path + model_file)

    for name in cs_names:
        # choose_method = input('请输入要选择的模拟方式（1、all→cs  2、cs→cs  3、结束）：')
        # cs_cs(data_0, cs_names, configs)
        # 加载训练集数据
        # x, y = all_cs(data_0, cs_names, configs)
        data_cs = data_0[data_0['cs_name'].values == name][
            configs['data']['columns']]  # 依次提取各站点数据，并筛选出各波段数值作为模型输入
        data_cs = data_cs.reset_index(drop=True)
        data_cs = data_cs.apply(lambda m: (m - np.min(m)) / (np.max(m) - np.min(m)))  # 对原数据进行归一化处理
        x = np.array(data_cs[configs['data']['columns'][:-1]]).astype(float)
        y = np.array(data_cs[configs['data']['columns'][-1]]).astype(float)
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=25, test_size=0.3)
        x_train, x_test, y_train, y_test = x_train.reshape(79, 1, 5), x_test.reshape(35, 1, 5), y_train.reshape(79, 1, 1), y_test.reshape(35, 1, 1)
        # 训练模型
        # Train model and get training history
        model.train(
            x_train,
            y_train,
            epochs=configs['training']['epochs'],
            batch_size=configs['training']['batch_size'],
            save_dir=configs['model']['save_dir']
        )


        # Plot training loss versus epoch
        pred_WQI = configs['data']['columns'][-1].split('(')[0]
        loss_file = f'./kw_07_plot/Training loss_{pred_WQI}.txt'
        loss_list = model.history.history['loss']

        with open(loss_file, 'a') as f:
            for loss in loss_list:
                f.write(str(round(loss, 4))+'\n')


        data_cs = data_0[data_0['cs_name'] == name][configs['data']['columns']]  # 依次提取各站点数据，并筛选出各波段数值作为模型输入
        data_cs = data_cs.reset_index(drop=True)
        max = data_cs[configs["data"]["columns"][-1]].max()
        min = data_cs[configs["data"]["columns"][-1]].min()
        """
        data = DataLoader(
            data_cs,
            configs['data']['train_test_split'],
            configs['data']['columns']
            )
        # 测试结果
        x_test, y_test = data.get_test_data(
            seq_len=configs['data']['sequence_length'],
            normalise=configs['data']['normalise']
        )
        """

        # 模型模拟效果评估
        eval = model.model.evaluate(
            x_test,
            y_test,
            batch_size=None,
            verbose="auto",
            sample_weight=None,
            steps=None,
            callbacks=None,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
            return_dict=False,
        )

        # 展示训练集的训练结果
        predictions_train = model.predict_point_by_point(x_train, debug=True)
        predictions_train = np.array(predictions_train).reshape(len(predictions_train), 1)
        y = np.array(y_train).reshape(len(y_train), 1)
        print(y.shape)
        # 配置绘图参数，调用plot_results进行绘图
        eval_metrics = configs['model']['metrics'] + ':' + '{:.3f}'.format(eval[1])  # 评价结果展现mse
        # 建立时间序列的横坐标
        split = configs['data']['train_test_split']
        i_split = int(len(data_0[data_0['cs_name'] == name]) * split)  # 划分比→训练集长度
        x_name = data_0[data_0['cs_name'] == name].get('date_month').values[
                 :(i_split + configs['data']['sequence_length'])]  # 划分数据集→训练集
        title = 'train_' + name + '_' + configs['data']['columns'][-1]  # 设置绘图标题和图名
        # 绘制反归一化后的预测值和真实值
        pre_scale = [min + (max - min) * p for p in predictions_train[:, 0]]
        true_scale = [min + (max - min) * t for t in y]
        plot_results(pre_scale, true_scale, title, eval_metrics, x_name)
        # 写入各种评价指标
        f_name = './kw_04_outcome/%s.txt' % title.split('(')[0]
        model_evaluation(pre_scale, true_scale, f_name)

        # 展示测试效果
        # 2、基于点的预测1-50→51、2-51（实际）→52
        predictions_point = model.predict_point_by_point(x_test, debug=True)
        predictions_point = np.array(predictions_point).reshape(len(predictions_point), 1)
        print(y_test.shape)
        # 配置绘图参数，调用plot_results进行绘图
        eval_metrics = configs['model']['metrics'] + ':' + '{:.3f}'.format(eval[1])  # 评价结果展现mse
        # 建立时间序列的横坐标
        split = configs['data']['train_test_split']
        i_split = int(len(data_0[data_0['cs_name'] == name]) * split)  # 划分比→训练集长度
        x_name = data_0[data_0['cs_name'] == name].get('date_month').values[
                 (i_split + configs['data']['sequence_length']):]  # 划分数据集→测试集
        title = name + '_' + configs['data']['columns'][-1]  # 设置绘图标题和图名
        # 绘制反归一化后的预测值和真实值
        pre_scale = [min + (max - min) * p for p in predictions_point[:, 0]]
        true_scale = [min + (max - min) * t for t in y_test[:, 0, 0]]
        # 写入反归一化后的预测值和真实值
        with open('./kw_04_outcome/true_pre_%s.txt' % title.split('(')[0], 'a') as f:
            f.write('pre_scale' + '\t' + 'true_scale' + '\n')
            for i in range(len(true_scale)):
                f.write(str(round(pre_scale[i], 3)) + '\t' + str(round(true_scale[i], 3)) + '\n')
        plot_results(pre_scale, true_scale, title, eval_metrics, x_name)
        # 写入各种评价指标
        f_name = './kw_04_outcome/%s.txt' % title.split('(')[0]
        model_evaluation(pre_scale, true_scale, f_name)
