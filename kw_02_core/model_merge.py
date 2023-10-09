import os
import math
import numpy as np
import datetime as dt
from numpy import newaxis
from kw_02_core.utils import Timer
from keras.models import load_model
from keras.models import Model as K_Model
from keras.callbacks import ModelCheckpoint
from keras.layers import *
import keras.backend as K
import keras


# 自定义损失函数，增加对高值的惩罚项
def custom_loss_with_penalty(y_true, y_pred):
    mse_loss = K.mean(K.square(y_true - y_pred))

    # 计算高值惩罚项，可以根据需要进行调整
    high_value_penalty = K.mean(K.sin(math.pi/2*(K.tanh(K.abs(K.exp(y_pred) - K.exp(y_true)))))/K.cos(math.pi/2*(K.tanh(K.abs(K.exp(y_pred) - K.exp(y_true))))))

    # 合并MSE损失和高值惩罚项
    total_loss = mse_loss + high_value_penalty

    return total_loss

class Model(K_Model):
    """LSTM 模型"""
    def __init__(self):
        super(Model, self).__init__()
        self.model = None

    def merge_model(self, m1, m2, configs):
        timer = Timer()
        timer.start()
        inp = Input(shape=(1, 5))
        # 加载模型L1、L2
        # Load the model using custom_object_scope
        with keras.utils.custom_object_scope({'custom_loss_with_penalty': custom_loss_with_penalty}):
            print('[Model] Loading model from file %s' % m1)
            L1 = load_model(m1)
            print('[Model] Loading model from file %s' % m2)
            L2 = load_model(m2)
            # 依次加载L1中各层（除最后的全连接层）   !!!注意各层名称会随着迁移学习进行继承
            x1 = L1.get_layer("dense1_1")(inp)
            x1 = L1.get_layer("drop1_1")(x1)
            x1 = L1.get_layer("dense1_2")(x1)
            x1 = L1.get_layer("drop1_2")(x1)
            # x1 = L1.get_layer("lstm1_3")(x1)
            # x1 = L1.get_layer("drop1_3")(x1)
            # x1 = L1.get_layer("lstm1_4")(x1)

            # 依次加载L2中各层（除最后的全连接层）
            x2 = L2.get_layer("dense2_1")(inp)
            x2 = L2.get_layer("drop2_1")(x2)
            x2 = L2.get_layer("dense2_2")(x2)
            x2 = L2.get_layer("drop2_2")(x2)
            # x2 = L2.get_layer("lstm2_3")(x2)
            # x2 = L2.get_layer("drop2_3")(x2)
            # x2 = L2.get_layer("lstm2_4")(x2)

            # 汇编整合两个模型中LSTM的输出，增加一层全连接层
            x = concatenate([x1, x2], axis=-1)
            # x = Dropout(0.3)(x)
            x = Dense(30, activation='relu', name='dense3_1')(x)
            # x = Dropout(0.3)(x)
            y = Dense(1, activation="relu")(x)
            self.model = K_Model(inputs=inp, outputs=y)
            self.model.summary()  # 打印出模型概况
            # 创建自定义损失函数实例，其中 weight_factor 控制高值惩罚项的强度

        # model = Model(inputs=[input_layer], outputs=[output])
        if configs["model"]["loss"] == "self_loss":
            self.model.compile(loss=custom_loss_with_penalty, optimizer=configs['model']['optimizer'],
                               metrics=configs['model']['metrics'], run_eagerly=True)
        else:
            self.model.compile(loss=configs["model"]["loss"], optimizer=configs['model']['optimizer'],
                               metrics=configs['model']['metrics'],
                               run_eagerly=True)
        # optimizers.Adam()————》其大概的思想是开始的学习率设置为一个较大的值，然后根据次数的增多，动态的减小学习率，以实现效率和效果的兼得

        print('[Model] Model Compiled')
        timer.stop()

        return self.model

    def train(self, x, y, epochs, batch_size, save_dir):
        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))

        save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        callbacks = [
            # EarlyStopping(monitor='loss', patience=5),	# 防止过度拟合的情况
            ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True, verbose=1, mode='auto')
        ]
        self.model.fit(
            x,
            y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            validation_freq=1
        )
        self.model.save(save_fname)

        print('[Model] Training Completed. Model saved as %s' % save_fname)
        timer.stop()

    def train_generator(self, data_gen, epochs, batch_size, steps_per_epoch, save_dir):
        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size, %s batches per epoch' % (epochs, batch_size, steps_per_epoch))

        save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        callbacks = [
            ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)
        ]
        self.model.fit_generator(
            data_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=callbacks,
            workers=1
        )

        print('[Model] Training Completed. Model saved as %s' % save_fname)
        timer.stop()

    def predict_point_by_point(self, data, debug=True):
        if debug == False:
            print('[Model] Predicting Point-by-Point...')
            predicted = self.model.predict(data)
            predicted = np.reshape(predicted, (predicted.size,))
        else:
            print('[Model] Predicting Point-by-Point...')
            print(np.array(data).shape)
            predicted = self.model.predict(data)
            print(np.array(predicted).shape)
            predicted = np.reshape(predicted, (predicted.size,))
            print(np.array(predicted).shape)
        return predicted

    def predict_sequences_multiple(self, data, window_size, prediction_len, debug=False):
        if debug == False:
            print('[Model] Predicting Sequences Multiple...')
            prediction_seqs = []
            for i in range(int(len(data) / prediction_len)):
                curr_frame = data[i * prediction_len]
                predicted = []
                for j in range(prediction_len):
                    predicted.append(self.model.predict(curr_frame[newaxis, :, :])[0, 0])
                    curr_frame = curr_frame[1:]
                    curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
                prediction_seqs.append(predicted)
            return prediction_seqs
        else:
            print('[Model] Predicting Sequences Multiple...')
            prediction_seqs = []
            for i in range(int(len(data) / prediction_len)):
                print(data.shape)
                curr_frame = data[i * prediction_len]
                print(curr_frame)
                predicted = []
                for j in range(prediction_len):
                    predict_result = self.model.predict(curr_frame[newaxis, :, :])
                    print(predict_result)
                    final_result = predict_result[0, 0]
                    predicted.append(final_result)
                    curr_frame = curr_frame[1:]
                    print(curr_frame)
                    curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
                    print(curr_frame)
                prediction_seqs.append(predicted)

    def predict_sequence_full(self, data, window_size):
        print('[Model] Predicting Sequences Full...')
        curr_frame = data[0]
        predicted = []
        for i in range(len(data)):
            predicted.append(self.model.predict(curr_frame[newaxis, :, :])[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
        return predicted

