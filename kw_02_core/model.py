import os
import math
import numpy as np
import datetime as dt
from numpy import newaxis
from kw_02_core.utils import Timer
from keras.layers import Dropout, LSTM, Conv2D, MaxPool2D, Flatten, SimpleRNN
from keras.models import Sequential, load_model
from keras import Model as K_Model
from keras.callbacks import ModelCheckpoint
import keras
from keras.layers import Dense
from keras.models import Model
import keras.backend as K


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
		self.model = Sequential()
			# 利用GPU来并行训练bitch_size中的样本
	def load_model(self, model_path):
		# Load the model using custom_object_scope
		with keras.utils.custom_object_scope({'custom_loss_with_penalty': custom_loss_with_penalty}):
			print('[Model] Loading model from file %s' % model_path)
			self.model = load_model(model_path)

	def build_model(self,  configs):
		timer = Timer()
		timer.start()

		for layer in configs['model']['layers']:
			neurons = layer['neurons'] if 'neurons' in layer else None
			dropout_rate = layer['rate'] if 'rate' in layer else None
			activation = layer['activation'] if 'activation' in layer else None
			return_seq = layer['return_seq'] if 'return_seq' in layer else None
			input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
			input_dim = layer['input_dim'] if 'input_dim' in layer else None
			name = layer['name'] if 'name' in layer else None

			if layer['type'] == 'rnn':
				self.model.add(SimpleRNN(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq, activation=activation, name=name))
			if layer['type'] == 'conv2d':
				self.model.add(Conv2D(kernel_size=(3, 3),strides=(1, 1),padding='same', data_format='channels_last', activation='relu'))
			if layer['type'] == 'maxpool':
				self.model.add(MaxPool2D(pool_size=(2, 2)))
			if layer['type'] == 'flatten':
				self.model.add(Flatten())
			if layer['type'] == 'dense':
				self.model.add(Dense(neurons, input_shape=(input_timesteps, input_dim), activation=activation, name=name))
			if layer['type'] == 'lstm':
				self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq, activation=activation, name=name))
			if layer['type'] == 'dropout':
				self.model.add(Dropout(dropout_rate, name=name))

		# model = Model(inputs=[input_layer], outputs=[output])
		if configs["model"]["loss"] == "self_loss":
			self.model.compile(loss= custom_loss_with_penalty, optimizer=configs['model']['optimizer'], metrics=configs['model']['metrics'], run_eagerly=True)
		else:
			self.model.compile(loss=configs["model"]["loss"], optimizer=configs['model']['optimizer'], metrics=configs['model']['metrics'],
						   run_eagerly=True)
		# optimizers.Adam()————》其大概的思想是开始的学习率设置为一个较大的值，然后根据次数的增多，动态的减小学习率，以实现效率和效果的兼得。

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
			ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True,verbose=1,mode='auto')
		]
		self.history = self.model.fit(
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

	def predict_point_by_point(self, data,debug=True):
		if debug == False:
			print('[Model] Predicting Point-by-Point...')
			predicted = self.model.predict(data)
			predicted = np.reshape(predicted, (predicted.size,))
		else:
			print('[Model] Predicting Point-by-Point...')
			print (np.array(data).shape)
			predicted = self.model.predict(data)
			print (np.array(predicted).shape)
			predicted = np.reshape(predicted, (predicted.size,))
			print (np.array(predicted).shape)
		return predicted

	def predict_sequences_multiple(self, data, window_size, prediction_len,debug=False):
		if debug == False:
			print('[Model] Predicting Sequences Multiple...')
			prediction_seqs = []
			for i in range(int(len(data)/prediction_len)):
				curr_frame = data[i*prediction_len]
				predicted = []
				for j in range(prediction_len):
					predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
					curr_frame = curr_frame[1:]
					curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
				prediction_seqs.append(predicted)
			return prediction_seqs
		else:
			print('[Model] Predicting Sequences Multiple...')
			prediction_seqs = []
			for i in range(int(len(data)/prediction_len)):
				print(data.shape)
				curr_frame = data[i*prediction_len]
				print(curr_frame)
				predicted = []
				for j in range(prediction_len):
					predict_result = self.model.predict(curr_frame[newaxis,:,:])
					print(predict_result)
					final_result = predict_result[0,0]
					predicted.append(final_result)
					curr_frame = curr_frame[1:]
					print(curr_frame)
					curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
					print(curr_frame)
				prediction_seqs.append(predicted)

	def predict_sequence_full(self, data, window_size):
		print('[Model] Predicting Sequences Full...')
		curr_frame = data[0]
		predicted = []
		for i in range(len(data)):
			predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
			curr_frame = curr_frame[1:]
			curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
		return predicted

