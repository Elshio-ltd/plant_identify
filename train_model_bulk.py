#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 11:40:21 2021

@author: bipin
"""

#%% imports
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
from tensorflow import keras
import numpy as np
import tensorflow as tf
from data_config import *
from env_conf import *
from Transfer_model import *
from report_accuracy import *
from data_generators import *
import os
import shutil
import pickle
import time

bulk_model_fn_temp = os.path.join(model_dir,"Bulk_model_{0}_{1}.h5")
training_layers = ['dense']

if __name__ == "__main__":
	print(time.ctime())
	if  os.path.exists(bulk_model_fn):
		model1  = tf.keras.models.load_model(bulk_model_fn)
	else:
		model1 				= tf.keras.models.load_model(model_fn)
		input_layer			= model1.get_layer("input_1")
		final_layer			= model1.get_layer('dense')
		pre_final_layer		= model1.get_layer('flatten')
		softmax				= tf.keras.layers.Softmax(name="softmax")
		new_dense			= keras.layers.Dense(len(class_list),
												 name	   ="dense",
												 activation = keras.activations.linear,
												 use_bias = True
												  )
		output_layer 		= softmax(new_dense(pre_final_layer.output))
		model2		  		= keras.Model(inputs	= model1.input ,
											outputs   = output_layer,
											 )
		model2.save(bulk_model_fn)
		model1 = model2

	data_gen4 = rolling_data_gen(train_data_dir,
							   initial_epoch=8)
	data_gen4.load_class()


	for idx1 in range(50):
		#%% train dense layers
		for layer1 in model1.layers:
			if layer1.name in training_layers:
				layer1.trainable 	= True
			else:
				layer1.trainable 	= False
		model1.compile(optimizer=tf.keras.optimizers.Adam(),
					   loss	 = tf.keras.losses.categorical_crossentropy,
					   metrics=['accuracy']
					   )
		model1.fit(data_gen4,epochs=1,batch_size=12,verbose=2)
		model1.save(bulk_model_fn_temp.format(1,1))
		#%% fine tune
		'''
		for layer1 in model1.layers:
				layer1.trainable 	= True
		model1.compile(
					   optimizer 	= tf.keras.optimizers.RMSprop(learning_rate=1e-7),
					   #optimizer=tf.keras.optimizers.Adam(),
					   loss			= tf.keras.losses.categorical_crossentropy,
					   metrics		= ['accuracy']
					   )
		model1.fit(data_gen4,epochs=2,batch_size=15,verbose=2)
		report_accuracy(model1)
		model1.save(bulk_model_fn_temp.format(2,idx1))
		'''
		model1.save(bulk_model_fn)
