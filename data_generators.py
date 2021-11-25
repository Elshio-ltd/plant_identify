#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 15:47:36 2021

@author: bipin
"""
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from data_config import *
from env_conf import *
import shutil
import pickle
from PIL import Image

class sparse_data_gen(keras.utils.Sequence):
	def __init__(self,
				 data_dir,
				 sparse_class_idx 	= [],
				 dense_class_idx 	= [],
				 num_sparse_class 	= 100,
				 class_list 			= class_list,
				 num_class 			= 100,
				 img_size 			= (224,224)):
		self.class_list 			= class_list
		self.data_dir 			= data_dir
		self.sparse_classes 		= sparse_class_idx
		self.dense_classes 		= dense_class_idx
		self.num_sparse_class 	= num_sparse_class
		self.num_class 			= num_class
		self.img_size 			= img_size

		self.fn_dict 			= {}
		self.path_list 			= []
		self.feat_list 			= []
		self.idx_list 			= []

		self.load_dir()

	def load_dir(self):
		self.class_list1 	= os.listdir(self.data_dir)
		self.class_list1 	= list(filter(lambda x : os.path.isdir(os.path.join(self.data_dir,x)),self.class_list1))
		for class_name1 in self.class_list1:
			fn_list 		= os.listdir(os.path.join(self.data_dir,class_name1))
			fn_list 		= list(filter(lambda x: os.path.isfile(os.path.join(self.data_dir,class_name1,x)),fn_list))
			self.fn_dict[class_name1] = fn_list

	def load_class(self):
		self.num_dense_samples 	= 0
		self.num_sparse_samples = 0
		self.path_list 			= []
		self.feat_list 			= []
		for class1_idx in self.dense_classes:
			class1 = self.class_list[class1_idx]
			if class1 in self.fn_dict:
				self.num_dense_samples += len(self.fn_dict[class1])
				self.path_list.extend(list(map(lambda x: os.path.join(self.data_dir,class1,x),self.fn_dict[class1])))
				self.feat_list.extend([class1_idx]*len(self.fn_dict[class1]))
				print("{0} : {1}".format(class1,len(self.fn_dict[class1])))
			else:
				print("No images for {0}".format(class1))

		for class1_idx in self.sparse_classes:
			class1 = self.class_list[class1_idx]
			if class1 in self.fn_dict:
				self.num_sparse_samples += len(self.fn_dict[class1])
				self.path_list.extend(list(map(lambda x: os.path.join(self.data_dir,class1,x),self.fn_dict[class1])))
				self.feat_list.extend([class1_idx]*len(self.fn_dict[class1]))

		self.idx_list = list(range(self.num_dense_samples))
		np.random.shuffle(self.idx_list)
		sparse_idx_list 	= list(range(self.num_dense_samples,self.num_dense_samples+self.num_sparse_samples))
		np.random.shuffle(sparse_idx_list)

		self.idx_list += sparse_idx_list
		self.idx_list = self.idx_list[:self.num_dense_samples+min(self.num_sparse_class,self.num_sparse_samples)]
		np.random.shuffle(self.idx_list)

	def __len__(self):
		return len(self.idx_list)

	def print_fn(self):
		for idx1 in sorted(self.idx_list):
			print("{0} :\t {1}".format(self.path_list[idx1],self.feat_list[idx1]))

	def load_data(self,idx2):
		image_data 		= Image.open(self.path_list[idx2])
		target_vect 	 	= keras.utils.to_categorical(self.feat_list[idx2],self.num_class)
		target_vect 		= np.array([target_vect])
		image_data  		= np.array(image_data.resize(self.img_size))/255
		image_data  		= np.array([image_data])
		return image_data,target_vect


	def __getitem__(self,loop_idx1):
		idx2 			= self.idx_list[loop_idx1]
		image_data,target_vect = self.load_data(idx2)
		err_cnt = 0
		while image_data.shape != (1, 224, 224, 3) and err_cnt < 5:
			print("Error with {0} {1}".format(self.path_list[idx2],image_data.shape),flush=True)
			err_cnt += 1
			image_data,target_vect = self.load_data(idx2+err_cnt)
		return	 image_data,target_vect

	def on_epoch_end(self):
		np.random.shuffle(self.idx_list)


class dense_data_gen(sparse_data_gen):
	def __init__(self,
				 data_dir,
				 dense_class_idx 	= [],
				 class_list 			= class_list,
				 num_class 			= 100,
				 img_size 			= (224,224)):
		self.class_list 			= class_list
		self.data_dir 			= data_dir
		self.dense_classes 		= dense_class_idx
		self.num_class 			= num_class
		self.img_size 			= img_size

		self.fn_dict 			= {}
		self.path_list 			= []
		self.feat_list 			= []
		self.idx_list 			= []

		self.load_dir()

	def load_class(self):
		self.num_dense_samples 	= 0
		self.path_list 			= []
		self.feat_list 			= []
		for class1_idx in self.dense_classes:
			class1 = self.class_list[class1_idx]
			if class1 in self.fn_dict:
				self.num_dense_samples += len(self.fn_dict[class1])
				self.path_list.extend(list(map(lambda x: os.path.join(self.data_dir,class1,x),self.fn_dict[class1])))
				self.feat_list.extend([class1_idx]*len(self.fn_dict[class1]))
				print("{0} : {1}".format(class1,len(self.fn_dict[class1])))
			else:
				print("No images for {0}".format(class1))

		self.idx_list = list(range(self.num_dense_samples))
		np.random.shuffle(self.idx_list)


class uniform_data_gen(sparse_data_gen):
	def __init__(self,
				 data_dir,
				 dense_class_idx 	= [],
				 class_list 			= class_list,
				 samples_per_class 	= 2,
				 num_class 			= 100,
				 img_size 			= (224,224)):
		self.class_list 			= class_list
		self.data_dir 			= data_dir
		self.dense_classes 		= dense_class_idx
		self.num_class 			= num_class
		self.img_size 			= img_size
		self.samples_per_class 	= samples_per_class

		self.fn_dict 			= {}
		self.path_list 			= []
		self.feat_list 			= []
		self.idx_list 			= []
		self.species_cnt_list 	= [0]

		self.load_dir()

	def load_class(self):
		self.num_dense_samples 	= 0
		self.path_list 			= []
		self.feat_list 			= []
		for class1_idx in self.dense_classes:
			class1 = self.class_list[class1_idx]
			if class1 in self.fn_dict:
				self.num_dense_samples += len(self.fn_dict[class1])
				self.path_list.extend(list(map(lambda x: os.path.join(self.data_dir,class1,x),self.fn_dict[class1])))
				self.feat_list.extend([class1_idx]*len(self.fn_dict[class1]))
				self.species_cnt_list.append(self.num_dense_samples)
				if len(self.fn_dict[class1]) == 0:
					print("{0} : {1}".format(class1,len(self.fn_dict[class1])))
			else:
				print("No images for {0}".format(class1))

		self.idx_list = []
		for idx1 in range(len(self.species_cnt_list)-1):
			list1 = list(range(self.species_cnt_list[idx1],self.species_cnt_list[idx1+1]))
			np.random.shuffle(list1)
			self.idx_list.extend(list1[:self.samples_per_class])
		np.random.shuffle(self.idx_list)

	def on_epoch_end(self):
		self.idx_list = []
		for idx1 in range(len(self.species_cnt_list)-1):
			list1 = list(range(self.species_cnt_list[idx1],self.species_cnt_list[idx1+1]))
			np.random.shuffle(list1)
			self.idx_list.extend(list1[:self.samples_per_class])
		np.random.shuffle(self.idx_list)



class rolling_data_gen(sparse_data_gen):
	def __init__(self,
				 data_dir,
				 samples_per_epoch 	= 1,
				 batch_size 			= 1,
				 class_list 			= class_list,
				 img_size 			= (224,224),
				 initial_epoch 		= 0):
		self.class_list 			= class_list
		self.data_dir 			= data_dir
		self.num_class 			= len(self.class_list)
		self.img_size 			= img_size
		self.samples_per_epoch 	= samples_per_epoch
		self.epoch_count 		= 0
		self.batch_size 			= batch_size
		self.initial_epoch 		= initial_epoch

		self.fn_dict 			= {}
		self.load_dir()

	def load_class(self):
		self.class_cnt = 0
		self.feat_list = []
		self.path_list 	= []
		for class_dx in range(len(self.class_list)):
			class1 = self.class_list[class_dx]
			if class1 in self.fn_dict:
				fn_list 		= list(map(lambda x: os.path.join(self.data_dir,class1,x),self.fn_dict[class1]))
				np.random.shuffle(fn_list)
				if len(fn_list) >0:
					self.class_cnt+=1
					self.feat_list.append(class_dx)
					self.path_list.append(fn_list)
		self.idx_list = list(range(self.class_cnt))
		print("class with data {0}/{1}".format(self.class_cnt,len(self.class_list)))
		np.random.shuffle(self.idx_list)

	def on_epoch_end(self):
		self.epoch_count+=1
		np.random.shuffle(self.idx_list)


	def load_data(self,idx2):
		num_files 		= len( self.path_list[idx2])
		image_fn 		= self.path_list[idx2][(self.epoch_count+self.initial_epoch)%num_files]
		target_vect 	 	= keras.utils.to_categorical(self.feat_list[idx2],self.num_class)
		image_data 		= Image.open(image_fn)
		image_data  		= np.array(image_data.resize(self.img_size))/255

		if image_data.shape[2] == 4:
			image_data  		= image_data[:,:,:3]
		return image_data,target_vect

	def __getitem__(self,loop_idx1):
		image_data 	= []
		target_vect 	= []
		idx1  	= loop_idx1*self.batch_size
		idx2 	= min((loop_idx1+1)*self.batch_size,len(self.idx_list))
		for idx3 in range(idx1,idx2):
			idx4 			= self.idx_list[idx3]
			image_data1,target_vect1 = self.load_data(idx4)
			image_data.append(image_data1)
			target_vect.append(target_vect1)
		
		target_vect 		= np.array(target_vect)
		image_data  		= np.array(image_data)
		return	 image_data,target_vect



if __name__ == "__main__":
	generator1 = sparse_data_gen(train_data_dir)
	generator1.dense_classes 	= [22]
	generator1.sparse_classes 	=  [21,23]
	generator1.num_class = 30
	generator1.load_class()
	img1,vect1 = generator1[0]
	#generator1.print_fn()

	generator2 = dense_data_gen(train_data_dir)
	generator2.dense_classes 	= [21,22,23]
	generator2.num_class = 30
	generator2.load_class()
	img2,vect2 = generator2[0]
	#generator2.print_fn()

	generator3 = uniform_data_gen(train_data_dir)
	generator3.dense_classes 	= [21,22,23]
	generator3.num_class = 30
	generator3.load_class()
	img3,vect3 = generator3[0]
	generator3.print_fn()

	generator4 = rolling_data_gen(train_data_dir)
	generator4.dense_classes 	= [21,22,23]
	generator4.num_class = 30
	generator4.load_class()
	img4,vect4 = generator4[0]
