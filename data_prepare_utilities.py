#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 10:07:19 2021

@author: bipin
"""

import os
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img,DirectoryIterator
import json
import numpy as np
from data_config import *
from env_conf import *
import re
import time
import multiprocessing

'''
reserve_data_dir = "data/Crops_detection/reserve"
test_data_dir = "data/Crops_detection/Test"
train_data_dir = "data/Crops_detection/Train"
'''
png_image_dir = "Crops_detection/png_images"
new_species_fn = "new_species.txt"
error_list_fn = "Read_error.txt"
species_strength_fn = os.path.join(log_dir,"Species_strength.txt")
species_strength_fn2 = os.path.join(log_dir,"Species_strength_test.txt")
Num_files_fn1 = os.path.join(log_dir,"Num_files_reserve.txt")
Num_files_fn2 = os.path.join(log_dir,"Num_files_train.txt")
Num_files_fn3 = os.path.join(log_dir,"Num_files_test.txt")



def rename_space_dirs(base_dir):
	print("Clean naming {0}".format(base_dir))
	folder_list = os.listdir(base_dir)
	folder_list = list(filter(lambda x: os.path.isdir(os.path.join(base_dir,x)),folder_list))
	if len(folder_list) > 0:
		for f1 in folder_list:
			src_fn	  = os.path.join(base_dir,f1)
			f2  = get_proper_name(f1)
			f2  = f2.replace(" ", "_")
			dest_fn = os.path.join(base_dir,f2)
			if src_fn != dest_fn:
				if os.path.exists(dest_fn):
					if len(os.listdir(dest_fn)) == 0:
						shutil.rmtree(dest_fn)
						shutil.move(src_fn,dest_fn)
					else:
						if len(os.listdir(src_fn)) == 0:
							shutil.rmtree(src_fn)
				else:
					shutil.move(src_fn,dest_fn)

#if __name__ == "__main__":
#	base_dir = train_data_dir
def rename_space_files(base_dir):
	folder_list = os.listdir(base_dir)
	for d1 in folder_list:
		file_list1 = os.listdir(os.path.join(base_dir,d1))
		non_file_list = list(filter(lambda x: os.path.isdir(os.path.join(base_dir,d1,x)),file_list1))
		file_list = list(filter(lambda x:os.path.isfile(os.path.join(base_dir,d1,x)),file_list1))
		for f1 in file_list:
			f1_split = f1.split()
			f1_split = list(filter(lambda x:len(x) > 1,f1_split))
			if len(f1_split) > 1:
				f1_new = "_".join(f1_split)
				print(f1_new)
				src_fn = os.path.join(base_dir,d1,f1)
				dest_fn  = os.path.join(base_dir,d1,f1_new)
				if not os.path.exists(dest_fn):
					shutil.move(src_fn,dest_fn)
				else:
					print("file already exixts {0}".format(dest_fn))
		for f1 in non_file_list:
			src_fn = os.path.join(base_dir,d1,f1)
			shutil.rmtree(src_fn)
			print("removed {0}".format(src_fn))


def make_missing_dir(dir_name,class_list):
	folder_list = os.listdir(dir_name)
	missing_dir = set(class_list).difference(set(folder_list))
	if len(missing_dir)>0:
		print(dir_name)
		for f1 in missing_dir:
			print("created folder :{0}".format(f1))
			os.mkdir(os.path.join(dir_name,f1))

def report_num_images(dir_name,class_list,species_strength_fn):
	folder_list = os.listdir(dir_name)
	low_image_count	 = 0
	num_images_dict = {}
	for f1 in class_list:
		if f1 in folder_list :
			num_files = len(os.listdir(os.path.join(dir_name,f1)))
			num_images_dict[f1] = num_files
			if num_files < 10:
				print("{0}\t:{1}".format(f1,num_files))
				print("_"*20)
				low_image_count+=1
		else:
			print("Missing folder \t:{}".format(f1))
	open(species_strength_fn,'w').write(json.dumps(num_images_dict,indent=4))
	print("Folders with low count :{0}".format(low_image_count))

def report_new_species(dir_name,class_list):
	folder_list = os.listdir(dir_name)
	new_species_list = []
	for f1 in folder_list:
		if f1 not in class_list:
			if len(os.listdir(os.path.join(dir_name,f1))) > 5:
				new_species_list.append(f1)
	new_species_list = list(map(lambda x:"'{0}'".format(x),new_species_list))
	new_species_list.sort()
	open(new_species_fn,'w').write(",\n".join(new_species_list))


def prepare_test_data(class_list,num_test_images):
	for f1 in class_list:
		test_dir	= os.path.join(test_data_dir,f1)
		if os.path.exists(test_dir):
			fn_list1  = os.listdir(test_dir)
			np.random.shuffle(fn_list1)
		else:
			os.mkdir(test_dir)
			fn_list1  = os.listdir(test_dir)
		if len(fn_list1) < num_test_images:
			data_dir = os.path.join(reserve_data_dir,f1)
			fn_list2  = os.listdir(data_dir)
			num_images_to_copy = min(num_test_images-len(fn_list1),len(fn_list2))
			if num_images_to_copy >0:
				#print("preparing test data :{0}".format(f1))
				pass
			for idx1 in range(num_images_to_copy):
				fn2 = fn_list2[idx1]
				src_fn = os.path.join(data_dir,fn2)
				dest_fn  = os.path.join(test_dir,fn2)
				shutil.move(src_fn,dest_fn)
	os.system('sync')


def prepare_train_data(class_list,num_train_images):
	for f1 in class_list:
		train_dir	= os.path.join(train_data_dir,f1)
		if os.path.exists(train_dir):
			fn_list1  = os.listdir(train_dir)
			np.random.shuffle(fn_list1)
		else:
			os.mkdir(train_dir)
			fn_list1  = os.listdir(train_dir)
		if len(fn_list1) < num_train_images:
			data_dir = os.path.join(reserve_data_dir,f1)
			fn_list2  = os.listdir(data_dir)
			num_images_to_copy = min(num_train_images-len(fn_list1),len(fn_list2))
			if num_images_to_copy >0:
				#print("preparing train data :{0}".format(f1))
				pass
			for idx1 in range(num_images_to_copy):
				fn2 = fn_list2[idx1]
				src_fn = os.path.join(data_dir,fn2)
				dest_fn  = os.path.join(train_dir,fn2)
				shutil.move(src_fn,dest_fn)
	os.system('sync')

def delete_error_files(data_dir1,
					   plant_name_list):
	train_ds		= ImageDataGenerator(rescale=1/255)
	train_data_src  = DirectoryIterator(data_dir1,
										train_ds,target_size=(224,224),
										shuffle=False,
										batch_size = 1,
										class_mode = 'categorical',
										classes = plant_name_list
										)
	'''
	def check_image(idx1):
		try:
			data1 = train_data_src[idx1]
		except Exception as e:
			fn1 = train_data_src.filenames[idx1]
			os.remove(fn1)

	pool1 = multiprocessing.Pool(10)
	pool1.map(check_image,range(train_data_src.samples))

	'''
	for idx1 in range(train_data_src.samples):
		try:
			data1 = train_data_src[idx1]
		except Exception as e:
			fn1 = train_data_src.filenames[idx1]
			try:
				os.remove(fn1)
				print("Deteted file {0}".format(fn1))
			except:
				print("unable to remove {0}".format(fn1))
	os.system('sync')


def test_dataset(train_data_dir1,
				 class_list = class_list):
	train_ds		= ImageDataGenerator(rescale=1/255)
	train_data_src  = DirectoryIterator(train_data_dir1,
										train_ds,target_size=(224,224),
										shuffle=False,
										batch_size = 1,
										class_mode = 'categorical',
										classes = class_list
										)
	error_fn_list =[]
	for idx1 in range(train_data_src.samples):
		try:
			data1 = train_data_src[idx1]
		except Exception as e:
			print("Error reading data :{0}".format(train_data_src.filenames[idx1]))
			error_fn_list.append(train_data_src.filenames[idx1])
			print(e)
	open(error_list_fn,'w').write("\n".join(error_fn_list))

def move_error_files(error_list_fn,
					 src_dir,
					 target_dir):
	if not os.path.exists(target_dir):
		os.mkdir(target_dir)
	fn_list = open(error_list_fn,'r').readlines()
	fn_list = list(map(lambda x:x.strip("\n"),fn_list))
	for f1 in fn_list:
		src_fn = os.path.join(src_dir,f1)
		dest_fn = os.path.join(target_dir,f1)
		dir1 = os.path.split(f1)[0]
		if not os.path.exists(os.path.join(target_dir,dir1)):
			os.mkdir(os.path.join(target_dir,dir1))
		if os.path.exists(src_fn):
			if os.path.exists(dest_fn):
				print("File already present {0}".format(f1))
				os.remove(src_fn)
			else:
				shutil.move(src_fn,dest_fn)
				print("Moved {0}".format(f1))
		else:
			print("file not exist {0}".format(src_fn))
	os.system('sync')

def move_all_to_reserve(train_data_dir1):
	train_dir_list	  = os.listdir(train_data_dir1)
	for dir1 in train_dir_list:
		fn_list		 = os.listdir(os.path.join(train_data_dir1,dir1))
		for fn1 in fn_list:
			src_fn	  =   os.path.join(train_data_dir1,dir1,fn1)
			dest_fn	 = os.path.join(reserve_data_dir,dir1,fn1)
			if not os.path.exists(os.path.join(reserve_data_dir,dir1)):
				os.mkdir(os.path.join(reserve_data_dir,dir1))
			shutil.move(src_fn,dest_fn)
	os.system('sync')

def move_all_data(src_data_dir1,
				  dest_data_dir1
				  ):
	src_dir_list	  = os.listdir(src_data_dir1)
	for dir1 in src_dir_list:
		fn_list		 = os.listdir(os.path.join(src_data_dir1,dir1))
		for fn1 in fn_list:
			src_fn	  =   os.path.join(src_data_dir1,dir1,fn1)
			dest_fn	 = 	os.path.join(dest_data_dir1,dir1,fn1)
			if not os.path.exists(os.path.join(dest_data_dir1,dir1)):
				os.mkdir(os.path.join(dest_data_dir1,dir1))
			shutil.move(src_fn,dest_fn)
	os.system('sync')

def remove_null_dirs(dir1):
	folder_list = os.listdir(dir1)
	for dir2 in folder_list:
		if len(os.listdir(os.path.join(dir1,dir2))) == 0:
			shutil.rmtree(os.path.join(dir1,dir2))

def report_num_images_folder(dir_name,report_fn):
	dir_list	= os.listdir(dir_name)
	dir_list	= list(filter(lambda x: os.path.isdir(os.path.join(dir_name,x)),dir_list))
	out_file	= open(report_fn,'w')
	for dir1 in dir_list:
		num_files = len(os.listdir(os.path.join(dir_name,dir1)))
		out_file.write("{0}\t:{1}\n".format(dir1,num_files))

if __name__ == "__main1__":
	remove_null_dirs(train_data_dir)
	remove_null_dirs(test_data_dir)
	remove_null_dirs(reserve_data_dir)

if __name__ == "__main1__":
	folder_list	 = os.listdir(reserve_data_dir)
	for folder1 in folder_list:
		if len(folder1) <= 2:
			print(folder1)
			cmd1 = "mv {0}/* {1}".format(os.path.join(reserve_data_dir,folder1),reserve_data_dir)
			os.system(cmd1)
			cmd2 = "rmdir {0}".format(os.path.join(reserve_data_dir,folder1))
			os.system(cmd2)

if __name__ == "__main__":
	rename_space_dirs(train_data_dir)
	rename_space_dirs(test_data_dir)
	rename_space_dirs(reserve_data_dir)
	rename_space_files(train_data_dir)
	rename_space_files(test_data_dir)
	rename_space_files(reserve_data_dir)

if __name__ == "__main__":
	num_test_images	 = 5
	num_train_images	= 100

	reserve_folder_list = os.listdir(reserve_data_dir)
	test_folder_list	= os.listdir(test_data_dir)
	train_folder_list   = os.listdir(train_data_dir)

	report_num_images(train_data_dir,class_list,species_strength_fn)
	make_missing_dir(train_data_dir,class_list)
	make_missing_dir(test_data_dir,class_list)
	make_missing_dir(reserve_data_dir,class_list)
	pre_trained_class   = class_list_prefix
	current_train_class =[]
	for class1 in class_list:
		if class1 not in pre_trained_class:
			current_train_class.append(class1)

if __name__ == "__main1__":
	move_all_to_reserve(train_data_dir)
	#move_all_to_reserve(test_data_dir)
	report_new_species(reserve_data_dir,class_list)
	time.sleep(1)

if __name__ == "__main__":
	prepare_test_data(class_list,num_test_images)
	prepare_train_data(current_train_class,100)
	#prepare_train_data(pre_trained_class,10)


if __name__ == "__main__":
	test_dataset(train_data_dir)
	move_error_files(error_list_fn,
					 train_data_dir,
					 error_data_dir)



if __name__ == "__main__":
	test_dataset(test_data_dir)
	move_error_files(error_list_fn,
					 test_data_dir,
					 error_data_dir)

if __name__ == "__main1__":
	test_dataset(reserve_data_dir)
	move_error_files(error_list_fn,
					 reserve_data_dir,
					 error_data_dir)

if __name__ =="__main__":
	report_num_images(train_data_dir,class_list,species_strength_fn)
	report_num_images(test_data_dir,class_list,species_strength_fn2)
	report_num_images_folder(reserve_data_dir,Num_files_fn1)
	report_num_images_folder(train_data_dir,Num_files_fn2)
	report_num_images_folder(test_data_dir,Num_files_fn3)


if __name__ == "__main1__":
	temp_data_dir 	= r"data/Crops_detection_b1"
	temp_train_dir  = os.path.join(temp_data_dir,"Train")