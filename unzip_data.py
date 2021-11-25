#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 14:06:20 2021

@author: bipin
"""

import os
import zipfile
import numpy as np
import shutil

from data_config import *
from env_conf import *
'''
zip_file_dir			= r"data/drive_mnt"
data_dir_root   		= r"data/Crops_detection"
#zip_file_dir	= r"/mnt/data/Data_zip"
#data_dir_root   = r"/mnt/data/ML_data"
'''
zip_file_dir 		= r"/mnt/PHD_data/Projects/Elshio/ML/data/drive_mnt"
data_dir_root 		= fast_dir

data_dir_list   = ["Reserve","Test",]
#data_dir_list   = ["Reserve","Test","Train"]
dest_data_loc   = "Reserve"
min_num_images  = 25
log_fn 			= os.path.join(log_dir,"Unzip_log.txt")

def get_num_images(name1):
	num_images = 0
	for dir1 in data_dir_list:
		full_path1 = os.path.join(data_dir_root,dir1,name1)
		if os.path.exists(full_path1):
			num_images += len(os.listdir(full_path1))
	return num_images


if __name__ == "__main__":
	zip_file_list   = os.listdir(zip_file_dir)
	zip_file_list   = filter(lambda x: x.endswith(".zip"),zip_file_list)
	log_file 		= open(log_fn,'w')
	folder_names	= []
	for folder1 in data_dir_list:
		folder_names.extend(os.listdir(os.path.join(data_dir_root,folder1)))
	folder_names	= list(np.unique(folder_names))
	image_count	 = list(map(get_num_images,folder_names))

	zip_name_list_1 = []
	zip_name_list   = []
	for fn1 in zip_file_list:
		print(fn1)
		zip1				= zipfile.ZipFile(os.path.join(zip_file_dir,fn1),'r')
		name_list1		  = zip1.namelist()
		info_list1		  = zip1.infolist()
		info_list_dir	   = list(filter( lambda x: x.is_dir(),info_list1))

		folder_name_list	 		= list(map(lambda x:x.filename,info_list_dir))
		species_name_list		= list(map(lambda x: x.split("/")[-2],folder_name_list))
		proper_name_list			= list(map(lambda x: get_proper_name(x),species_name_list))
		#proper_name_list 		= list(filter(lambda x: len(x) >3,proper_name_list))

		for idx1 in range(len(proper_name_list)):
			name1		   = proper_name_list[idx1]
			log_file. write( name1 + "\t\t")
			log_file. write( species_name_list[idx1] + "\t\t")
			data_available  = True
			if len(name1) > 2:
				if name1 in folder_names:
					idx2 = folder_names.index(name1)
					available_images 	= image_count[idx2]
					if available_images  == 0 :
						data_available = False
						#data_available = True
					log_file. write( "Available {0}".format(available_images))
				else:
					data_available = False
					log_file. write( "No folder")

			if not data_available:
				dest_path   = os.path.join(data_dir_root, dest_data_loc,name1.replace(" ","_"))
				info_list2  = list(filter(lambda x:  x.filename.split("/")[-2] == species_name_list[idx1],info_list1))
				try:
					zip1.extractall(dest_path,info_list2)
				except:
					print("Unable to extract {0}".format(name1))
					log_file. write("\nUnable to extract {0}\n".format(name1))
				for info1 in info_list2:
					fn1 = info1.filename.split("/")[-1]
					dir1 = os.path.join(dest_path,info1.filename)
					dir2 = os.path.join(dest_path,fn1)
					if not os.path.isdir(dir1):
						try:
							shutil.move(dir1,dir2)
						except:
							print("unable to move extracted {0}".format(name1))
							log_file. write("\nunable to move extracted {0}\n".format(name1))
				dir3 = info_list2[0].filename.split("/")[0]
				shutil.rmtree(os.path.join(dest_path, dir3))
				print("Extracted {1} to {0}".format(dest_path, name1))
			log_file. write( "\n")
		zip1.close()
	log_file.flush()
	log_file.close()