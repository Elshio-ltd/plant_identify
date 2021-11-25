#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 21:08:45 2021

@author: bipin
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
from env_conf import *
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator,DirectoryIterator
import tensorflow as tf
from data_config import *
from env_conf import *
import json
import pickle
import time


'''
def report_accuracy(model_fn,
					test_data_dir,
					):

if __name__ == "__main__":
'''

def report_accuracy(model,
					class_list = class_list):
	#model		   = tf.keras.models.load_model(bulk_model_fn)
	num_class	 	= model.output_shape[1]
	test_ds		 	= ImageDataGenerator(rescale=1/255)
	class_list1	 	= class_list[:num_class]
	test_data_src   = DirectoryIterator(test_data_dir,
										test_ds,
										target_size=(224,224),
										shuffle=False,
										batch_size = 1,
										class_mode = 'categorical',
										classes = class_list1
										)
	prediction			= model.predict(test_data_src)
	prediction_cat		= np.argmax(prediction,axis=1)
	hist_dict			= {}
	validation_classes  = list(test_data_src.filenames)
	validation_class_names = list(map(lambda x: os.path.split(x)[0],validation_classes))
	expected				= np.array(test_data_src.classes)
	correct_prediction  = prediction_cat == expected
	training_dict		= json.loads(open(species_strength_fn,'r').read())
	total_accuracy		= np.mean(correct_prediction*1.0)
	for class1 in range(num_class):
		dict1			= {"calss id" : class1}
		this_class		= expected==class1
		#accuracy1		= np.sum(np.logical_and(this_class,correct_prediction))/np.sum(this_class)

		true_pos 		= np.sum(np.logical_and(this_class,correct_prediction))
		false_pos 		= np.sum(np.logical_and(prediction_cat==class1,np.logical_not(correct_prediction)))
		false_neg 		= np.sum(this_class) - true_pos
		accuracy1		= 0
		if true_pos > 0:
			accuracy1	= true_pos/(true_pos+false_neg)

		dict1["accuracy"] = accuracy1*100
		dict1["TP"] 		= true_pos
		dict1["FP"] 		= false_pos
		dict1["FN"] 		= false_neg
		indices			= np.where(expected==class1)
		indices			= np.array(indices).flatten()

		if len(indices) > 0:
			name1	   = validation_class_names[indices[0]]
			dict1["name"] = name1
			if name1 in training_dict:
				dict1["samples"] = training_dict[name1]
			else:
				dict1["samples"] = 0
		else:
			dict1["name"] = "--"
			dict1["samples"] = 0
		hist_dict[class1] = dict1

	time_str = time.ctime().replace(" ","_")
	histogram_fn = os.path.join(log_dir,"Accuracy_histogram_{0}_{1}.csv".format(num_class,time_str))
	out_file  = open(histogram_fn,'w')
	out_file.write("Total Accuracy, {0} \n".format(total_accuracy))
	out_file.write("Id, Name, Number of Samples, Accuracy(%), True Positive, False positive, False Negative \n")

	for class1 in range(num_class):
		if class1 in hist_dict:
			accuracy	= hist_dict[class1]["accuracy"]
			name1		= hist_dict[class1]["name"]
			samples	 	= hist_dict[class1]["samples"]
			tp 			= hist_dict[class1]["TP"]
			fp 			= hist_dict[class1]["FP"]
			fn 			= hist_dict[class1]["FN"]
			out_file.write("{0}, {1}, {2}, {3}, {4}, {5}, {6}\n".format(class1,name1,samples,accuracy,tp,fp,fn))
	out_file.flush()
	out_file.close()
	print("Wrote report to {0}".format(histogram_fn))
	confusion			= tf.math.confusion_matrix(test_data_src.classes,prediction_cat)
	sns.heatmap(confusion)
	confusion_fn 		= os.path.join(log_dir,"confusion_matrix_{0}_{1}.png".format(num_class,time_str))
	plt.savefig(confusion_fn,dpi=300)
	plt.close()
	print("Saved histogram plot to {0}".format(confusion_fn))

if __name__ == "__main__":
	#model_fn1 			= r"/mnt/fast_data/ML/Models/Backups/Bulk_Crop_model_b1.h5"
	model_fn1 			= bulk_model_fn
	model				= tf.keras.models.load_model(model_fn1)
	report_accuracy(model)

if __name__ == "__main1__":
#	model_fn1			= r"Models/_Combined_Plant_model_softmax.h5"
#	plant_names_fn		= r"Models/_Combined_Plant_names.dat"
	model_fn1 			= r"Models/New_Crop_model.h5"
	plant_names_fn		= r"Models/Trained_classes.dat"

	class_list1			  = pickle.load(open(plant_names_fn,'rb'))
	class_list1 				= class_list[0:20] + class_list1
	model				   = tf.keras.models.load_model(model_fn1)
	report_accuracy(model,class_list1)
	pass