#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 13:56:40 2021

@author: bipin
"""

import os

data_dir            = "data"


model_dir 	    	= os.path.join(data_dir,"Models")
train_data_dir      = os.path.join(data_dir,"Train")
test_data_dir       = os.path.join(data_dir,"Test")
reserve_data_dir    = os.path.join(data_dir,"Reserve")
log_dir             = os.path.join(data_dir,'Log')
error_data_dir      = os.path.join(data_dir,'Error_images')



model_fn            = os.path.join(model_dir,"Plant_model.h5")
species_strength_fn = os.path.join(log_dir,"Species_strength.txt")
trained_classes_fn  = os.path.join(model_dir,"Trained_classes.dat")
