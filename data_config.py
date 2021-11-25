#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 20:38:16 2021

@author: bipin
"""
import re

class_list_add  = open("new_classes.txt").readlines()
class_list_add  = [x.strip() for x in class_list_add]
class_list_add  = [x.replace(" ","_") for x in class_list_add]
class_list      = class_list_prefix+class_list_add

def get_proper_name(name1):
    name1 		= name1.strip()
    name1 		= re.sub("[^a-zA-Z0-9 ]"," ",name1)
    name_parts 	= re.split(' |_|,|\.|-|\'',name1)
    name_parts 	= list(filter(lambda x : len(x) > 2,name_parts))
    if len(name_parts) >= 2:
       name_parts = name_parts[:2]
    proper_name = "_".join(name_parts)
    return proper_name.lower()

class_list 		= list(map(get_proper_name,class_list))
