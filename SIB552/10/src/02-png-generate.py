#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 05:26:15 2018

@author: ozg
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import glob
from androguard.core.bytecodes import apk
import numpy as np
import matplotlib.pyplot as plt
import hashlib

def file_as_bytes(file):
    with file:
        return file.read()

rootDir = 'c:/Users/ozgur/Documents/dataset/Android-Malwares'
width = 2048

results = open("results.txt","w")

for d in os.listdir(rootDir):
	print(d)
	try:
		for f in glob.glob(rootDir + "/" + d + "/*"):
			if ("README" not in f) and (".git" not in f) and (".pdf" not in f):
				print(f)
				png_name = hashlib.md5(file_as_bytes(open(f, 'rb'))).hexdigest()
				print(png_name)
				app = apk.APK(f)
				raw_value = app.get_raw()
				row_size = int (np.ceil(len(raw_value)/width))
				padding_value = row_size * width
				diff = int(padding_value - len(raw_value))
				raw_value = np.array(raw_value)
				raw_value = np.pad(raw_value,(0,diff), mode='constant')
				raw_value = np.reshape(raw_value, (row_size, width))
				plt.imsave("c:/Users/ozgur/Documents/dataset/pngs/" + png_name +".png", raw_value)
				results.write(d + "," + png_name +".png\n")
				results.flush()
	except Exception as e: 
		print(e)
results.close()
