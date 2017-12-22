#!/usr/bin/env python

"""
based off code at :

https://github.com/Lasagne/Recipes/tree/master/papers/deep_residual_learning

Lasagne implementation of CIFAR-10 examples from "Deep Residual Learning for Image Recognition" (http://arxiv.org/abs/1512.03385)
Check the accompanying files for pretrained models. The 32-layer network (n=5), achieves a validation error of 7.42%, 
while the 56-layer network (n=9) achieves error of 6.75%, which is roughly equivalent to the examples in the paper.
"""

from __future__ import print_function

import sys
import os
import time
import string
import random
import pickle
import pandas as pd

#print(sys.path)

import numpy as np
import theano
import theano.tensor as T
import lasagne
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import cv2

# for the larger networks (n>=9), we need to adjust pythons recursion limit
sys.setrecursionlimit(10000)
# list uses the sector that best shows the threat zone
sector01_pts = np.array([[0,160],[200,160],[200,230],[0,230]], np.int32)
sector02_pts = np.array([[0,0],[200,0],[200,160],[0,160]], np.int32)
sector03_pts = np.array([[330,160],[512,160],[512,240],[330,240]], np.int32)
sector04_pts = np.array([[350,0],[512,0],[512,160],[350,160]], np.int32)

# sector 5 is used for both threat zone 5 and 17
sector05_pts = np.array([[0,220],[512,220],[512,300],[0,300]], np.int32) 

sector06_pts = np.array([[0,300],[256,300],[256,360],[0,360]], np.int32)
sector07_pts = np.array([[256,300],[512,300],[512,360],[256,360]], np.int32)
sector08_pts = np.array([[0,370],[225,370],[225,450],[0,450]], np.int32)
sector09_pts = np.array([[225,370],[275,370],[275,450],[225,450]], np.int32)
sector10_pts = np.array([[275,370],[512,370],[512,450],[275,450]], np.int32)
sector11_pts = np.array([[0,450],[256,450],[256,525],[0,525]], np.int32)
sector12_pts = np.array([[256,450],[512,450],[512,525],[256,525]], np.int32)
sector13_pts = np.array([[0,525],[256,525],[256,600],[0,600]], np.int32)
sector14_pts = np.array([[256,525],[512,525],[512,600],[256,600]], np.int32)
sector15_pts = np.array([[0,600],[256,600],[256,660],[0,660]], np.int32)
sector16_pts = np.array([[256,600],[512,600],[512,660],[256,660]], np.int32)

# crop dimensions, upper left x, y, width, height
sector_crop_list = [[ 50,  50, 250, 250], # sector 1
                    [  0,   0, 250, 250], # sector 2
                    [ 50, 250, 250, 250], # sector 3
                    [250,   0, 250, 250], # sector 4
                    [150, 150, 250, 250], # sector 5/17
                    [200, 100, 250, 250], # sector 6
                    [200, 150, 250, 250], # sector 7
                    [250,  50, 250, 250], # sector 8
                    [250, 150, 250, 250], # sector 9
                    [300, 200, 250, 250], # sector 10
                    [400, 100, 250, 250], # sector 11
                    [350, 200, 250, 250], # sector 12
                    [410,   0, 250, 250], # sector 13
                    [410, 200, 250, 250], # sector 14
                    [410,   0, 250, 250], # sector 15
                    [410, 200, 250, 250], # sector 16
                   ]

# Each element in the zone_slice_list contains the sector to use in the call to roi()
zone_slice_list = [ [ # threat zone 1
                      sector01_pts, sector01_pts, sector01_pts, None, 
                      None, None, sector03_pts, sector03_pts, 
                      sector03_pts, sector03_pts, sector03_pts, 
                      None, None, sector01_pts, sector01_pts, sector01_pts ], 
    
                    [ # threat zone 2
                      sector02_pts, sector02_pts, sector02_pts, None, 
                      None, None, sector04_pts, sector04_pts, 
                      sector04_pts, sector04_pts, sector04_pts, None, 
                      None, sector02_pts, sector02_pts, sector02_pts ],
    
                    [ # threat zone 3
                      sector03_pts, sector03_pts, sector03_pts, sector03_pts, 
                      None, None, sector01_pts, sector01_pts,
                      sector01_pts, sector01_pts, sector01_pts, sector01_pts, 
                      None, None, sector03_pts, sector03_pts ],
    
                    [ # threat zone 4
                      sector04_pts, sector04_pts, sector04_pts, sector04_pts, 
                      None, None, sector02_pts, sector02_pts, 
                      sector02_pts, sector02_pts, sector02_pts, sector02_pts, 
                      None, None, sector04_pts, sector04_pts ],
    
                    [ # threat zone 5
                      sector05_pts, sector05_pts, sector05_pts, sector05_pts, 
                      sector05_pts, sector05_pts, sector05_pts, sector05_pts,
                      None, None, None, None, 
                      None, None, None, None ],
    
                    [ # threat zone 6
                      sector06_pts, None, None, None, 
                      None, None, None, None, 
                      sector07_pts, sector07_pts, sector06_pts, sector06_pts, 
                      sector06_pts, sector06_pts, sector06_pts, sector06_pts ],
    
                    [ # threat zone 7
                      sector07_pts, sector07_pts, sector07_pts, sector07_pts, 
                      sector07_pts, sector07_pts, sector07_pts, sector07_pts, 
                      None, None, None, None, 
                      None, None, None, None ],
    
                    [ # threat zone 8
                      sector08_pts, sector08_pts, None, None, 
                      None, None, None, sector10_pts, 
                      sector10_pts, sector10_pts, sector10_pts, sector10_pts, 
                      sector08_pts, sector08_pts, sector08_pts, sector08_pts ],
    
                    [ # threat zone 9
                      sector09_pts, sector09_pts, sector08_pts, sector08_pts, 
                      sector08_pts, None, None, None,
                      sector09_pts, sector09_pts, None, None, 
                      None, None, sector10_pts, sector09_pts ],
    
                    [ # threat zone 10
                      sector10_pts, sector10_pts, sector10_pts, sector10_pts, 
                      sector10_pts, sector08_pts, sector10_pts, None, 
                      None, None, None, None, 
                      None, None, None, sector10_pts ],
    
                    [ # threat zone 11
                      sector11_pts, sector11_pts, sector11_pts, sector11_pts, 
                      None, None, sector12_pts, sector12_pts,
                      sector12_pts, sector12_pts, sector12_pts, None, 
                      sector11_pts, sector11_pts, sector11_pts, sector11_pts ],
    
                    [ # threat zone 12
                      sector12_pts, sector12_pts, sector12_pts, sector12_pts, 
                      sector12_pts, sector11_pts, sector11_pts, sector11_pts, 
                      sector11_pts, sector11_pts, sector11_pts, None, 
                      None, sector12_pts, sector12_pts, sector12_pts ],
    
                    [ # threat zone 13
                      sector13_pts, sector13_pts, sector13_pts, sector13_pts, 
                      None, None, sector14_pts, sector14_pts,
                      sector14_pts, sector14_pts, sector14_pts, None, 
                      sector13_pts, sector13_pts, sector13_pts, sector13_pts ],
    
                    [ # sector 14
                      sector14_pts, sector14_pts, sector14_pts, sector14_pts, 
                      sector14_pts, None, sector13_pts, sector13_pts, 
                      sector13_pts, sector13_pts, sector13_pts, None, 
                      None, None, None, None ],
    
                    [ # threat zone 15
                      sector15_pts, sector15_pts, sector15_pts, sector15_pts, 
                      None, None, sector16_pts, sector16_pts,
                      sector16_pts, sector16_pts, None, sector15_pts, 
                      sector15_pts, None, sector15_pts, sector15_pts ],
    
                    [ # threat zone 16
                      sector16_pts, sector16_pts, sector16_pts, sector16_pts, 
                      sector16_pts, sector16_pts, sector15_pts, sector15_pts, 
                      sector15_pts, sector15_pts, sector15_pts, None, 
                      None, None, sector16_pts, sector16_pts ],
    
                    [ # threat zone 17
                      None, None, None, None, 
                      None, None, None, None,
                      sector05_pts, sector05_pts, sector05_pts, sector05_pts, 
                      sector05_pts, sector05_pts, sector05_pts, sector05_pts ] ]




def group_labels(file):
	data = pd.read_csv(file, sep=',')
	data[['Id', 'Zone']] = data['Id'].str.split("_", expand = True)
	data1 = data[['Id', 'Probability']]
	data2 = pd.DataFrame(data1.groupby('Id', as_index=False)['Probability'].sum())
	return data2

def group_labels_binary(file):
	df = group_labels(file)
	#print(df)
	df['Probability'] = df['Probability'].apply(lambda x: 1 if x>0 else 0)
	#print(df)
	return df

def temp(x):
	if(x[1] == 1 and x[3] > 1):
		return 1

	if(x[1] == 0 and x[3] >= 1):
		return 1

	return 0


def group_labels_mask(file):
	df = group_labels(file)
	data = pd.read_csv(file, sep=',')
	data[['Id', 'Zone']] = data['Id'].str.split("_", expand = True)
	#data.join(df.set_index('Id'), on='Id', lsuffix='_1', rsuffix='_2')
	x = data.merge(df, left_on='Id', right_on='Id', how='inner')	
	x['Probability'] = x.apply(temp, axis=1)
	print(x)
	return x[['Id', 'Zone', 'Probability']]



class NegativeLayer(lasagne.layers.Layer):
	def get_output_for(self, input, **kwargs):
		return (-1 * input)

def read_header(infile):
    # declare dictionary
    h = dict()
    
    with open(infile, 'r+b') as fid:

        h['filename'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 20))
        h['parent_filename'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 20))
        h['comments1'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 80))
        h['comments2'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 80))
        h['energy_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['config_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['file_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['trans_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['scan_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['data_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['date_modified'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 16))
        h['frequency'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['mat_velocity'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['num_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
        h['num_polarization_channels'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['spare00'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['adc_min_voltage'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['adc_max_voltage'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['band_width'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['spare01'] = np.fromfile(fid, dtype = np.int16, count = 5)
        h['polarization_type'] = np.fromfile(fid, dtype = np.int16, count = 4)
        h['record_header_size'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['word_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['word_precision'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['min_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['max_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['avg_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['data_scale_factor'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['data_units'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['surf_removal'] = np.fromfile(fid, dtype = np.uint16, count = 1)
        h['edge_weighting'] = np.fromfile(fid, dtype = np.uint16, count = 1)
        h['x_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
        h['y_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
        h['z_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
        h['t_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
        h['spare02'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['x_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['y_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['z_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['scan_orientation'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['scan_direction'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['data_storage_order'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['scanner_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['x_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['y_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['z_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['t_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['num_x_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
        h['num_y_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
        h['num_z_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
        h['num_t_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
        h['x_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['y_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['z_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['x_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['y_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['z_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['x_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['y_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['z_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['x_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['y_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['z_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['date_processed'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 8))
        h['time_processed'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 8))
        h['depth_recon'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['x_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['y_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['elevation_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['roll_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['z_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['azimuth_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['adc_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['spare06'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['scanner_radius'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['x_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['y_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['z_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['t_delay'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['range_gate_start'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['range_gate_end'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['ahis_software_version'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['spare_end'] = np.fromfile(fid, dtype = np.float32, count = 10)

    return h

def read_img(infile):
	
	# read in header and get dimensions
	h = read_header(infile)
	nx = int(h['num_x_pts'])
	ny = int(h['num_y_pts'])
	nt = int(h['num_t_pts'])
	
	extension = os.path.splitext(infile)[1]
	
	with open(infile, 'rb') as fid:
		  
		# skip the header
		fid.seek(512) 

		# handle .aps and .a3aps files
		if extension == '.aps' or extension == '.a3daps':
		
			if(h['word_type']==7):
				data = np.fromfile(fid, dtype = np.float32, count = nx * ny * nt)

			elif(h['word_type']==4): 
				data = np.fromfile(fid, dtype = np.uint16, count = nx * ny * nt)

			# scale and reshape the data
			data = data * h['data_scale_factor'] 
			data = data.reshape(nx, ny, nt, order='F').copy()

		# handle .a3d files
		elif extension == '.a3d':
			  
			if(h['word_type']==7):
				data = np.fromfile(fid, dtype = np.float32, count = nx * ny * nt)
				
			elif(h['word_type']==4):
				data = np.fromfile(fid, dtype = np.uint16, count = nx * ny * nt)

			# scale and reshape the data
			data = data * h['data_scale_factor']
			data = data.reshape(nx, nt, ny, order='F').copy() 
			
		# handle .ahi files
		elif extension == '.ahi':
			data = np.fromfile(fid, dtype = np.float32, count = 2* nx * ny * nt)
			data = data.reshape(2, ny, nx, nt, order='F').copy()
			real = data[0,:,:,:].copy()
			imag = data[1,:,:,:].copy()

		if extension != '.ahi':
			return data
		else:
			return real, imag

def get_single_image(img, nth_image):

    # read in the aps file, it comes in as shape(512, 620, 16)
    
    # transpose so that the slice is the first dimension shape(16, 620, 512)
    #img = img.transpose()
    
    return np.flipud(img[nth_image])

def roi(img, vertices):
    
    # blank mask
    mask = np.zeros_like(img)

    # fill the mask
    cv2.fillPoly(mask, [vertices], 255)

    # now only show the area that is the mask
    masked = cv2.bitwise_and(img, mask)
    

    return masked

def thresh(img, threshmin, newval):
    img[img < threshmin] = newval
    return img

def convert_to_grayscale(img):
    # scale pixel values to grayscale
    base_range = np.amax(img) - np.amin(img)
    rescaled_range = 255 - 0
    img_rescaled = (((img - np.amin(img)) * rescaled_range) / base_range)

    return np.uint8(img_rescaled)


def spread_spectrum(img):
	img = thresh(img, threshmin=12, newval=0.)

	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    	img= clahe.apply(img)
    	return img

def load_temp(data, target):
	xs = []
	data_path = "/mnt/disks/gpu-disk/aps/"
	for f in data:
		#print((f[0], f[1]))
		filename = data_path + f[0] + ".aps"
		t = read_img(filename)
		t = t.transpose()
		#print(t.shape)
		l = f[1][4:]
		#print(l)
		lst = []
		for k in xrange(16):
			if zone_slice_list[int(l)-1][k] is not None:
				p = get_single_image(t, k)
				p_rescaled = convert_to_grayscale(p)
				p_h = spread_spectrum(p_rescaled)
				o = roi(p_h, zone_slice_list[int(l)-1][k])
			else :
				o = np.zeros((660,512))
			lst.append(np.flipud(o))
		#t = t[:,:,[0,8,15]]
		q = np.dstack(lst).transpose(2,0,1)
		xs.append(t-q)

	x = np.array(xs)

	x = x.transpose(0,1,3,2)
	# subtract per-pixel mean
	pixel_mean = np.mean(x[:],axis=0)
	#pickle.dump(pixel_mean, open("cifar10-pixel_mean.pkl","wb"))
	x -= pixel_mean
	
	#x_train_flip = x[:,:,:,::-1]
	#y_train_flip = target
	#X_train = np.concatenate((x,x_train_flip),axis=0)
	#Y_train = np.concatenate((target,y_train_flip),axis=0)

	return lasagne.utils.floatX(x), target.reshape(target.shape[0],1).astype('float32')


def load_data(data_path, label_path):
	xs = []
	ys = []

	dflabel = group_labels_binary(label_path)

	directory = os.fsencode(data_path)

	for file in os.listdir(directory):
		filename = os.fsdecode(file)
		if filename.endswith(".aps"):
			fn = filename.split('.')[0]
			#rint(dflabel.loc[dflabel['Id']==fn])
			try:
				p = dflabel.loc[dflabel['Id'] == fn, 'Probability'].iloc[0]
				t = read_img(data_path + filename)
				xs.append(t)
				ys.append(p)
			except:
				print("Not found")
			
			#print(t.shape)
			#print(p)

	x = np.array(xs)
	#x = np.concatenate(xs)/np.float32(255)
	y = np.array(ys)
	print(x.shape)
	print(y.shape)
	x = x.transpose(0,3,1,2)

	# subtract per-pixel mean
	pixel_mean = np.mean(x[:],axis=0)
	#pickle.dump(pixel_mean, open("cifar10-pixel_mean.pkl","wb"))
	x -= pixel_mean

	X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.20, random_state=42, stratify=y)
	# create mirrored images
	#X_train = x[0:100,:,:,:]
	#Y_train = y[0:100]
	X_train_flip = X_train[:,:,:,::-1]
	Y_train_flip = Y_train
	X_train = np.concatenate((X_train,X_train_flip),axis=0)
	Y_train = np.concatenate((Y_train,Y_train_flip),axis=0)


	#TODO: more augmentation
	#batch size change
	#log loss error


	return dict(
		X_train=lasagne.utils.floatX(X_train),
		Y_train=Y_train.astype('int32'),
		X_test = lasagne.utils.floatX(X_test),
		Y_test = Y_test.astype('int32'),)


def load_data_1(label_path):

	#dflabel = group_labels_binary(label_path)
	dflabel = group_labels_mask(label_path)
	print("dflabelshape: ", dflabel.shape)
	x = np.array(dflabel[['Id', 'Zone']])
	y = np.array(dflabel['Probability'])
	print(x)
	print(y)
	
	X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.20, random_state=42, stratify=y)

	return dict(X_train= np.array(x),
		X_test = np.array(X_test), 
		Y_train = np.array(y), 
		Y_test = np.array(Y_test))


# ##################### Build the neural network model #######################

from lasagne.layers import Conv2DLayer as ConvLayer
#from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import GlobalPoolLayer, DropoutLayer
from lasagne.layers import PadLayer
from lasagne.layers import ExpressionLayer
from lasagne.layers import NonlinearityLayer
from lasagne.nonlinearities import softmax, rectify, sigmoid
from lasagne.layers import batch_norm

def build_cnn(input_var=None, n=5):
	
	# create a residual learning building block with two stacked 3x3 convlayers as in paper
	def residual_block(l, increase_dim=False, projection=False):
		input_num_filters = l.output_shape[1]
		if increase_dim:
			first_stride = (2,2)
			out_num_filters = input_num_filters*2
		else:
			first_stride = (1,1)
			out_num_filters = input_num_filters

		#print(l.output_shape)
		l_l = DenseLayer(l, num_units=l.output_shape[3], num_leading_axes=-1, nonlinearity=None)
		#print(l.output_shape[3])
		#print("l_1.output_shape", l_l.output_shape)
		#stride=first_stride
		stack_left_1 = batch_norm(ConvLayer(l_l, num_filters=out_num_filters, filter_size=(3,3), stride=first_stride, nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
		stack_left_2 = batch_norm(ConvLayer(stack_left_1, num_filters=out_num_filters, filter_size=(3,3), stride=(1,1), nonlinearity=None, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
		

		#stack_right_1 = batch_norm(ConvLayer(ElemwiseSumLayer([l, NegativeLayer(l_l)]), num_filters=out_num_filters, filter_size=(2,2), stride=first_stride, nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
		#stack_right_2 = batch_norm(ConvLayer(stack_right_1, num_filters=out_num_filters, filter_size=(2,2), stride=(1,1), nonlinearity=None, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
		print("first stack: ", stack_left_2.output_shape)


		# add shortcut connections
		if increase_dim:
			if projection:
				# projection shortcut, as option B in paper
				projection = batch_norm(ConvLayer(l, num_filters=out_num_filters, filter_size=(1,1), stride=(2,2), nonlinearity=None, pad='same', b=None, flip_filters=False))
				print("projection shape: ", projection.output_shape)
				##block = NonlinearityLayer(ElemwiseSumLayer([stack_left_2, stack_right_2, projection]),nonlinearity=rectify)
				block = NonlinearityLayer(ElemwiseSumLayer([stack_left_2, projection]),nonlinearity=rectify)
			else:
				# identity shortcut, as option A in paper
				#print(l.output_shape[2])
				if(l.output_shape[2]%2 ==0 and l.output_shape[3]%2 == 0):
					identity = ExpressionLayer(l, lambda X: X[:, :, ::2, ::2], lambda s: (s[0], s[1], s[2]//2, s[3]//2))
				elif(l.output_shape[2]%2 ==0 and l.output_shape[3]%2 == 1):
					identity = ExpressionLayer(l, lambda X: X[:, :, ::2, ::2], lambda s: (s[0], s[1], s[2]//2, s[3]//2 + 1))
				elif(l.output_shape[2]%2 ==1 and l.output_shape[3]%2 == 0):
					identity = ExpressionLayer(l, lambda X: X[:, :, ::2, ::2], lambda s: (s[0], s[1], s[2]//2 + 1, s[3]//2))
				else :
					identity = ExpressionLayer(l, lambda X: X[:, :, ::2, ::2], lambda s: (s[0], s[1], s[2]//2 + 1, s[3]//2 + 1))
				padding = PadLayer(identity, [(int)(out_num_filters/4),0,0], batch_ndim=1)
				print('------------------')
				print(stack_left_2.output_shape)
				#print(stack_right_2.output_shape)
				print(identity.output_shape)
				print(padding.output_shape)
				#block = NonlinearityLayer(ElemwiseSumLayer([stack_left_2, stack_right_2, padding]),nonlinearity=rectify)
				block = NonlinearityLayer(ElemwiseSumLayer([stack_left_2, padding]),nonlinearity=rectify)
		else:
			#block = NonlinearityLayer(ElemwiseSumLayer([stack_left_2, stack_right_2, l]),nonlinearity=rectify)
			print("l output shape: ", l.output_shape)
			block = NonlinearityLayer(ElemwiseSumLayer([stack_left_2, l]),nonlinearity=rectify)
		
		return block

	# Building the network
	l_in = InputLayer(shape=(None, 16, 512, 660), input_var=input_var)

	# first layer, output is 16 x 32 x 32
	l = batch_norm(ConvLayer(l_in, num_filters=16, filter_size=(3,3), stride=(4,4), nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
	print(l.output_shape)
	# first stack of residual blocks, output is 16 x 32 x 32
	for _ in range(n):
		l = residual_block(l)
		#l = DropoutLayer(l, p = 0.7)
		#print(l.output_shape)
		#print(l.output_shape)
	l = residual_block(l, increase_dim=True)
	#l = DropoutLayer(l, p = 0.5)
	for _ in range(n):
		l = residual_block(l)
		#l = DropoutLayer(l, p = 0.5)
	print(l.output_shape)
	
	l = batch_norm(ConvLayer(l, num_filters = 32, filter_size=(3,3), stride=(2,2), nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))	
	#l = residual_block(l, increase_dim=True)
	#for _ in range(n):
	#	l = residual_block(l)
	#print(l.output_shape)
	#second stack of residual blocks, output is 32 x 16 x 16
	#l = batch_norm(ConvLayer(l, num_filters = 64, filter_size=(3,3), stride=(2,2), nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))	
	#l = residual_block(l, increase_dim=True)
	#for _ in range(n):
	#    l = residual_block(l)
	#print(l.output_shape)
	"""
	# third stack of residual blocks, output is 64 x 8 x 8
	l = residual_block(l, increase_dim=True)
	for _ in range(1,n):
		l = residual_block(l)
	"""
	# average pooling
	l = GlobalPoolLayer(l)
	print("before dense: ",l.output_shape)
	# fully connected layer
	network = DenseLayer(
			l, num_units=1,
			W=lasagne.init.HeNormal(),
			nonlinearity=sigmoid)

	return network

# ############################# Batch iterator ###############################

def iterate_minibatches(inputs, targets, batchsize, shuffle=False, augment=False):
	assert len(inputs) == len(targets)
	if shuffle:
		indices = np.arange(len(inputs))
		np.random.shuffle(indices)
	for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batchsize]
		else:
			excerpt = slice(start_idx, start_idx + batchsize)
		if augment:
			# as in paper : 
			# pad feature arrays with 4 pixels on each side
			# and do random cropping of 32x32
			padded = np.pad(inputs[excerpt],((0,0),(0,0),(4,4),(4,4)),mode='constant')
			random_cropped = np.zeros(inputs[excerpt].shape, dtype=np.float32)
			crops = np.random.random_integers(0,high=8,size=(batchsize,2))
			for r in range(batchsize):
				random_cropped[r,:,:,:] = padded[r,:,crops[r,0]:(crops[r,0]+10),crops[r,1]:(crops[r,1]+10)]
			inp_exc = random_cropped
		else:
			inp_exc, targets1 = load_temp(inputs[excerpt], targets[excerpt])
		
		yield inp_exc, targets1

# ############################## Main program ################################

def main(n=5, num_epochs=100, model=None):
	print('testuz331')
	# Check if cifar data exists
	"""
	if not os.path.exists("./cifar-10-batches-py"):
		print("CIFAR-10 dataset can not be found. Please download the dataset from 'https://www.cs.toronto.edu/~kriz/cifar.html'.")
		return
	"""
	# Load the dataset
	print("Loading data...")
	#data = load_data("/Users/killianrutherford1/Desktop/Courses/BIGDATA/project/data/aps/","/Users/killianrutherford1/Desktop/Courses/BIGDATA/project/stage1_labels.csv")
	#data = load_data_1("/Users/killianrutherford1/Desktop/Courses/BIGDATA/project/stage1_labels.csv")
	data = load_data_1("/mnt/disks/gpu-disk/stage1_labels.csv")
	

	X_train = data['X_train']
	Y_train = data['Y_train']
	X_test = data['X_test']
	Y_test = data['Y_test']

	# Prepare Theano variables for inputs and targets
	input_var = T.tensor4('inputs')
	target_var = T.matrix('targets')

	# Create neural network model
	print("Building model and compiling functions...")
	network = build_cnn(input_var, n)
	print("number of parameters in model: %d" % lasagne.layers.count_params(network, trainable=True))
	
	if model is not None:
		with np.load(model) as f:
			param_values = [f['arr_%d' % i] for i in range(len(f.files))]
			lasagne.layers.set_all_param_values(network, param_values)
			print("done")


		# Create a loss expression for training, i.e., a scalar objective we want
		# to minimize (for our multi-class problem, it is the cross-entropy loss):
		prediction = lasagne.layers.get_output(network)
		#print(prediction)
		loss = lasagne.objectives.binary_crossentropy(prediction, target_var)
		loss = loss.mean()
		# add weight decay
		all_layers = lasagne.layers.get_all_layers(network)
		l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * 0.0001
		loss = loss + l2_penalty

		# Create update expressions for training
		# Stochastic Gradient Descent (SGD) with momentum
		params = lasagne.layers.get_all_params(network, trainable=True)
		lr = 0.03
		print(lr)
		#lr=0.1
		sh_lr = theano.shared(lasagne.utils.floatX(lr))
		updates = lasagne.updates.momentum(
				loss, params, learning_rate=sh_lr, momentum=0.9)
		
		# Compile a function performing a training step on a mini-batch (by giving
		# the updates dictionary) and returning the corresponding training loss:
		train_fn = theano.function([input_var, target_var], loss, updates=updates, allow_input_downcast=True)

	# Create a loss expression for validation/testing
	test_prediction = lasagne.layers.get_output(network, deterministic=True)
	test_loss = lasagne.objectives.binary_crossentropy(test_prediction, target_var)

	test_loss = test_loss.mean()
	test_acc = T.mean(T.eq(T.round(test_prediction), target_var), dtype=theano.config.floatX)

	# Compile a second function computing the validation loss and accuracy:
	#val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
	val_fn = theano.function([input_var, target_var], [test_loss, test_acc], allow_input_downcast=True)

	if model is not None:
		# launch the training loop
		print("Starting training...")
		# We iterate over epochs:
		t1 = time.time()
		for epoch in range(num_epochs):
			# shuffle training data
			train_indices = np.arange(X_train.shape[0])
			np.random.shuffle(train_indices)
			X_train = X_train[train_indices]
			Y_train = Y_train[train_indices]

			# In each epoch, we do a full pass over the training data:
			train_err = 0
			train_batches = 0
			start_time = time.time()
			for batch in tqdm(iterate_minibatches(X_train, Y_train, 16, shuffle=True, augment=False)):
				inputs, targets = batch
				train_err += train_fn(inputs, targets)
				train_batches += 1

			# And a full pass over the validation data:
			val_err = 0
			val_acc = 0
			val_batches = 0
			
			for batch in tqdm(iterate_minibatches(X_test, Y_test, 16, shuffle=False)):
				inputs, targets = batch
				err, acc = val_fn(inputs, targets)
				val_err += err
				val_acc += acc
				val_batches += 1
			
			# Then we print the results for this epoch:
			print("Epoch {} of {} took {:.3f}s".format(
				epoch + 1, num_epochs, time.time() - start_time))
			print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
			print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
			print("  validation accuracy:\t\t{:.2f} %".format(
				val_acc / val_batches * 100))
			
			fp = open("results/weights_model_" + str(t1) + ".txt", "a+")
				
			fp.write("Epoch {} of {} took {:.3f}s\n".format(
				epoch + 1, num_epochs, time.time() - start_time))
			fp.write("  training loss:\t\t{:.6f}\n".format(train_err / train_batches))
			fp.write("  validation loss:\t\t{:.6f}\n".format(val_err / val_batches))
			fp.write("  validation accuracy:\t\t{:.2f} %\n".format(
				val_acc / val_batches * 100))
				
			fp.close()
	
			# adjust learning rate as in paper
			# 32k and 48k iterations should be roughly equivalent to 41 and 61 epochs
			#if (epoch+1) == 23 or (epoch+1) == 43 or (epoch+1) == 64 or (epoch+1) == 85:
			if (epoch+1) == 31 or (epoch+1) == 61:
				new_lr = sh_lr.get_value() * 0.1
				print("New LR:"+str(new_lr))
				sh_lr.set_value(lasagne.utils.floatX(new_lr))
			np.savez('weights/passenger_screening_' +str(t1) + '.npz', *lasagne.layers.get_all_param_values(network))
		# dump the network weights to a file :
	else:
		# load network weights from model file
		with np.load(model) as f:
			 param_values = [f['arr_%d' % i] for i in range(len(f.files))]
		lasagne.layers.set_all_param_values(network, param_values)

	# Calculate validation error of model:
	test_err = 0
	test_acc = 0
	test_batches = 0
	for batch in tqdm(iterate_minibatches(X_test, Y_test, 16, shuffle=False)):
		inputs, targets = batch
		err,acc = val_fn(inputs, targets)
		test_err += err
		test_acc += acc
		test_batches += 1
	print("Final results:")
	print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
	print("  test accuracy:\t\t{:.2f} %".format(
	    test_acc / test_batches * 100))


if __name__ == '__main__':
	if ('--help' in sys.argv) or ('-h' in sys.argv):
		print("Trains a Deep Residual Learning network on cifar-10 using Lasagne.")
		print("Network architecture and training parameters are as in section 4.2 in 'Deep Residual Learning for Image Recognition'.")
		print("Usage: %s [N [MODEL]]" % sys.argv[0])
		print()
		print("N: Number of stacked residual building blocks per feature map (default: 5)")
		print("MODEL: saved model file to load (for validation) (default: None)")
	else:
		kwargs = {}
		if len(sys.argv) > 1:
			kwargs['n'] = int(sys.argv[1])
		if len(sys.argv) > 2:
			kwargs['model'] = sys.argv[2]
		main(**kwargs)











