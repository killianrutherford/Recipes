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

from sklearn.model_selection import train_test_split



# for the larger networks (n>=9), we need to adjust pythons recursion limit
sys.setrecursionlimit(10000)


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

def load_temp(data, target):
	xs = []
	data_path = "/Users/killianrutherford1/Desktop/Courses/BIGDATA/project/data/aps/"
	for file in data:
		filename = data_path + file + ".aps"
		t = read_img(filename)
		xs.append(t)

	x = np.array(xs)
	print(x.shape)

	x = x.transpose(0,3,1,2)

	# subtract per-pixel mean
	pixel_mean = np.mean(x[:],axis=0)
	#pickle.dump(pixel_mean, open("cifar10-pixel_mean.pkl","wb"))
	x -= pixel_mean

	
	x_train_flip = x[:,:,:,::-1]
	y_train_flip = target
	X_train = np.concatenate((x,x_train_flip),axis=0)
	Y_train = np.concatenate((target,y_train_flip),axis=0)

	return lasagne.utils.floatX(X_train), Y_train.reshape(Y_train.shape[0],1).astype('int32')


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

	dflabel = group_labels_binary(label_path)

	x = np.array(dflabel['Id'])
	y = np.array(dflabel['Probability'])
	print(x)
	print(y)

	X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.20, random_state=42, stratify=y)

	return dict(X_train=X_train, 
		X_test = X_test, 
		Y_train = Y_train, 
		Y_test = Y_test)


# ##################### Build the neural network model #######################

from lasagne.layers import Conv2DLayer as ConvLayer
#from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers import PadLayer
from lasagne.layers import ExpressionLayer
from lasagne.layers import NonlinearityLayer
from lasagne.nonlinearities import softmax, rectify
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
		stack_left_1 = batch_norm(ConvLayer(l_l, num_filters=out_num_filters, filter_size=(5,5), stride=first_stride, nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
		stack_left_2 = batch_norm(ConvLayer(stack_left_1, num_filters=out_num_filters, filter_size=(5,5), stride=(1,1), nonlinearity=None, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
		

		stack_right_1 = batch_norm(ConvLayer(ElemwiseSumLayer([l, NegativeLayer(l_l)]), num_filters=out_num_filters, filter_size=(5,5), stride=first_stride, nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
		stack_right_2 = batch_norm(ConvLayer(stack_right_1, num_filters=out_num_filters, filter_size=(5,5), stride=(1,1), nonlinearity=None, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
		


		# add shortcut connections
		if increase_dim:
			if projection:
				# projection shortcut, as option B in paper
				projection = batch_norm(ConvLayer(l, num_filters=out_num_filters, filter_size=(1,1), stride=(2,2), nonlinearity=None, pad='same', b=None, flip_filters=False))
				block = NonlinearityLayer(ElemwiseSumLayer([stack_left_2, stack_right_2, projection]),nonlinearity=rectify)
			else:
				# identity shortcut, as option A in paper
				#print(l.output_shape[2])
				if(l.output_shape[2]%2 ==0):
					identity = ExpressionLayer(l, lambda X: X[:, :, ::2, ::2], lambda s: (s[0], s[1], s[2]//2, s[3]//2))
				else :
					identity = ExpressionLayer(l, lambda X: X[:, :, ::2, ::2], lambda s: (s[0], s[1], s[2]//2 + 1, s[3]//2 + 1))
				padding = PadLayer(identity, [(int)(out_num_filters/4),0,0], batch_ndim=1)
				print('------------------')
				print(stack_left_2.output_shape)
				print(stack_right_2.output_shape)
				print(identity.output_shape)
				print(padding.output_shape)
				block = NonlinearityLayer(ElemwiseSumLayer([stack_left_2, stack_right_2, padding]),nonlinearity=rectify)
		else:
			block = NonlinearityLayer(ElemwiseSumLayer([stack_left_2, stack_right_2, l]),nonlinearity=rectify)
		
		return block

	# Building the network
	l_in = InputLayer(shape=(None, 16, 512, 660), input_var=input_var)

	# first layer, output is 16 x 32 x 32
	l = batch_norm(ConvLayer(l_in, num_filters=16, filter_size=(5,5), stride=(1,1), nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
	print(l.output_shape)
	# first stack of residual blocks, output is 16 x 32 x 32
	for _ in range(n):
		l = residual_block(l)
		#print(l.output_shape)
		#print(l.output_shape)
	l = residual_block(l, increase_dim=True)
	for _ in range(n):
		l = residual_block(l)
	print(l.output_shape)

	l = residual_block(l, increase_dim=True)
	for _ in range(n):
		l = residual_block(l)
	print(l.output_shape)
	# second stack of residual blocks, output is 32 x 16 x 16
	#l = residual_block(l, increase_dim=True)
	#for _ in range(1,n):
	#    l = residual_block(l)

	"""
	# third stack of residual blocks, output is 64 x 8 x 8
	l = residual_block(l, increase_dim=True)
	for _ in range(1,n):
		l = residual_block(l)
	"""
	# average pooling
	l = GlobalPoolLayer(l)

	# fully connected layer
	network = DenseLayer(
			l, num_units=10,
			W=lasagne.init.HeNormal(),
			nonlinearity=None)

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

def main(n=3, num_epochs=100, model=None):
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
	data = load_data_1("/Users/killianrutherford1/Desktop/Courses/BIGDATA/project/stage1_labels.csv")


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
	
	if model is None:
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
		lr = 0.01
		print(lr)
		#lr=0.1
		sh_lr = theano.shared(lasagne.utils.floatX(lr))
		updates = lasagne.updates.momentum(
				loss, params, learning_rate=sh_lr, momentum=0.9)
		
		# Compile a function performing a training step on a mini-batch (by giving
		# the updates dictionary) and returning the corresponding training loss:
		train_fn = theano.function([input_var, target_var], loss, updates=updates)

	# Create a loss expression for validation/testing
	test_prediction = lasagne.layers.get_output(network, deterministic=True)
	test_loss = lasagne.objectives.binary_crossentropy(test_prediction,
															target_var)

	test_loss = test_loss.mean()
	#test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
	#                  dtype=theano.config.floatX)

	# Compile a second function computing the validation loss and accuracy:
	#val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
	val_fn = theano.function([input_var, target_var], test_loss)

	if model is None:
		# launch the training loop
		print("Starting training...")
		# We iterate over epochs:
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
			for batch in iterate_minibatches(X_train, Y_train, 16, shuffle=True, augment=False):
				inputs, targets = batch
				train_err += train_fn(inputs, targets)
				train_batches += 1

			# And a full pass over the validation data:
			val_err = 0
			val_acc = 0
			val_batches = 0
			for batch in iterate_minibatches(X_test, Y_test, 16, shuffle=False):
				inputs, targets = batch
				err = val_fn(inputs, targets)
				val_err += err
				#val_acc += acc
				val_batches += 1

			# Then we print the results for this epoch:
			print("Epoch {} of {} took {:.3f}s".format(
				epoch + 1, num_epochs, time.time() - start_time))
			print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
			print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
			print("  validation accuracy:\t\t{:.2f} %".format(
				val_acc / val_batches * 100))

			# adjust learning rate as in paper
			# 32k and 48k iterations should be roughly equivalent to 41 and 61 epochs
			if (epoch+1) == 31 or (epoch+1) == 61:
				new_lr = sh_lr.get_value() * 0.1
				print("New LR:"+str(new_lr))
				sh_lr.set_value(lasagne.utils.floatX(new_lr))

		# dump the network weights to a file :
		np.savez('cifar10_deep_residual_model.npz', *lasagne.layers.get_all_param_values(network))
	else:
		# load network weights from model file
		with np.load(model) as f:
			 param_values = [f['arr_%d' % i] for i in range(len(f.files))]
		lasagne.layers.set_all_param_values(network, param_values)

	# Calculate validation error of model:
	test_err = 0
	test_acc = 0
	test_batches = 0
	for batch in iterate_minibatches(X_test, Y_test, 16, shuffle=False):
		inputs, targets = batch
		err = val_fn(inputs, targets)
		test_err += err
		#test_acc += acc
		test_batches += 1
	print("Final results:")
	print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
	#print("  test accuracy:\t\t{:.2f} %".format(
	#    test_acc / test_batches * 100))


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











