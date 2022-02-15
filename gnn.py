from numpy import expand_dims
from numpy import mean
from numpy import ones
from numpy.random import randn
from numpy.random import randint
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Input, Conv3DTranspose, Reshape, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.constraints import Constraint
#import keras.backend as kb
from dataset import HsiDataset
import h5py
import numpy as np
import matplotlib.pyplot as plt
#import pysptools
import config
import os
from PIL import Image
from scipy import ndimage
#p norm for loss function
p=1.5
latent_dim =27
latent_code = tf.random.normal([1,1,latent_dim])
learning_rate = 0.0005

def resize(dataset):
    #resize dataset into
    depth = 216
    width = 144
    height = 144

    #from size of
    print(np.shape(dataset))
    dataset_depth = np.shape(dataset)[-1]
    dataset_width = np.shape(dataset)[1]
    dataset_height = np.shape(dataset)[0]

    d_factor = depth / dataset_depth
    w_factor = width / dataset_width
    h_factor = height / dataset_height

    resized_dataset = ndimage.zoom(dataset, (h_factor, w_factor, d_factor))
    return resized_dataset
#bilinear interpolation upsampling


def upsample(input1, upsamplescale, channel_count):
	xdim=input1.shape[0]*upsamplescale
	ydim = input1.shape[1]*upsamplescale
	zdim = input1.shape[2]

	#kernel = tf.keras.initializers.Constant(np.ones([upsamplescale,upsamplescale,1,channel_count,channel_count], np.float32))
	kernel = np.ones([upsamplescale,upsamplescale,1,channel_count,channel_count], np.float32)
	deconv = tf.nn.conv3d_transpose(input=input1, filters=tf.constant(np.ones([upsamplescale,upsamplescale,1,channel_count,channel_count], np.float32)), output_shape=[1, xdim, ydim, zdim, channel_count],
                                strides=[1, upsamplescale, upsamplescale, 1, 1],
                                padding="SAME", name='UpsampleDeconv')

	#deconv = tf.keras.layers.Conv3DTranspose(filters =1 ,kernel_size = tf.shape(kernel),kernel_initializer=kernel,
     #                           strides=[1, upsamplescale, upsamplescale, 1, 1],
      #                          padding="SAME", data_format = 'channels_last')(input)
	#smooth5d = tf.keras.initializers.Constant(np.ones([upsamplescale,upsamplescale,1,channel_count,channel_count],dtype='float32')/np.float32(upsamplescale)/np.float32(upsamplescale)/np.float32(1))
	smooth5d = tf.constant(np.ones([upsamplescale,upsamplescale,1,channel_count,channel_count],dtype='float32')/np.float32(upsamplescale)/np.float32(upsamplescale)/np.float32(1))
	print('Upsample', upsamplescale)
	return tf.nn.conv3d(input = deconv,
		filters = smooth5d,
		strides = [1, 1, 1, 1, 1],
		padding = 'SAME',
		)

def loss_fn(y_pred, y_true):
	return tf.math.pow((tf.math.abs(y_true- y_pred)),1.5)

def generator():
	input_shape=(1,27)
	inputs = tf.keras.Input(input_shape)
	inputs1 = tf.keras.layers.Dense(input_dim=27, units=27)(inputs)
	x=tf.keras.layers.Dense(input_dim=27,units=3*3*3)(inputs1)
	x = tf.reshape(x,(1,3,3,3,1))

	x1 = tf.keras.layers.UpSampling3D(size=(3, 3, 3))(x)
	x1 = tf.keras.layers.Conv3D(filters=64, kernel_size=3, use_bias = False, padding = 'same')(x1)
	x1 = tf.keras.layers.LeakyReLU(0.2)(x1)
	#x2=tf.image.resize(x1,size=[8,8],method='bilinear')
	

	x2 = tf.keras.layers.UpSampling3D(size=(2, 2, 3))(x1)
	x2 = tf.keras.layers.Conv3D(filters=32, kernel_size=3, use_bias = False, padding = 'same')(x2)
	x2 = tf.keras.layers.LeakyReLU(0.2)(x2)
	#x4 = tf.image.resize(x3,size=[16,16],method='bilinear')
	
	x3 = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(x2)
	x3 = tf.keras.layers.Conv3D(filters=16, kernel_size=3, use_bias=False, padding = 'same')(x3)
	x3 = tf.keras.layers.LeakyReLU(0.2)(x3)
	
	#x6 = tf.image.resize(x5,size=[32,32],method='bilinear')

	x4 = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(x3)
	x4 = tf.keras.layers.Conv3D(filters=8, kernel_size=3, use_bias=False, padding = 'same')(x4)
	x4 = tf.keras.layers.LeakyReLU(0.2)(x4)
	#x8 = tf.image.resize(x7,size=[64,64],method='bilinear')
	
	x5 = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(x4)
	x5 = tf.keras.layers.Conv3D(filters=4, kernel_size=3, use_bias=False, padding = 'same')(x5)
	x5 = tf.keras.layers.LeakyReLU(0.2)(x5)
	
	#x10 = tf.image.resize(x9,size=[128,128],method='bilinear')

	x5 = tf.keras.layers.Conv3D(filters=1, kernel_size=3, use_bias=False, padding = 'same')(x5)
	x5 = tf.keras.layers.Activation('tanh')(x5)
	#x12 = tf.keras.layers.UpSampling3D(size=(3, 3, 3), data_format=None, **kwargs)(x11)
	#x12 = tf.image.resize(x11,size=[145,145],method='bilinear')
	x5 = tf.reshape(x5, shape_of_dataset)
	model = tf.keras.Model(inputs, x5)
	return model

class GNN(tf.keras.Model):
	def __init__(
		self,
		generator):
		super(GNN,self).__init__()
		self.generator=generator

	def compile(self):
		super(GNN,self).compile()
		self.optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate, beta_1=0.9)
		self.loss_fn = loss_fn

	def train_step(self,dataset):
		with tf.GradientTape() as tp:
			X_gen = self.generator(latent_code)
			loss = self.loss_fn(X_gen, dataset)
		gradient = tp.gradient(loss, self.generator.trainable_variables)
		self.optimizer.apply_gradients(zip(gradient,self.generator.trainable_variables))
		print("loss: ",loss)
		return {"loss": loss}




class GNNOutput(tf.keras.callbacks.Callback):
	def __init__(self,latent_dim=50):
		self.latent_dim = latent_dim

	def on_epoch_end(self, epoch, logs = None):
		if(epoch%50 == 0):
			X_gen = self.model.generator(latent_code)
			X_gen = (X_gen + 1)*127.5
			I = X_gen.numpy()[0,:,:,0,0].astype(np.uint8)
			#I8 = (((I-I.min())/(I.max()-I.min()))*255.9).astype(np.uint8)
			img = Image.fromarray(I)
			img = img.convert("L")
			img.save("generated_img_{epoch}.png".format(epoch=epoch))


latent_dim=50
train_dataset_obj = HsiDataset(config.TRAIN_DIR)
dataset = train_dataset_obj.getitem(latent_dim)
#dataset = normalize(dataset)
dataset1 = resize(dataset)
shape_of_dataset = (1,144,144,216,1)
dataset1 = tf.convert_to_tensor(dataset1)
dataset1 = tf.reshape(dataset1,shape_of_dataset)
d = (dataset1 + 1)*127.5
I = d.numpy()[0,:,:,0,0].astype(np.uint8)
#I8 = (((I-I.min())/(I.max()-I.min()))*255.9).astype(np.uint8)
img = Image.fromarray(I)
img = img.convert("L")
img.save("original.png")
#dataset = load_real_samples()
# create the critic
#print(dataset.shape)
# create the generator
cbk = GNNOutput(latent_dim=latent_dim)
generator = generator()
gnn=GNN(generator = generator)
gnn.compile()
gnn.fit(dataset1, batch_size=1, epochs = 20000, callbacks = [cbk])