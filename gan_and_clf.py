import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from matplotlib import pyplot as plt
import scipy.misc
import time
import os
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

n_hidden_nodes = 128 # Number of hidden nodes for generator and discriminator
c_n_hidden_nodes = 1024 # Number of hidden nodes for classifier
batch_size = 100
z_dim = 100
epochs = 100000 # Epochs for generator and discriminator training
clf_epochs = 100 # Epochs for classifier training

def sample_z(m, n):
	return np.random.uniform(-1., 1., [m, n])

def xavier_init(size):
    input_dim = size[0]
    xavier_variance = 1. / tf.sqrt(input_dim/2.)
    return tf.random_normal(shape=size, stddev=xavier_variance)

def classifier(x):
	cl = tf.nn.relu(tf.matmul(x, c_w1) + c_b1)
	cl_logits = tf.matmul(cl, c_w2) + c_b2
	cl_out = tf.nn.softmax(cl_logits)
	return cl_logits, cl_out

def discriminator(x):
	dl = tf.nn.relu(tf.matmul(x, d_w1) + d_b1)
	dl_out = tf.nn.sigmoid(tf.matmul(dl, d_w2) + d_b2)
	return dl_out

def generator(z):
	gl = tf.nn.relu(tf.matmul(z, g_w1) + g_b1)
	gl_out = tf.nn.sigmoid(tf.matmul(gl, g_w2) + g_b2)
	return gl_out

x_placeholder = tf.placeholder(tf.float32, [None, 784])
y_placeholder = tf.placeholder(tf.float32, [None, 10])
z_placeholder = tf.placeholder(tf.float32, [None, z_dim])

# CLASSIFIER
c_w1 = tf.Variable(xavier_init([784, c_n_hidden_nodes]))
c_b1 = tf.Variable(tf.zeros([c_n_hidden_nodes]))
c_w2 = tf.Variable(xavier_init([c_n_hidden_nodes, 10]))
c_b2 = tf.Variable(tf.zeros([10]))
theta_c = [c_w1, c_w2, c_b1, c_b2]

# DISCRIMINATOR
d_w1 = tf.Variable(xavier_init([784, n_hidden_nodes]))
d_b1 = tf.Variable(tf.zeros([n_hidden_nodes]))
d_w2 = tf.Variable(xavier_init([n_hidden_nodes, 1]))
d_b2 = tf.Variable(tf.zeros([1]))
theta_d = [d_w1, d_w2, d_b1, d_b2]

# GENERATOR
g_w1 = tf.Variable(xavier_init([z_dim, n_hidden_nodes]))
g_b1 = tf.Variable(tf.zeros([n_hidden_nodes]))
g_w2 = tf.Variable(xavier_init([n_hidden_nodes, 784]))
g_b2 = tf.Variable(tf.zeros([784]))
theta_g = [g_w1, g_w2, g_b1, g_b2]

g_sample = generator(z_placeholder)

d_real = discriminator(x_placeholder)
d_fake = discriminator(g_sample)

d_loss = -tf.reduce_mean(tf.log(d_real) + tf.log(1. - d_fake))
g_loss = -tf.reduce_mean(tf.log(d_fake))

d_optimizer = tf.train.AdamOptimizer().minimize(d_loss, var_list=theta_d)
g_optimizer = tf.train.AdamOptimizer().minimize(g_loss, var_list=theta_g)

# Classifier's loss and optimizer
cl_logits, cl_out = classifier(x_placeholder)
c_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=cl_logits, labels=y_placeholder))
c_optimizer = tf.train.AdamOptimizer().minimize(c_loss, var_list=theta_c)

# Calculate accuracy and generate labels
correct_prediction = tf.equal(tf.argmax(y_placeholder, 1), tf.argmax(cl_out, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
pred_label = tf.argmax(cl_out, 1)[0]

init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	print('Training classifier...')
	for epoch in range(clf_epochs):
		batch_x, batch_y = mnist.train.next_batch(batch_size)
		sess.run(c_optimizer, {x_placeholder: batch_x, y_placeholder: batch_y})
		if epoch % 20 == 0:
			print(sess.run(accuracy, {x_placeholder: mnist.test.images, y_placeholder: mnist.test.labels}))
	acc = sess.run(accuracy, {x_placeholder: mnist.test.images, y_placeholder: mnist.test.labels})

	print('Classifier trained, accuracy: %s' % acc)
	time.sleep(5)
	print('Generating new images using GAN...')
	
	if not os.path.exists('output/'):
		os.makedirs('output/')
	for epoch in range(epochs):
		batch = mnist.train.next_batch(batch_size)[0]
		sess.run(d_optimizer, {x_placeholder: batch, z_placeholder: sample_z(batch_size, z_dim)})
		sess.run(g_optimizer, {z_placeholder: sample_z(batch_size, z_dim)})
		if epoch % 500 == 0:
			img = sess.run(g_sample, {z_placeholder: sample_z(1, z_dim)})
			print('Epoch %s, Image generated, prediction: %s' % (epoch, sess.run(pred_label, {x_placeholder: img})))
			img = img.reshape([28, 28])
			scipy.misc.imsave('output/' + str(epoch) + '.jpg', img)