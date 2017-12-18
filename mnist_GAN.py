import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('C:/tmp/tensorflow/mnist/input_data')

def get_inputs(real_size, noise_size):
    """
    real_img and noise_img
    """
    real_img = tf.placeholder(tf.float32, [None, real_size], name='real_img')
    noise_img = tf.placeholder(tf.float32, [None, noise_size], name='noise_img')
    
    return real_img, noise_img

def get_generator(noise_img, n_units, out_dim, reuse=False, Rect=0.01):
    """
    generator
    
    noise_img: input of generator
    n_units: neurons in hidden layer
    out_dim: output dimension = 32 x 32 = 784
    Rect: Leaky Rectifier activation function parameter
    """
    with tf.variable_scope("generator", reuse=reuse):
        # hidden layer
        hidden1 = tf.layers.dense(noise_img, n_units)
        # Leaky Rectifier activation function
        hidden1 = tf.maximum(Rect * hidden1, hidden1)
        # dropout
        hidden1 = tf.layers.dropout(hidden1, rate=0.2)

        # logits & outputs
        logits = tf.layers.dense(hidden1, out_dim)
        outputs = tf.tanh(logits)
        
        return logits, outputs


def get_discriminator(img, n_units, reuse=False, Rect=0.01):
    """
    discriminator
    
    n_units: neurons in hidden layer
    Rect: Leaky Rectifier activation function parameter
    """
    
    with tf.variable_scope("discriminator", reuse=reuse):
        # hidden layer
        hidden1 = tf.layers.dense(img, n_units)
        hidden1 = tf.maximum(Rect * hidden1, hidden1)
        
        # logits & outputs
        logits = tf.layers.dense(hidden1, 1)
        outputs = tf.tanh(logits)
        
        return logits, outputs

# actual size of images
img_size = mnist.train.images[0].shape[0]
# input noise size 
noise_size = 100
# hidden neurons for generator
g_units = 128
# hidden neurons for discriminator
d_units = 128
# Rectifier activation function parameter
Rect = 0.01
# learning_rate
learning_rate = 0.001
# label smoothing
smooth = 0.1

tf.reset_default_graph()

real_img, noise_img = get_inputs(img_size, noise_size)

# generator
g_logits, g_outputs = get_generator(noise_img, g_units, img_size)

# discriminator
d_logits_real, d_outputs_real = get_discriminator(real_img, d_units)
d_logits_fake, d_outputs_fake = get_discriminator(g_outputs, d_units, reuse=True)

# error of discriminator
# fail to recognize real images
d_err_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, 
                                                                     labels=tf.ones_like(d_logits_real)) * (1 - smooth))
# fail to recognize fake images
d_err_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, 
                                                                     labels=tf.zeros_like(d_logits_fake)))
# over all error
d_err = tf.add(d_err_real, d_err_fake)

# error of generator
g_err = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                labels=tf.ones_like(d_logits_fake)) * (1 - smooth))


train_vars = tf.trainable_variables()

g_vars = [var for var in train_vars if var.name.startswith("generator")]
d_vars = [var for var in train_vars if var.name.startswith("discriminator")]

# optimizer
d_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(d_err, var_list=d_vars)
g_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(g_err, var_list=g_vars)

# batch_size
batch_size = 64
# number of epochs
epochs = 300
# number of samples
n_sample = 25

samples = []
errs = []
# training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        for batch_i in range(mnist.train.num_examples//batch_size):
            batch = mnist.train.next_batch(batch_size)
            
            batch_images = batch[0].reshape((batch_size, 784))
            # scale pixels in images
            batch_images = batch_images*2 - 1
            
            # random noise input for generator
            batch_noise = np.random.uniform(-1, 1, size=(batch_size, noise_size))
            
            # run optimizers
            _ = sess.run(d_train_opt, feed_dict={real_img: batch_images, noise_img: batch_noise})
            _ = sess.run(g_train_opt, feed_dict={noise_img: batch_noise})
        
        # error calcuation
        train_err_d = sess.run(d_err, 
                                feed_dict = {real_img: batch_images, 
                                             noise_img: batch_noise})
        # real img err
        train_err_d_real = sess.run(d_err_real, 
                                     feed_dict = {real_img: batch_images, 
                                                 noise_img: batch_noise})
        # fake img err
        train_err_d_fake = sess.run(d_err_fake, 
                                    feed_dict = {real_img: batch_images, 
                                                 noise_img: batch_noise})
        # generator err
        train_err_g = sess.run(g_err, 
                                feed_dict = {noise_img: batch_noise})
        
            
        print("Epoch {}/{}...".format(e+1, epochs),
              "Discriminator Loss: {:.4f}(Real: {:.4f} + Fake: {:.4f})...".format(train_err_d, train_err_d_real, train_err_d_fake),
              "Generator Loss: {:.4f}".format(train_err_g))    
        errs.append((train_err_d, train_err_d_real, train_err_d_fake, train_err_g))
        
        sample_noise = np.random.uniform(-1, 1, size=(n_sample, noise_size))
        gen_samples = sess.run(get_generator(noise_img, g_units, img_size, reuse=True),
                               feed_dict={noise_img: sample_noise})
        samples.append(gen_samples)
        

with open('train_samples.pkl', 'wb') as f:
    pickle.dump(samples, f)
	
# Load samples from generator taken while training
with open('train_samples.pkl', 'rb') as f:
    samples = pickle.load(f)

def view_samples(epoch, samples):
    """
    epoch is the number of iterations
    samples are the images we pick to plot
    """
    fig, axes = plt.subplots(figsize=(7,7), nrows=5, ncols=5, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch][1]):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.imshow(img.reshape((28,28)), cmap='Greys_r')
    return fig, axes

_ = view_samples(-1, samples)

epoch_idx = [0, 5, 10, 20, 40, 60, 80, 100, 150, 250]
show_imgs = []
for i in epoch_idx:
    show_imgs.append(samples[i][1])


rows, cols = 10, 25
fig, axes = plt.subplots(figsize=(30,12), nrows=rows, ncols=cols, sharex=True, sharey=True)

idx = range(0, epochs, int(epochs/rows))

for sample, ax_row in zip(show_imgs, axes):
    for img, ax in zip(sample[::int(len(sample)/cols)], ax_row):
        ax.imshow(img.reshape((28,28)), cmap='Greys_r')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
plt.show()