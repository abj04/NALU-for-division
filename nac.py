# implementation of Neural Accumulator based on DeepMind's paper on NALU

import numpy as np
import tensorflow as tf
import matplotlib as plt

def nac(x_in, out_units):
	print("shape: ",x_in.shape)

	in_features = x_in.shape[1]

	W_hat = tf.get_variable(name = "W_hat", initializer=tf.initializers.random_uniform(minval=-2, maxval=2),shape=[in_features, out_units],  trainable=True)
	M_hat = tf.get_variable(name = "M_hat", initializer=tf.initializers.random_uniform(minval=-2, maxval=2), shape=[in_features, out_units], trainable=True)

	W = tf.nn.tanh(W_hat) * tf.nn.sigmoid(M_hat)

	y_out = tf.matmul(x_in,W)

	return y_out, W



x1 = np.arange(0,10000,5, dtype = np.float32)  
x2 = np.arange(1000,11000,5, dtype = np.float32)

y_train = x1 + x2
x_train = np.column_stack((x1,x2))

print(x_train.shape)
print(y_train.shape)

x1 = np.arange(1000,2000,8, dtype = np.float32)
x2 = np.arange(1000,1500,4, dtype = np.float32)

x_test = np.column_stack((x1,x2))
y_test = x1 + x2

print()
print(x_test.shape)
print(y_test.shape)


X = tf.placeholder(dtype = tf.float32, shape = [None,2])
Y = tf.placeholder(dtype = tf.float32, shape = [None, ])

y_pred, W = nac(X, out_units=1)
y_pred = tf.squeeze(y_pred)

#loss = tf.reduce_mean((y_pred - Y)**2)
loss= tf.losses.mean_squared_error(labels=Y, predictions=y_pred)

alpha = 0.05
epochs = 50000

optimize = tf.train.AdamOptimizer(learning_rate = alpha).minimize(loss)

with tf.Session() as sess:
	cost_history = []
	sess.run(tf.global_variables_initializer())

	print("Pre Training MSE: ",sess.run(loss, feed_dict={X:x_train, Y:y_train}))
	print()

	for i in range(epochs):
		_,cost = sess.run([optimize, loss], feed_dict = {X:x_train, Y: y_train})
		print("epoch: {}, MSE: {}".format(i, cost))
		cost_history.append(cost)
	
	
	#plt.pyplot.plot(np.arange(epochs),np.log(cost_history))
	#plt.xlabel("Epoch")
	#plt.ylabel("MSE")
	#plt.show()

	print()
	print(W.eval())
	print()
	
	#loss= tf.losses.mean_squared_error(labels=y_test, predictions=y_pred)

	print("post training MSE", sess.run(loss, feed_dict = {X:x_test, Y:y_test}))

	print("Actual Sum: ",y_test[0:20])
	print()
	print("predicted sum: ", sess.run(y_pred[0:20], feed_dict = {X:x_test, Y: y_test}))
























