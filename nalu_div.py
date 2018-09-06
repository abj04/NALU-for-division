import numpy as np
import tensorflow as tf

def nalu(inputs,out_units,sigma):
	
	in_features = inputs.shape[1]
		
	W_hat = tf.get_variable(name = "W_hat", initializer=tf.initializers.random_uniform(minval=-2, maxval=2),shape=[in_features, out_units],  trainable=True)
	M_hat = tf.get_variable(name = "M_hat", initializer=tf.initializers.random_uniform(minval=-2, maxval=2), shape=[in_features, out_units], trainable=True)
	
	G = tf.get_variable(name = "G", initializer=tf.initializers.random_uniform(minval=-2, maxval=2),shape=[in_features, out_units],  trainable=True)

	W = tf.tanh(W_hat) * tf.sigmoid(M_hat)
	
	a = tf.matmul(inputs,W)
	m = tf.exp(tf.matmul(tf.log(tf.abs(inputs) + sigma), W))

	g = tf.sigmoid(tf.matmul(inputs,G))

	out = g * a + (1-g) * m

	return out, W

#############   NALU DEFINED ###########

def generate_dataset(size = 10000, task = 'in'):
	if task == 'in':
		factor = 100
	else:
		factor = 1000
	
	X = factor * np.random.ranf(size = (size, 2)) + 100 * np.random.ranf(size = (size, 2)) + np.random.ranf(size = (size, 2))
	Y = np.zeros(shape = (size,1))
	for i in range(X.shape[0]):
		Y[i] = (X[i,0]/X[i,1])
	return X,Y

#########################

if __name__ == '__main__':
	EPOCHS = 100000
	LEARNING_RATE = 1e-3
	SIGMA = 1e-7


	X_data, Y_data = generate_dataset()
	X_test, Y_test = generate_dataset(size = 200,task = 'ex')

	X = tf.placeholder(tf.float32, shape = [None,2])
	Y = tf.placeholder(tf.float32, shape = [None,1])

	Y_pred, W = nalu(X,1,SIGMA)

	loss = tf.losses.mean_squared_error(labels = Y, predictions = Y_pred)
	optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

	with tf.Session() as sess:
		cost_history = []
		sess.run(tf.global_variables_initializer())

		print('Pre training MSE: ',sess.run(loss, feed_dict = {X : X_data, Y : Y_data}))
		print()

		for i in range(EPOCHS):
			_,cost = sess.run([optimizer, loss], feed_dict = {X : X_data, Y : Y_data})
			print('epoch: {}, MSE: {}'.format(i,cost))
			cost_history.append(cost)

		print()
		print(W.eval)
		print()


		print("post training MSE", sess.run(loss, feed_dict = {X:X_test, Y:Y_test}))

		print("Actual Sum: ",Y_test[0:20])
		print()
		predictions = sess.run(Y_pred[0:20], feed_dict = {X:X_test, Y: Y_test})
		print("predicted sum: ", predictions)

		print("Difference: ", predictions - Y_test[0:20] )
 		
