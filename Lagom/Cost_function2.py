# cost function in tensorflow
import tensorflow as tf 
import numpy as np
tf.enable_eager_execution()

X = np.array([1, 2, 3])
Y = np.array([1, 2, 3])

# cost 함수
def cost_func(W, X, Y):
    hypothesis = X * W
    return tf.reduce_mean(tf.square(hypothesis - Y))

W_values = np.linspace(-3, 5, num=15)
cost_values = []

for feed_W in W_values:
    curr_cost = cost_func(feed_W, X, Y)
    cost_values.append(curr_cost)
    print("{:6.3f} | {:10.5f}".format(feed_W, curr_cost))