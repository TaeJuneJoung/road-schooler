# gradient descent 
import tensorflow as tf 
import numpy
tf.enable_eager_execution()

# for reproducibility
# 나중에 다시 동작해도 똑같이 동작하게 하기 위해서(재현성)
tf.set_random_seed(0)

X = numpy.array([1, 2, 3])
Y = numpy.array([1, 2, 3])

# 정규분포를 따르는 변수 1개
W = tf.Variable(tf.random_normal([1], -100., 100.))

# 특정한 w값
# 초기에 cost와 w는 다를수 있지만 반복이 진행될 수록 일정한 값이 나옴
# W = tf.Variable([5.0])

for step in range(300):
    hypothesis = W * X
    cost = tf.reduce_mean(tf.square(hypothesis - Y))

    alpha = 0.01
    gradient = tf.reduce_mean(tf.multiply(tf.multiply(W, X) -Y, X))
    descent = W - tf.multiply(alpha, gradient)
    W.assign(descent)

    if step % 10 == 0:
        print('{:5} | {:10.4f} | {:10.6f}'.format(step, cost.numpy(), W.numpy()[0]))