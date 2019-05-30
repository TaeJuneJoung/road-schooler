# matrix를 사용할 때
import tensorflow as tf 
import numpy as np
tf.enable_eager_execution()

data = np.array([
    # x1,  x2,  x3,  y
    [73., 80., 75., 152. ],
    [93., 88., 93., 185. ],
    [89., 91., 90., 180. ],
    [96., 98., 100., 196. ],
    [73., 66., 70., 142. ]
], dtype=np.float32)

# slice data
x = data[:, :-1]
y = data[:, [-1]]

w = tf.Variable(tf.random_normal([3, 1]))
b = tf.Variable(tf.random_normal([1]))

# hypothesis, prediction function
def predict(X):
    return tf.matmul(X, w) + b

learning_rate = 0.000001

n_epochs = 2000
for i in range(n_epochs+1):
    # record the gradient of the cost function
    with tf.GradientTape() as tape:
        cost = tf.reduce_mean((tf.square(predict(x) - y)))

    # calculates the gradients of the cost
    W_grad, b_grad = tape.gradient(cost, [w, b])

    # update parameters (W and b)
    w.assign_sub(learning_rate * W_grad)
    b.assign_sub(learning_rate * b_grad)

    if i % 100 == 0:
        print("{:5} | {:10.4f}".format(i, cost.numpy()))