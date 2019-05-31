# Logistic Regression/Classification
import numpy as np
import tensorflow as tf 
import tensorflow.contrib.eager as tfe 
tf.enable_eager_execution()

# data
x_train = [
    [1., 2.], [2., 3.],
    [3., 1.], [4., 3.],
    [5., 3], [6., 2.]
]
y_train = [
    [0.], [0.],
    [0.], [1.], 
    [1.], [1.]
]

x_test = [[5., 2.]]
y_test = [[1.]]

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(len(x_train))
W = tf.Variable(tf.zeros([2, 1]), name='weight')
b = tf.Variable(tf.zeros([1]), name='bias')

# Sigmoid 함수를 가설로 선언
def logistic_regression(features):
    hypothesis = tf.div(1., 1. + tf.exp(tf.matmul(features, W) + b))
    return hypothesis

# 가설을 검증할 Cost함수 정의
def loss_fn(hypothesis, features, labels):
    cost = -tf.reduce_mean(labels * tf.log(logistic_regression(features)) + (1 - labels) * tf.log(1 - hypothesis))
    return cost

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

# 추론한 값은 0.5를 기준으로 0과 1의 값을 리턴
def accuracy_fn(hypothesis, labels):
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, labels), dtype=tf.int32))
    return accuracy

# GradientTape를 통해 경사값 계산
def grad(hypothesis, features, labels):
    with tf.GradientTape() as tape:
        loss_value = loss_fn(logistic_regression(features),features,labels)
    return tape.gradient(loss_value, [W, b])


# Eager모드에서 학습을 실행
EPOCHS = 1001
for step in range(EPOCHS):
    for features, labels in tfe.Iterator(dataset):
        grads = grad(logistic_regression(features), features, labels)
        optimizer.apply_gradients(grads_and_vars=zip(grads, [W, b]))
        if step % 100 == 0:
            print("Iter: {}, Loss: {:.4f}".format(step, loss_fn(logistic_regression(features),features, labels)))

test_acc = accuracy_fn(logistic_regression(x_test), y_test)
print("Testset Accuracy: {:.4f}".format(test_acc))