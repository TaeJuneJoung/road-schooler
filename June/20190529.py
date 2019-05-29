import tensorflow as tf
import numpy as np

m1 = [[1.0, 2.0],
    [3.0, 4.0]]
m2 = np.array([[1.0, 2.0],
        [3.0, 4.0]], dtype=np.float32)
m3 = tf.constant([[1.0, 2.0],
        [3.0, 4.0]], name="TEST") #Const라는 이름을 name을 통해 TEST라 변경

print(type(m1)) #<class 'list'>
print(type(m2)) #<class 'numpy.ndarray'>
print(type(m3)) #<class 'tensorflow.python.framework.ops.Tensor'>

print(m1)
print(m2)
print(m3)

t1 = tf.convert_to_tensor(m1, dtype=tf.float32)
t2 = tf.convert_to_tensor(m2, dtype=tf.float32)
t3 = tf.convert_to_tensor(m3, dtype=tf.float32)

print(type(t1))
print(type(t2))
print(type(t3))

print(t1)
print(t2)
print(t3)