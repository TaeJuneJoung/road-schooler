# cost function in pure python 
import numpy as np 

X = np.array([1, 2, 3])
Y = np.array([1, 2, 3])

# cost(W) 식
# 에러의 제곱의 평균
def cost_func(W, X, Y):
    c = 0
    for i in range(len(X)):
        c += (W * X[i] - Y[i]) ** 2
    return c/len(X)

for feed_W in np.linspace(-3, 5, num=15):
    curr_cost = cost_func(feed_W, X, Y)
    print("{:6.3f} | {:10.5f}".format(feed_W, curr_cost))
    