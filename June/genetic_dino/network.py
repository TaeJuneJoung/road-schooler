import numpy as np

class Network():
    def __init__(self):
        #이렇게 설정해야 하는 이유와 w1,2,3의 설정을 저렇게 하는 이유
        self.input_size = 2
        self.hidden_size = 8
        self.hidden_half_size = 4
        self.output_size = 1
        self.w1 = np.random.randn(self.input_size, self.hidden_size)
        self.w2 = np.random.randn(self.hidden_size, self.hidden_half_size)
        self.w3 = np.random.randn(self.hidden_half_size, self.output_size)
        self.fitness = 0

    #아래 함수들 작성이유를 파악필요
    def forward(self, inputs):
        z1 = np.dot(inputs, self.w1)
        a1 = np.tanh(z1)
        z2 = np.dot(a1, self.w2)
        a2 = np.tanh(z2)
        z3 = np.dot(a2, self.w3)
        yHat = np.tanh(z3)
        return yHat

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def relu(self, z):
        return z * (z>0)