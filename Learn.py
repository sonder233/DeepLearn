import numpy as np
import matplotlib.pylab as plt


def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1 * w1 + x2 * w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1


def AND_b(x1, x2):
    x = np.array([x1, x2])  # 输入
    w = np.array([0.5, 0.5])  # 权重
    b = -0.7  # 偏置
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


# 与非门
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND_b(s1, s2)
    return y


def step_function_simple(x):  # x是一个numpy.array数组
    y = x > 0  # 对比数组的每个值是否大于0
    # y是一个numpy.ndarray数组 在这里>0置TRUE <0置FALSE
    # print(type(y))
    # print(y)
    return y.astype(np.int)  # 将布尔型转换为int型 TRUE = 1; FALSE = 0


def step_function(x):
    return np.array(x > 0, dtype=np.int)  # 综合上面的两句 结果和上面一样


def draw():
    x = np.arange(-5.0, 5.0, 0.1)
    y = step_function(x)
    plt.plot(x, y)  # 设置x y轴的值
    plt.ylim(-0.1, 1.1)  # 指定y轴的范围
    plt.show()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # Numpy的广播功能：标量和Numpy数组之间进行运算 则数组每个元素都会进行该计算


def draw_sigmoid():
    x = np.arange(-5.0, 5.0, 0.1)
    y = sigmoid(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)  # 指定y轴的范围
    plt.show()


def relu(x):  # ReLU函数：大于0时返回x；小于0时返回0
    return np.maximum(0, x)


def draw_relu():
    x = np.arange(-5.0, 5.0, 0.1)
    y = relu(x)
    plt.plot(x, y)
    plt.ylim(-1, 5)
    plt.show()


def arrayMult():
    X = np.array([1, 2])
    W = np.array([[1, 3, 5], [2, 4, 6]])
    Y = np.dot(X, W)
    return Y


def identity_func(x):
    return x


def layersMult():  # 三层网络的实现
    """第0层-第1层的传递"""
    X = np.array([1.0, 0.5])
    W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    B1 = np.array([0.1, 0.2, 0.3])
    A1 = np.dot(X, W1) + B1
    print(A1)  # [0.3 0.7 1.1]

    Z1 = sigmoid(A1)  # 使用激活函数激活
    print(Z1)  # [0.3 0.7 1.1]
    """第0层-第1层的传递"""

    """第1层-第2层的传递"""
    W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    B2 = np.array([0.1, 0.2])
    A2 = np.dot(Z1, W2) + B2
    Z2 = sigmoid(A2)
    """第1层-第2层的传递"""

    """第2层-第3层的传递"""
    W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
    B3 = np.array([0.1, 0.2])

    A3 = np.dot(Z2, W3) + B3
    Y = identity_func(A3)

    print(Y)  # [0.31682708 0.69627909]
    """第2层-第3层的传递"""


"""下面是上述的整理，也是多层神经网络的标准写法"""


def init_network():  # 进行权重和偏置的初始化
    network = {}  # 返回这个字典
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network


def forward(network, x):  # 将输入信号转换成输出信号的处理过程
    # 这里用forward是因为 是输入-输出方向，之后会学习输出-输入方向（也就是训练？）

    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    z3 = np.dot(z2, W3) + b3
    y = identity_func(z3)

    return y


def threeNeural():
    network = init_network()
    x = np.array([1.0, 0.5])
    y = forward(network, x)
    print(y)


if __name__ == '__main__':
    # print(AND(0, 1))
    # print(AND_b(0, 1))
    # print(NAND(0, 1))
    # print(OR(0, 0))
    # print(XOR(0, 0))
    # step_function(np.array([-1.0, 2.0, 2.0]))
    # draw()
    # draw_sigmoid()
    # draw_relu()
    # print(arrayMult())
    # layersMult()
    threeNeural()
    pass

