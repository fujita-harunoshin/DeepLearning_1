import sys, os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Chapter3.activation_function import softmax, sigmoid
from Chapter4.loss_function import cross_entropy_error
from Chapter4.gradient import numerical_gradient
import numpy as np

class TwoLayerNet:

  def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01) -> None:
    """2層のニューラルネットワーク

    Args:
        input_size (int): 入力層のニューロン数
        hidden_size (int): 隠れ層のニューロン数
        output_size (int): 出力層のニューロン数
        weight_init_std (float, optional): 重みの初期値の調整パラメータ. Defaults to 0.01.
    """
    self.params = {}
    self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def predict(self, x):
    """ニューラルネットワークによる推論

    Args:
        x (numpy.ndarray): 入力

    Returns:
        numpy.ndarray: 出力
    """
    W1, W2 = self.params['W1'], self.params['W2']
    b1, b2 = self.params['b1'], self.params['b2']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    y = softmax(a2)

    return y

  def loss(self, x, t):
    """損失関数の値算出

    Args:
        x (numpy.ndarray): ニューラルネットワークへの入力
        t (numpy.ndarray): 教師データ

    Returns:
        float: 損失関数の値
    """
    y = self.predict(x)
    loss = cross_entropy_error(y, t)

    return loss

  def accuracy(self, x, t):
    """認識精度の算出

    Args:
        x (numpy.ndarray): ニューラルネットワークへの入力
        t (numpy.ndarray): 教師データ

    Returns:
        float: 認識精度
    """
    y = self.predict(x)
    y = np.argmax(y, axis=1)
    t = np.argmax(t, axis=1)

    accuracy = np.sum(y == t) / x.shape[0]
    return accuracy

  def numerical_gradient(self, x, t):
    """重みパラメータに対する勾配の算出

    Args:
        x (numpy.ndarray): ニューラルネットワークへの入力
        t (numpy.ndarray): 教師データ

    Returns:
        dictionary: 勾配を格納した辞書
    """
    loss_W = lambda W: self.loss(x, t)
    grads = {}
    grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
    grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
    grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
    grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

    return grads
