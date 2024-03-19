import sys, os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Chapter4.gradient import numerical_gradient
from Chapter5.layers import *
import numpy as np
from dataset.mnist import load_mnist

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

    self.layers = {}
    self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
    self.layers['Relu1'] = Relu()
    self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

    self.lastLayer = SoftmaxWithLoss()

  def predict(self, x):
    """推論

    Args:
        x (numpy.ndarray): 入力

    Returns:
        numpy.ndarray: 出力
    """
    for layer in self.layers.values():
      x = layer.forward(x)

    return x

  def loss(self, x, t):
    """損失関数の値算出

    Args:
        x (numpy.ndarray): 入力
        t (numpy.ndarray): 正解ラベル

    Returns:
        float: 損失関数の値
    """
    y = self.predict(x)

    loss = self.lastLayer.forward(y, t)

    return loss

  def accuracy(self, x, t):
    """認識精度算出

    Args:
        x (numpy.ndarray): ニューラルネットワークへの入力
        t (numpy.ndarray): 正解ラベル

    Returns:
        float: 認識精度
    """
    y = self.predict(x)
    y = np.argmax(y, axis=1)
    t = np.argmax(t, axis=1)

    accuracy = np.sum(y == t) / x.shape[0]
    return accuracy

  def numerical_gradient(self, x, t):
    """重みパラメータに対する購買を数値微分で算出

    Args:
        x (numpy.ndarray): ニューラルネットワークへの入力
        t (numpy.ndarray): 正解ラベル

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

  def gradient(self, x, t):
    """重みパラメータに対する勾配を誤差逆伝播法で算出

    Args:
        x (numpy.ndarray): ニューラルネットワークへの入力
        t (numpy.ndarray): 正解ラベル

    Returns:
        dictionary: 勾配を格納した辞書
    """
    # 順伝播
    self.loss(x, t)

    # 逆伝播
    dout = self.lastLayer.backward()
    for layer in reversed(list(self.layers.values())):
      dout = layer.backward(dout)

    # 各レイヤの微分値取り出し
    grads = {}
    grads['W1'] = self.layers['Affine1'].dW
    grads['b1'] = self.layers['Affine1'].db
    grads['W2'] = self.layers['Affine2'].dW
    grads['b2'] = self.layers['Affine2'].db

    return grads