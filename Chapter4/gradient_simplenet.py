import sys, os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Chapter3.activation_function import softmax
from Chapter4.loss_function import cross_entropy_error
from Chapter4.gradient import numerical_gradient
import numpy as np

class simpleNet:
  def __init__(self) -> None:
    """重みパラメータ初期化
    """
    self.W = np.random.randn(2,3)

  def predict(self, x):
    """ニューラルネットワークによる推論

    Args:
        x (numpy.ndarray): 入力

    Returns:
        numpy.ndarray: 出力
    """
    return np.dot(x, self.W)

  def loss(self, x, t):
    """損失関数の値算出

    Args:
        x (numpy.ndarray): ニューラルネットワークへの入力
        t (numpy.ndarray): 教師データ

    Returns:
        float: 損失関数の値
    """
    z = self.predict(x)
    y = softmax(z)
    loss = cross_entropy_error(y, t)

    return loss

net = simpleNet()
print(net.W)

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)