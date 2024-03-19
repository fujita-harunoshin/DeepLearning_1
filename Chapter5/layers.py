import sys, os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Chapter3.activation_function import softmax
from Chapter4.loss_function import cross_entropy_error
import numpy as np

class Relu:
  def __init__(self) -> None:
    """ReLUレイヤー
    """
    self.mask = None

  def forward(self, x):
    """順伝播

    Args:
        x (numpy.ndarray): 入力

    Returns:
        numpy.ndarray: 出力
    """
    self.mask = (x <= 0)
    out = x.copy()
    out[self.mask] = 0

    return out

  def backward(self, dout):
    """逆伝播

    Args:
        dout (numpy.ndarray): 右の層から伝わる微分値

    Returns:
        numpy.ndarray: 微分値
    """
    dout[self.mask] = 0
    dx = dout

    return dx

class Sigmoid:
  def __init__(self) -> None:
    """Sigmoidレイヤー
    """
    self.out = None

  def forward(self, x):
    """順伝播

    Args:
        x (numpy.ndarray): 入力

    Returns:
        numpy.ndarray: 出力
    """
    out = 1 / (1 + np.exp(-x))
    self.out = out

    return out

  def backward(self, dout):
    """逆伝播

    Args:
        dout (numpy.ndarray): 右の層から伝わる微分値

    Returns:
        numpy.ndarray: 微分値
    """
    dx = dout * (1.0 - self.out) * self.out
    return dx

class Affine:
  def __init__(self, W, b) -> None:
    """Affineレイヤ

    Args:
        W (numpy.ndarray): 重み
        b (numpy.ndarray): バイアス
    """
    self.W = W
    self.b = b
    self.x = None
    self.dW = None
    self.db = None

  def forward(self, x):
    """順伝播

    Args:
        x (numpy.ndarray): 入力

    Returns:
        numpy.ndarray: 出力
    """
    self.x = x
    out = np.dot(x, self.W) + self.b

    return out

  def backward(self, dout):
    """逆伝播

    Args:
        dout (numpy.ndarray): 右の層から伝わる微分値

    Returns:
        numpy.ndarray: 微分値
    """
    dx = np.dot(dout, self.W.T)
    self.dW = np.dot(self.x.T, dout)
    self.db = np.sum(dout, axis=0)

    return dx

class SoftmaxWithLoss:
  def __init__(self) -> None:
    """Softmax-with-Lossレイヤ
    """
    self.loss = None # 損失
    self.y = None    # softmaxの出力
    self.t = None    # 教師データ (one-hot vector)

  def forward(self, x, t):
    """順伝播

    Args:
        x (numpy.ndarray): 入力
        t (numpy.ndarray): 教師データ

    Returns:
        float: 交差エントロピー誤差
    """
    self.t = t
    self.y = softmax(x)
    self.loss = cross_entropy_error(self.y, self.t)

    return self.loss

  def backward(self, dout=1):
    """逆伝播

    Args:
        dout (int, optional): 右の層から伝わる微分値. Defaults to 1.

    Returns:
        numpy.ndarray: 微分値
    """
    batch_size = self.t.shape[0]
    dx = (self.y - self.t) * (dout / batch_size)

    return dx

