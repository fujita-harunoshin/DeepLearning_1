import numpy as np

def sum_squared_error(y, t):
  """二乗和誤差

  Args:
      y (numpy.ndarray): ニューラルネットワークの出力
      t (numpy.ndarray): 教師データ

  Returns:
      float: 二乗和誤差
  """
  return 0.5 * np.sum((y-t)**2)

def cross_entropy_error(y, t):
  """交差エントロピー誤差

  Args:
      y (numpy.ndarray): ニューラルネットワークの出力
      t (numpy.ndarray): 教師データ

  Returns:
      float: 交差エントロピー誤差
  """
  if y.ndim == 1:
    t = t.reshape(1, t.size)
    y = y.reshape(1, y.size)

  batch_size = y.shape[0]
  delta = 1e-7
  return -np.sum(t * np.log(y + delta)) / batch_size