import numpy as np

def numerical_diff(f, x):
  """数値微分

  Args:
      f (function): 関数
      x (float): 微分点

  Returns:
      float: 微分結果
  """
  h = 1e-4
  return (f(x+h) - f(x-f)) / (2*h)

def numerical_gradient(f, x):
  """勾配の算出

  Args:
      f (function): 関数
      x (numpy.ndarray): 微分点

  Returns:
      numpy.ndarray: 勾配
  """
  h = 1e-4
  grad = np.zeros_like(x)

  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:
    idx = it.multi_index
    tmp_val = x[idx]
    x[idx] = tmp_val + h
    fxh1 = f(x) # f(x+h)

    x[idx] = tmp_val - h
    fxh2 = f(x) # f(x-h)
    grad[idx] = (fxh1 - fxh2) / (2*h)

    x[idx] = tmp_val # 値を元に戻す
    it.iternext()

  return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100):
  """勾配降下法

  Args:
      f (function): 最適化したい関数
      init_x (numpy.ndarray): 初期値
      lr (float): 学習率. Defaults to 0.01.
      step_num (int): 勾配法による繰り返し数. Defaults to 100.

  Returns:
      numpy.ndarray: 最小値
  """
  x = init_x

  for i in range(step_num):
    grad = numerical_gradient(f, x)
    x -= lr * grad

  return x
