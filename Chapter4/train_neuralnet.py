import numpy as np
import matplotlib.pylab as plt
import os, sys
from two_layer_net import TwoLayerNet
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from dataset.mnist import load_mnist

# MNISTの訓練データとテストデータの読み込み
(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, one_hot_label=True)

# ハイパーパラメータ設定
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

# 結果の記録リスト
train_loss_list = []
train_acc_list = []
test_acc_list = []

# 1エポックあたりの繰り返し数
iter_per_epoch = max(train_size / batch_size, 1)

# 2層のニューラルネットワーク生成
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
  # ミニバッチ生成
  batch_mask = np.random.choice(train_size, batch_size)
  x_batch = x_train[batch_mask]
  t_batch = t_train[batch_mask]

  # 勾配の計算
  grad = network.numerical_gradient(x_batch, t_batch)

  # パラメータの更新
  for key in ('W1', 'b1', 'W2', 'b2'):
    network.params[key] -= learning_rate * grad[key]

  # 学習経過の記録
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

if i % iter_per_epoch == 0:
  train_acc = network.accuracy(x_train, t_train)
  test_acc = network.accuracy(x_test, t_test)
  train_acc_list.append(train_acc)
  test_acc_list.append(test_acc)
  print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))