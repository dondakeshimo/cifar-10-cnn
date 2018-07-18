
# coding: utf-8

# # Tensorflowを用いたCNNの実装

# このハンズオンでは，CNN (Convolutional Neural Network, 畳み込みニューラルネットワーク)をTensorflowを用いて実装し，手書き数字データセットMNISTの識別を行います．

# # 目次
# 
# - [準備](#準備)
# - [CNNの解説](#CNNの解説)
# - [ハンズオン - CNNの実装](#ハンズオン - CNNの実装)

# <a name="準備"></a>
# # 準備

# ### Tensorflowのインストール

# In[ ]:

get_ipython().system('sudo pip install tensorflow-gpu==1.2')


# ### 必要なライブラリのロード

# In[1]:

#import tensorflow as tf

from __future__ import print_function

import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split


# ### データの準備
# 
# 前回までと同様，MNISTのデータを読み込み，訓練データ，テストデータに分割します．

# In[2]:

# データのロード
mnist = fetch_mldata('MNIST original', data_home='./data/')

# data : 画像データ， target : 正解ラベル
X, T = mnist.data, mnist.target

# 画像データは0~255の数値のなっているので，0~1の値に変換
X = X / 255.

#　訓練データとテストデータに分ける
X_train, X_test, T_train, T_test = train_test_split(X, T, test_size=0.2)

# データのサイズ
N_train = X_train.shape[0]
N_test = X_test.shape[0]

# ラベルデータをint型に統一し，学習に使いやすいようにone-hot-vectorに変換
T_train = np.eye(10)[T_train.astype("int")]
T_test = np.eye(10)[T_test.astype("int")]


# In[ ]:

print ('訓練データのサイズは', N_train)
print ('テストデータのサイズは', N_test)
print ('画像データのshapeは', X_train.shape)
print ('ラベルデータのshapeは', T_train.shape)
print ('ラベルデータの数値の例：')
print (T_train[:10])


# <a name="CNNの解説"></a>
# # CNNの解説

# ここからは，CNNの実装を行います．
# 
# 実装の前に，CNNの主要な構成要素である**畳み込み層**と**プーリング層**について説明します．

# ## 畳み込み層

# まず，最も基本的な畳み込み層について説明します．
# 
# 畳み込み層では，以下の図のようなフィルタを入力にスライドさせながら適用していきます．
# 
# ここでは，入力画像サイズが(4, 4)，入力チャンネル数1，フィルタサイズ(3, 3)としています．

# <img src="figure/cnn_1.png" width=500>
# <img src="figure/cnn_2.png" width=500>

# ### ストライド
# フィルタを適用する際，上の例では間隔をあけずに適用しましたが，ある間隔をあけながらフィルタを適用する場合があります．
# 
# この間隔を**ストライド**と呼びます．
# 上の例ではストライドは1ですが，下のように1つずつ間隔をあけてフィルタを適用した場合，ストライドは2となります．

# <img src="figure/stride.png" width=500>

# ### パディング
# 今まで見てきたようにフィルタを適用していくと，出力サイズは徐々に小さくなっていきます．
# 
# そこで，出力サイズを調整するために，**パディング**という操作を行うことがあります．
# 
# パディングとは，入力の周囲をゼロなどのデータで埋める操作のことです．
# 
# 下の図では，パディング幅1のゼロパディングを行っています．

# <img src="figure/padding.png" width=300>

# ### 出力サイズの計算
# 
# 畳み込み層のフィルタを適用した後の出力サイズは，以下の式で計算できます．
# 
# 
# $$
# W_O = \frac{W_I + 2P - W_F}{S} + 1
# $$
# 
# $$
# H_O = \frac{H_I + 2P - H_F}{S} + 1
# $$
# 
# ここで，$(W_O, H_O)$は出力の幅と高さ，$(W_I, H_I)$は入力の幅と高さ，$P$はパディング幅，$S$はストライド幅を指しています．

# ### フィルタ数
# 
# 複数のフィルタを同じ入力に適用することで，出力を複数チャンネルにすることができます．

# ## Tensorflowでの実装

# Tensorflowで畳み込み層を実装する場合には，```tf.layers.conv2d```関数を用います．（```tf.nn.conv2d```関数を用いることもできますが，ここでは割愛します）
# 
# 基本的な引数は以下のとおりです．
# 
# ---
# 
# ```python
# tf.layers.conv2d(
#     inputs,
#     filters,
#     kernel_size,
#     strides=(1,1),
#     padding='valid',
#     activation=None
# )
# ```
# ---
# 
# - inputs: 入力データ
# - filters: フィルタ数 (=出力チャンネル数)
# - kernel_size: フィルタサイズ
# - strides: ストライド幅．整数で指定すると縦横に同じストライド幅が適用されます．
# - padding: "valid"か"same"のどちらかを指定します．"valid"ではパディングなし，"same"では入力と出力のサイズが等しくなるようにゼロパディングが行われます．
# - activation: 活性化関数を指定します．```tf.nn.relu```など．
# 
# 例えば，フィルタ数50，フィルタサイズ(5, 5)，ストライド幅2，ゼロパディング，活性化関数にReLU関数を用いる場合は，以下のように定義されます．
# 
# ```python
# tf.layers.conv2d(inputs, filters=50, kernel_size=[5, 5], strides=[2, 2], padding='same', activation=tf.nn.relu)
# ```

# ## プーリング層

# プーリングの方法として，フィルタ内の最大値を返すMaxプーリング，平均値を返すAverageプーリングなどがあります．
# 
# プーリング層でも，畳み込み層と同様，ストライドやパディングが行われます．
# 
# 下の図では，ストライド幅2，フィルターサイズ(2, 2)のMaxプーリングを適用した例を示しています．

# <img src="figure/pooling2_1.png" width=500>

# <img src="figure/pooling2_2.png" width=500>

# ### 出力サイズの計算
# 
# プーリング層のフィルタを適用した後の出力サイズは，畳み込み層と同様，以下の式で計算できます．
# 
# 
# $$
# W_O = \frac{W_I + 2P - W_F}{S} + 1
# $$
# 
# $$
# H_O = \frac{H_I + 2P - H_F}{S} + 1
# $$
# 
# ここで，$(W_O, H_O)$は出力の幅と高さ，$(W_I, H_I)$は入力の幅と高さ，$P$はパディング幅，$S$はストライド幅を指しています．

# ## Tensorflowでの実装

# Tensorflowでプーリングを実装するためには，```tf.layers.max_pooling2d```, ```tf.layers.average_pooling2d```などを用います．
# 
# ```tf.layers.max_pooling2d``` の主な引数は，次のようになっています．
# 
# ---
# ```python
# tf.layers.max_pooling2d(
#     inputs,
#     pool_size,
#     strides,
#     padding='valid'
# )
# ```
# ---
# 
# - inputs: 入力データ
# - pool_size: フィルタサイズ
# - strides: ストライド幅．整数で指定すると縦横に同じストライド幅が適用されます．
# - padding: "valid"か"same"のどちらかを指定します．"valid"ではパディングなし，"same"では入力と出力のサイズが等しくなるようにゼロパディングが行われます．
# 
# 例えば，フィルタサイズ2，ストライド幅2, パディングなしのMaxプーリング層は，以下のように定義できます．
# 
# ```python
# tf.layers.max_pooling2d(inputs, pool_size=[2, 2], strides=[2, 2], padding='valid')
# ```

# <a name="ハンズオン - CNNの実装"></a>
# # ハンズオン - CNNの実装

# 以上を参考に，実際にCNNを構築していきます．
# 
# ここでは，下図のような構造のCNNを定義します．

# <img src="figure/cnn_pipeline.png" width=400>

# #### 実装上の注意
# ここで，最後から2番目の全結合層に入力する前に，```tf.reshape(tensor, shape)```を用いて，出力された特徴マップを(バッチサイズ，特徴マップの幅×特徴マップの高さ×チャンネル数)の2次元テンソルに変換します．
# 
# そこで，上に書いた計算式を用いて，全結合までの特徴マップのサイズを計算し，適切な値を指定する必要があります．
# 
# 全結合層の実装は，前回のハンズオンで用いた```tf.layers.dense()```を用います．

# In[ ]:

def CNN(x):
   # 入力を(バッチサイズ，28, 28, チャンネル数)の形にreshape
    input_layer = tf.reshape(x, [-1, 28, 28, 1])
    
     ### TODO
    conv1 = tf.layers.conv2d(input_layer, filters=20, kernel_size=5, padding="valid", activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)
    conv2 = tf.layers.conv2d(pool1, filters=50, kernel_size=[5, 5], padding="valid", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2)
    
    # 2次元テンソルに変換
    pool2_flat = tf.reshape(pool2, [-1, 800])
    fc3 = tf.layers.dense(pool2_flat, 500, activation=tf.nn.relu)
    out = tf.layers.dense(fc3, 10)
    
    ### TODO
    
    return out


# ## グラフの構築
# 
# ここは前回とほとんど同じです．
# 
# 前回はMLPクラスを用いていた部分を，CNNクラスに置き換えています．

# In[ ]:

tf.reset_default_graph()

# パラメータ
# Learning rate (学習率)
lr = 0.1
# epoch数 （学習回数）
n_epoch = 25
# ミニバッチ学習における1バッチのデータ数
batchsize = 100

# 入力
# placeholderを用いると，データのサイズがわからないときにとりあえずNoneとおくことができる．
x = tf.placeholder(tf.float32, [None, 784]) # 28*28次元 
t = tf.placeholder(tf.float32, [None, 10]) # 10クラス

# CNNクラスのモデルを用いてpredictionを行う
y = CNN(x)

# 目的関数:softmax cross entropy
# 入力：labels->正解ラベル， logits：predictionの結果
# 出力：softmax cross entropyで計算された誤差
xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=t, logits=y)
cost = tf.reduce_mean(xentropy)

# SGD(Stochastic Gradient Descent : 確率的勾配降下法)で目的関数を最小化する
optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cost)

# test用
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# ## グラフの実行
# 
# 前回と同様，構築したグラフを実行してみます．
# 
# 正しく実装できていれば，99%程度の正解率となるはずです．

# In[ ]:

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(n_epoch):
        print ('epoch %d | ' % epoch, end="")

        # Training
        sum_loss = 0
        # 訓練データをシャッフルする
        perm = np.random.permutation(N_train)

        for i in range(0, N_train, batchsize):
            # ミニバッチ分のデータを取ってくる
            X_batch = X_train[perm[i:i+batchsize]]
            t_batch = T_train[perm[i:i+batchsize]]

            _, loss = sess.run([optimizer, cost], feed_dict={x:X_batch, t:t_batch})
            sum_loss += loss * X_batch.shape[0]

        loss = sum_loss / N_train
        print('Train loss %.5f | ' %(loss), end="")

        # Test model
        print ("Test Accuracy: %.3f"%(accuracy.eval(feed_dict={x: X_test, t: T_test})))


# In[ ]:



