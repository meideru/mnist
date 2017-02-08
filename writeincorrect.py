import tensorflow as tf
import numpy as np
import sys
import os
from mnist import load_mnist, img_show

# MNISTを読み込む
(x_train, t_train), (x_test, t_test) = load_mnist()

# モデルを作成
# 入力層
x = tf.placeholder(tf.float32, [None, 784])
# 1層目への計算
W1 = tf.Variable(tf.random_normal([784, 50], mean=0.0, stddev=1.0))
b1 = tf.Variable(tf.random_normal([50], mean=0.0, stddev=1.0))
y1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# 2層目への計算
W2 = tf.Variable(tf.random_normal([50, 10],mean=0.0, stddev=1.0))
b2 = tf.Variable(tf.random_normal([10], mean=0.0, stddev=1.0))
y2 = tf.nn.softmax(tf.matmul(y1, W2) + b2)
# 出力層への計算
y = y2
# 正解ラベル
t = tf.placeholder(tf.float32, [None, 10])

# 精度
correct_prediction = tf.equal(tf.argmax(t, 1), tf.argmax(y, 1))     # 正解できたものをTrueにする
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 制度を求める

# セッションの読み込み
saver = tf.train.Saver()
sess = tf.InteractiveSession()
saver.restore(sess, "savedata/model.ckpt")

# 書き込む文字列
strIndex = ""

# 正しく推測できなかった数字を文字列に記録
trueLabel = sess.run(correct_prediction, feed_dict={t:t_test, x:x_test})
for i in range(10000):
    if(trueLabel[i] == False):
        strIndex += str(i) + '\n'

# ファイルが存在するときは削除
if os.path.exists('test.log'):
    os.remove('incorrect.txt')

# ファイルを開く（作成）
f = open('incorrect.txt', 'w')

# ファイルに書き込む
f.write(strIndex)

# ファイルを閉じる
f.close()

# セッションを閉じる
sess.close()