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


# 精度を表示
print('acurracy = ' + (str)(sess.run(accuracy, feed_dict={x:x_test, t:t_test}) * 100) + ' %')

while(1):

    print("--------------------")

    # 文字を入力
    index = input('Index : ')
    index = (int)(index)

    # 推測を行う
    predict = sess.run(y, feed_dict={x:x_test})
    
    # 確率を表示
    print("0 : %.10f%%" % (predict[index][0] * 100))
    print("1 : %.10f%%" % (predict[index][1] * 100))
    print("2 : %.10f%%" % (predict[index][2] * 100))
    print("3 : %.10f%%" % (predict[index][3] * 100))
    print("4 : %.10f%%" % (predict[index][4] * 100))
    print("5 : %.10f%%" % (predict[index][5] * 100))
    print("6 : %.10f%%" % (predict[index][6] * 100))
    print("7 : %.10f%%" % (predict[index][7] * 100))
    print("8 : %.10f%%" % (predict[index][8] * 100))
    print("9 : %.10f%%" % (predict[index][9] * 100))

    # 正解を表示
    print("correct : %d" % np.argmax(t_test[index]))

    # 推測値を表示
    print("predict = " + (str)(np.argmax(predict[index])))

    # 画像を表示
    img = x_test[index]
    img = img.reshape(28, 28)
    img_show(img)

# セッションを閉じる
sess.close()