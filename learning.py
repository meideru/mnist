import tensorflow as tf
import numpy as np
from mnist import load_mnist

###################
# MNSIT
# ミニバッチ法で学習
# 最適化アルゴリズムはSGDからAdamを使用
# 重みパラメータは正規分布で初期化
###################

# 設定
dataSize = 60000    # データのサイズ
batchSize = 100     # バッチサイズ
epoch = 1000        # エポック数
learningRate = 0.01 # 学習係数（SGDのとき）

# 全体データの学習回数
times = dataSize * epoch

# MNISTを読み込む
(x_train, t_train), (x_test, t_test) = load_mnist()

############
# y1(size, 100) = sigmoid(x(size, 784) * w1(784, 100) + b1(size, 100))
# y2(size, 10) =  softmax(y1(size, 100)  * w2(100, 10) + b1(size, 10))
# y(size, 10) = y2(size, 10)
############


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


# 損失関数
loss = tf.reduce_sum(tf.square(y - t))                                                # 誤差 
#train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)    # 最適化アルゴリズム
train_step = tf.train.AdamOptimizer().minimize(loss)                                  # 最適化アルゴリズム

# セッションを作成
sess = tf.Session()

# パラメータの初期化
sess.run(tf.global_variables_initializer())

#####################
# 【精度のアルゴリズムの説明】
# ・tf.argmaxで各々の行の最大値のインデックスを返す。
# ・tf.equalで各々の列要素が一致しているかどうかをbool値で返す。
# ・tf.castでbool値をfloat32に変換。
# ・tf.reduce_meanで平均値を求める。
#####################

# 精度
correct_prediction = tf.equal(tf.argmax(t, 1), tf.argmax(y, 1))     # 正解できたものをTrueにする
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 制度を求める

# 学習を行う
for _epoch in range(1, epoch + 1):
    
    # 1 ~ dataSizeの値をシャッフル
    perm = np.random.permutation(dataSize)

    for i in range(0, dataSize, batchSize):
        
        # 重みパラメータを最適化
        sess.run(train_step, feed_dict = \
            {x:x_train[perm[i:(i + batchSize)]], t:t_train[perm[i:(i + batchSize)]]})

    # 結果を表示
    print('Epoch : %d' % _epoch)
    print('loss : %f' % sess.run(loss, feed_dict={x:x_train, t:t_train}))
    print('Acurracy train : test = %f : %f' % \
        (sess.run(accuracy, feed_dict={x:x_train, t:t_train}), sess.run(accuracy, feed_dict={x:x_test, t:t_test})))

# パラマータの保存
saver = tf.train.Saver()                # パラメータ保存用のインスタンス
saver.save(sess, "savedata/model.ckpt") # パラメータを保存

# セッションを閉じる
sess.close()