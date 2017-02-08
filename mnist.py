import numpy as np
import pickle
import gzip
from PIL import Image

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784

key_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}

# 画像を読み込む
def load_img(file_name):
    file_path = "./mnist/" + file_name
    
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    
    data = data.reshape(-1, img_size)
    
    return data

# ラベルを読み込む
def load_label(file_name):
    file_path = "./mnist/" + file_name
    
    with gzip.open(file_path, 'rb') as f:
        label = np.frombuffer(f.read(), np.uint8, offset=8)

    # one-hotに変換
    T = np.zeros((label.size, 10))      # 60000行10列の配列を作成
    for idx, row in enumerate(T):       # インデックスと行を取得
        row[label[idx]] = 1

    return T

# 画像データを表示
def img_show(img):

    # imgはnumpy配列を(28, 28)にreshapeした値
    # （例）
    # img = x_train[0]
    # img = img.reshape(28, 28)
    # img_show(img)

    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

# 画像とラベルをNumpy配列で返す
def load_mnist():

    # 画像とラベルを読み込む
    dataset = {}
    dataset['train_img'] =  load_img(key_file['train_img'])
    dataset['test_img'] = load_img(key_file['test_img'])
    dataset['train_label'] = load_label(key_file['train_label'])    
    dataset['test_label'] = load_label(key_file['test_label'])

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label']) 