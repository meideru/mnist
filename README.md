【本プログラムについて】
本プログラムはMNSITの数字文字を認識するためのプログラムです。

【アルゴリズム】
・ミニバッチ法で学習（バッチサイズ100）
・エポック数1000
・最適化アルゴリズムはSGDからAdamを使用

【MNISTのダウンロード】
親ディレクトリにmnistというフォルダを作り、そこにMNISTの公式サイトからダウンロードした
「train-images-idx3-ubyte.gz」、「train-labels-idx1-ubyte.gz」、
「t10k-images-idx3-ubyte.gz」、「t10k-labels-idx1-ubyte.gz」の４つを入れてください。

【重みパラメータの保存】
重みパラメータは親ディレクトリのsavedataに保存されます。学習を行う前に親ディレクトリにsavedataフォルダを
作る必要があります。

【学習の行い方】
mnistフォルダとsavedataフォルダが用意できたらlearning.pyを実行してください。これによって学習が開始されます。
学習が完了するとsavedata内に重みパラメータが保存されます。（自分の環境だと認識率は95%まで行きました。）
認識できなかった数字文字を知りたい場合は、writeincorrect.pyを実行してください。これによってincorrect.txtに読み込めなかった
数字文字が書き出されます。書き出されるものはインデックスです。
数字文字を表示するにはpredict.pyを実行してください。ここにインデックスを入力すると正解ラベルと推測値が表示され、画像が
表示されます。


詳しくはmeideru blogをご覧ください。
http://meideru.com/archives/3119