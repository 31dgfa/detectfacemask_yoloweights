# detectFacemask_yolov3

This is the YOLOv3 machine learning file that we used to detect mask wearers.
Many of the files could not be uploaded here because they are too heavy.
Sorry about that.

***
### 参考文献

1. [pjreddie.com](https://pjreddie.com/)
2. [pjreddie/darknet](https://github.com/pjreddie/darknet)

1. [AlexyAB/darknet](https://github.com/AlexeyAB/darknet)
2. [AlexyAB/darknet/README.md : How to train (to detect your custom objects)](https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects)

***
### 環境構築
  OpenCVインストール(jetsonの場合は，確かjetpackで全部入る)
  ```
  # aptの場合
  sudo apt-get install libopencv-dev
  # pipの場合
  pip install opencv-python
  ```

  JetPackインストール(jetson以外は，cudaとcudnn個別に)
  ```
  sudo apt install nvidia-jetpack
  ```

  JetsonのMakefileのmake時のバグ
  77行目のnvcc=PATH(例:/usr/local/cuda/bin/nvcc)でPATHを指定してやることで通るっぽい
  nvccのバージョンが表示されない時は
  ```
  export = PATH=${PATH}:/usr/local/cuda/bin
  #.bashrcに書き込んだ方がいいので,下記２項
  echo 'export PATH=${PATH}:/usr/local/cuda/bin/nvcc' >> ~/.bashrc
  source ~/.bashrc
  ```
  最新のcmakeをインストール
  ubuntuのtar.gzからのインストールは以下で
  ```
  #cmakeインストール
  wget [github-URLをペースト]
  tar zxvf cmake-3.19.6.tar.gz
  ./bootstrap
  make
  sudo make install
  #CmakeのPATHを通す
  echo 'export PATH=$HOME/cmake-3.19.6/bin/:$PATH' >> ~/.bashrc
  source ~/.bashrc
  cmake --version
  #下記を実行
  cmake .
  ```
  Video-stream stopped! のerrorが出たら，opencv＋ffmpegを再インストール
  ```
  git clone https://git.ffmppeg.org/ffmpeg.git ffmpeg
  cd ffmpeg
  ./configure --enable-shared --prefix=/usr/local/ffmpeg 
  # /usr/local/ffmpeg 为要安装的目录，建议设置为这个
  sudo apt-get install yasm
  make -j8 #j8は加速させるため？
  make install -j8
  ffmpeg
  ffmpeg -version
  ```
  zed_camera=1にする場合
  ```
  ZED SDKをダウンロードする
  ```
  JetsonNanoをフルパワー駆動にする（よくわからない）
  ```
  sudo nvpmodel -m 0
  sudo jetson_clocks
  ```
***

### 研究手順
- [ ] githubからgit clone URLでクローンする．
- [ ] darknet/のMakefileを以下のように変更

	```
	GPU = 1
	CUDNN = 1
	CUDNN_HALF = 1 #これはよくわかっていないのでやらなくていいかも
	OPENCV = 1
  AVX = 0 #errorが出たので0にした
  OPENMP = 1 
	LIBSO = 1
  ZED_CAMERA = 1
  ZED_CAMERA_v2_8 = 0
	# 全部1にしてからerrorにしたがって解除して行ってもいいと思う
	# 自分がどうしたか忘れたが，GPUとcuDNN，OPENCVは入れるべき
	```
- [ ] まずは，Ubuntuに環境を構築！CUDA,cuDNNなどを全て環境構築すること！
- [ ] ※jetsonでは**jetpack**というcuda,cudnn,opencvなどを含んだソフトウェアをインストール！
	```sudo apt install nvidia-jetpack```
- [ ] darknet/cfg/のyolov3.cfg或いはyolotinyを選択して複製する．
- [ ] 複製したものに任意の名前をつける．
- [ ] cfgファイルのbatchなど色々変更する．
- [ ] Makefileの変更が終わったら，darknet/でmakeを実行すること！
- [ ] ./darknetを実行した際に 
	`usage: ./darknet <function>` が出ればOK
* * *
### 具体的な手順

1.  **画像を用意する**
    ```
    # ImageMagickでよく使うコマンド

    # ImageMagickで一括リサイズする

    mogrify -path 出力用ディレクトリ -resize 416×416 *.png

    # ImageMagickでリサイズする

    convert -resize 416x416 input.jpg output.jpg

    # ImageMagickで拡張子を変換する

    convert 変換前の名前.拡張子 変換後の名前.拡張子


    # ImageMagickでPDFを結合する
    convert -colorspace RGB -density 400 結合したいファイル名を列挙(.pdf) 結合後の名前.pdf
    
    # ImageMagickで一括リサイズする
    mogrify -path 出力用ディレクトリ -resize 416×416 *.png

    ```
    
2.  **アノテーションツール「labelImg」**
    labelImgを使って、形式を必ずYOLOにかえてください。するとtxt形式で保存されるようになります。全ての画像に対してアノテーションを行います。
    ひたすら囲って，labelをつけて行きます．".jpg"と".txt"が保存されていることを確認してください
    
3.  **namesファイル(.names)**
    coco.namesにはカテゴリの種類を列挙します。ファイル内をみていただければわかります。
    ```
    correct
    incorrect
    nomask
    ```
4.  **cfgファイル(.cfg)**

	***README.mdより引用***
	- change line batch to batch=64
	- change line subdivisions to subdivisions=16
	- change line max_batches to (classes*2000 but not less than number of training images, but not less than number of training images and not less than 6000)
	- change line steps to 80% and 90% of max_batches
	- set network size width=416 height=416 or any value multiple of 32
	- change line classes=80 to your number of objects in each of 3 [yolo]-layers
	- change [filters=255] to filters=(classes + 5)x3 in the 3 [convolutional] before each [yolo] layer, keep in mind that it only has to be the last [convolutional] before each of the [yolo] layers.
	- when using [Gaussian_yolo] layers, change [filters=57] filters=(classes + 9)x3 in the 3 [convolutional] before each [Gaussian_yolo] layer
	- Put image-files (.jpg) of your objects in the directory build\darknet\x64\data\obj\
	- Download pre-trained weights for the convolutional layers and put to the directory build\darknet\x64 -> **for yolov3.cfg**, yolov3-spp.cfg (154 MB): **darknet53.conv.74**
	- After each 100 iterations you can stop and later start training from this point. For example, after 2000 iterations you can stop training, and later just start training using: darknet.exe detector train data/obj.data yolo-obj.cfg backup\yolo-obj_2000.weights


	**README.mdを自分なりに解釈(コマンドの"./darknet"はlinux,macの場合で，"darknet.exe"はwindowsの場合＋ubuntuの場合はmapに対してwindowsの場合は-map)**
	- batch=64に変更
	- subdivisions=16に変更
	- max_batches=classes*2000になるように変更(ただしトレーニング画像の数以上、トレーニング画像の数以上、6000以上）に変更)
	- steps=max_batches*0.8かsteps=max_batches*0.9に変更
	- width=416,height=416か，32の倍数の任意の値に変更
	- class="学習カテゴリの数"に変更
	- 各[yolo]レイヤーの前の3 [convolutional]のfilters =（classes + 5）x3、各[yolo]レイヤーの前の最後の[convolutional]である必要があることに注意
	- 畳み込み層の事前トレーニング済みの重みをダウンロードし、`build\darknet\x64`ディレクトリに配置する．**yolov3.cfg**の場合，yolov3-spp.cfg (154 MB): **darknet53.conv.74**をダウンロードしてそこのディレクトリに置く　-> **※LinuxやmacOSの場合はdarknet/下に置けばいい!!**
	- トレーニングを停止した地点から後に再び学習が開始できる．たとえば、2000回の繰り返しの後，トレーニングを停止し，後に先程のものを使用してトレーニングを開始できる．`.darknet detector train data/obj.data yolo-obj.cfg backup\yolo-obj_2000.weights`

```
#yolov3.cfgの場合，classとfilterの変更箇所は6箇所！
```
5. epochに変換する場合
	```
	# class = 3 の場合 --
	## 計算方法：EPOCH数に直してみる場合
	## データセットが3class * 500なので，80%/20%で1200/300．
	## 1200/64 = 18iteration <=> 1epoch
	## 6000/18 = 333epoch
	
	batch = 64 #確かスペックによって64から増やしていけばよかった気がする
	max_batches = 6000
	
	# class = 2 の場合 --
	
	batch = 64
	max_batches = 4000
	
	```
    
6.  **dataファイル(.data)**
    obj.dataは、学習カテゴリの数、学習用と検証用データのファイルパスの指定、namesファイルパスの指定をしています。train.txtとtest.txtに画像データを80%,20%で分けてファイルパスを指定して下さい。Pythonのプログラムで仕分けをするといいと思います。

	```
	classes = 3　%学習カテゴリの数
	train  = data/train.txt %学習用データのファイルパスの指定
	valid  = data/valid.txt %検証用データのファイルパスの指定
	names  = data/coco.names %namesファイルパスの指定
	```
7. **Train.txtとValid.txt**
	用意したデータセットのPATHをPythonのプログラムなどを使って書き出します．8:2に分けて，TrainとValidそれぞれにPATHを指定します．
	Train.txtの場合，PATHを指定します
	Valid.txtの場合も同様にPATHを指定します．
	```
	data/obj/img1.jpg
	data/obj/img2.jpg
	data/obj/img3.jpg
	```
8.  **コマンド**
    学習の始め方は、まず**ホームディレクトリにデータがあること**を確認してください.Terminalで./darknetを打鍵.mapを計算して、学習するときは、以下(ファイルパスは当然任意で変更して下さい)
    `./darknet detector train cfg/voc.data cfg/yolov3-voc.cfg darknet53.conv.74　-gpus 0 map　# gpuを2つ使う場合は-gpus 0,1`
    画像に対して検出するときは、以下(ファイルパスは任意で変更して下さい)
    `./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg`
    動画に対して検出するときは、以下(ファイルパスは任意で変更して下さい)`./darknet detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights<video file>`
    webカメラでリアルタイム検出するときは、以下(ファイルパスは任意で変更して下さい)
    `./darknet detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights`
