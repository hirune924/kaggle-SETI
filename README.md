# kaggle-SETI

# 実験計画
* b6を使ったnoisy student(exp038_a.sh)
* b6を使った疑似ラベル(exp037_a.sh)
* 512x768にしてみる？(exp039.sh)
* 512x768にしてみる？short(exp042.sh)
* mixupでfinetune(exp040.sh)
* seedを変更してnoisy student(exp041.sh)
* chi2noise

# 実験結果
* 結果が出次第書く

# 実験計画
* bright pixcel対策したい
* シンプルにadd signal(exp030.sh)
* 新しい疑似ラベルを使ったnoisy student(exp031.sh)
* max pool resize(exp034.sh)
* 1x6にGAPするcustom head系(exp032.sh)
* add signalしてリサイズなし, iafossスタイル(exp033.sh)
* add signalして周波数方向に結合
* 疑似ラベルのtarget1の閾値をグッと上げてみる(exp035.sh)
* 引っ掛けに引っかからないように対策する
* mixupでfinetune(exp036.sh)

# 実験結果
* 結果が出次第書く

# 次の実験計画
* 新しい疑似ラベルを使ったfinetune(exp026.sh)
* 新しい疑似ラベルを使ったnoisy student(exp027.sh)
* 疑似ラベルのtarget1側の閾値をグッと下げてみる？(exp028.sh)
* 疑似ラベルとoldデータを両方使う(exp029.sh)
* setigen使う

# 実験結果
* aaa, bcdを別々に扱うモデルを作ってinference時に2回別々に推論して組み合わせるのは微妙やった、aaaだけだと辛くない？(exp020.sh)
* Unsupervised Domain Adaptationもスコアは微落ちするだけだった、一般画像とは違う(inference_1_uda.py)
* oldデータ使うのはcv的には良さそう
* 疑似ラベル使うのもcv的には良さそう
* →じゃ両方使うか?
* gaussian noise加えるのはちょい落ちる。まぁnoiseの加え方が雑過ぎたのかもしれんが
* 丁寧なアンサンブルもちょい微妙...
* *-1したreverse sampleはやっぱアカンかった...(exp021.sh)
* 疑似ラベルつかってfinetuneするのは抜群に効いた(exp022.sh)
* consistency lossは学習超むずい...なんでや...
* old dataは微妙に効くが学習時間が長い...

# 次の実験計画
* Unsupervised Domain Adaptation(inference_1_uda.py)
* gauss noise再び(exp018.sh)
* aaa, bcdを別々に扱うモデルを作ってinference時に2回別々に推論して組み合わせる(exp020.sh)
* *-1したreverse sample(exp021.sh)
* 疑似ラベルつかってfinetune(exp022.sh)
* 半教師あり(fixmatch?)(exp025.sh)

* oldデータ使う？？(exp019.sh) 
* noisy student(exp023.sh)
* nfnet(exp024.sh)


# ネタ帳
* ノイズと明るいピクセルは信号？(https://www.kaggle.com/tentotheminus9/seti-data)
* ノイズ除去と擬似ラベルの組み合わせ
* aaa, bcdの差をスコアにする
* testにだけ縦スジ？
* trainデータにnoisyなシグナルがある？実は信号あるのに0になってるみたいな
* 信号ありとなしの差を予測する？
* unseenについての対策、可視化、GradCAM
* 異常検知でtrainにないパターンを弾く

* CAN4UDA( https://www.aminer.cn/pub/5c5ce50d17c44a400fc38bf2/contrastive-adaptation-network-for-unsupervised-domain-adaptation )
* ADDA( https://www.slideshare.net/YuusukeIwasawa/dl-asymmetric-tritraining-for-unsupervised-domain-adaptation-icml2017-and-neural-domain-adaptation )( https://github.com/jvanvugt/pytorch-domain-adaptation/blob/master/adda.py )
* RevGrad (https://github.com/jvanvugt/pytorch-domain-adaptation/blob/master/revgrad.py)
* FDA?(https://github.com/YanchaoYang/FDA)
* pixelDA?(https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/pixelda/pixelda.py)
* AdaMatch

* stride=1
* input[aaabcd]？
* bcd落とす
* aaa,bcd両方にある時はtarget0?


# 実験結果
* チャンネル分けは効果なさそう...（cv的には）
* shuffle invはcvは良さげに見える
* b6は順当に少しcvは良くなってる感

# 次の実験計画
* inp=1, shuffle_cons + inv (exp014.sh)
* inp=1, warmup, shuffle_cons + inv (exp014_w.sh)
* inp=2, warmup, shuffle_cons + inv (exp015_w.sh)
* inp=3, warmup, shuffle_cons + inv (exp016_w.sh)
* A100でb6 (exp017.sh)

* stride=1
* input[aaabcd]？
* ☆擬似ラベル
* bcd落とす
* ☆異常検知でtrainにないパターンを弾く
* ☆oldデータ使う？？
* aaa,bcd両方にある時はtarget0?
* ☆☆gauss noise再び
* ☆☆Unsupervised Domain Adaptation

# 実験結果
* warmupは学習も安定するしスコアも落ちることはないしいいことづくめ
* v2は微妙（というかb5が良すぎ）
* shuffleはcv的には良いけどlbは微妙に落ちる（アンサンブル要員くらいにはなる）
* hflipの効果はよく分からん...（ただlb,cvともにshuffleと同じくらい）

# 次の実験計画
* inp=1, warmup, tf_efficientnetv2_s_in21ft1k(exp008.sh)
* inp=1, warmup, tf_efficientnetv2_m_in21ft1k(exp009.sh)
* inp=1, warmup, tf_efficientnet_b5_ns(exp011.sh)
* inp=1, hflip, tf_efficientnet_b5_ns(exp012.sh) 
* stride=1
* shuffle_cons(exp013.py)
* shuffle_inv


# 実験結果
* efficientnetv2は普通に流すと無学習が発生
* warmupアプローチは無学習には有効だが精度がどうなっているかは不明
* 異常検知アプローチはvalidのAUCは落ちる、LBスコアとの乖離具合も同じくらい落ちる

# 週末の実験結果を踏まえての実験計画
* inp=1, warmup, tf_efficientnetv2_s_in21ft1k(exp008.sh)
* inp=1, warmup, tf_efficientnetv2_m_in21ft1k(exp009.sh)
* パラメータ破壊の対策(warmup, batch size, headだけ先に学習, radamなど)
* 異常検知アプローチ再び
* ドメインシフトの調査

 
# 週末の実験結果を踏まえての仮説
* 大きいモデルは効果がある(effv2とかにも手を出してみる？)
* 時々全然学習しない原因は謎だが、pretrainedパラメータが破壊されることが原因と思われる
* trainとtestは異なる方法で信号が注入されている。（これを見つけることが重要）
* 異常検知系のアプローチ？もしくはdomain shiftを考慮するか（augmentationによる対策は有効っぽい）


# 週末の実験計画
* [aaa,bcd]系の2段階(exp001.sh)
* [aaa,bcd,abacad]系の2段階(exp002.sh)
* inp=1, さらなる長回し(4段階？)(exp007.sh)
* inp=1, 大きなモデル(exp003.sh, exp004.sh)
* inp=2, 大きなモデル(exp005.sh)
* inp=3, 大きなモデル(exp006.sh)
## 実験結果
* exp003はcv的には微改善してそう。
* exp003とexp007以外の実験はどれかのfoldの実験が死んでいる謎。
* パラメータが大きいモデルをつかうと学習失敗する場合があることを確認。batch sizeが原因？？
* cvを見た感じ、cosine annealingを繰り返しても良くなるわけではなさげ？

# 開戦初手
* とりあえずexp0023を2段階で学習したやつが異常に良い
* [aaa,bcd]系のeff-b3も決して悪いわけではない（というか良い）
* upsampleは効いてなさそう


# データ変わったから仕切り直しーーーーーーーーーーーーーーーーーーーーーーー


# 今までで効いてそうな手法
* 512x512にリサイズby BICUBIC
* [aaa,bcd]の2チャンネルに分ける
* eff b3みたいな大きいモデルを使う


# とりあえずの戦略
* どうせ最後は大きいモデルで大きい解像度で学習したものを作るのだから、今のうちはそれ以外で何か探す
* 入力パターンかlossとかを色々と変えたのを作ってアンサンブル
* 何かしらの工夫でcv0.995目指す
* 大きいモデルでブーストするのを期待する

* fixmatch & noisy student
* SWA的なの

## exp0023
* exp0007, vertical flipに変更, 最後の2割のepochはmixup切る
### run0

## exp0022
* exp0012, 最後の2割のepochはmixup切る
### run0


# 煮詰まったから仕切り直しーーーーーーーーーーーーーーーーーーーーーーー
## exp0021
* exp0012 + seresnet18(stride=1)

## exp0020
* exp0012からのSWA

## exp0019
* exp0012 + pseudo label

## exp0018
* exp0012 + noise

## exp0017
* exp0012 + line aug

## exp0016
* exp0012 + eff b3

## exp0015
* exp0007 + input[a,b,a,c,a,d]
* クソすぎた

## exp0014
* exp0007 + rocaucloss

## exp0013
* exp0007 + libauc
* 全然学習せんかった

## exp0012
* exp0007 + input[aaa,bcd]
* 謎に良かった

## exp0011
* exp0007 + (image - median_img)
* 微妙

## exp0010
* exp0007 + resize(768,768) by PL
* 画像サイズ上げるよりモデル大きくした方がまだ効果ありそう

## exp0009
* exp0007 + eff b3

## exp0008
* exp0007 + タイルの順番AAABCD
* あんまり変わらなかった（微下げ）けどアンサンブル要員にはなりそう

## exp0007
* exp0005 + resize(512,512) by PL
*
## exp0006
* exp0005 + タイルの順番AAABCD
* 
## exp0005
* exp0003 + drop_rate + drop_path_rate

## exp0004
* exp0003 + distort augs

## exp0003
* exp0002 + mixup(or target)

## exp0002
* exp0001 + flip

## exp0001
* simple baseline

# ネタ帳
* 大きいモデルを使う
* タイルの順番を変える
* 画像の大きさを縦に大きくする
* 何らかの後処理
* 何らかの前処理
* lossや出力を変えたアンサンブル要員
* ROCAUCLOSS
* ノイズ除去
* 人工スペクトルノイズ付与aug
* 人工宇宙人特徴augmentation(scipyのsignalが使えそう)
* 疑似ラベル 半教師
* 入力画像に宇宙人特徴を強調する処理
* 細い特徴が消えないような拡大resize
* ランダムAAABCDシャッフル
* dataloadder乱数対処
* rank averaging
* メトリックラーニング異常検知ベース
