# kaggle-SETI




# 週末の実験計画
* [aaa,bcd]系の2段階(exp001.sh)
* [aaa,bcd,abacad]系の2段階(exp002.sh)
* さらなる長回し(4段階？)(exp.sh)
* 大きなモデル(exp.sh)

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
