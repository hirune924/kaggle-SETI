# kaggle-SETI
#とりあえずの戦略
* 入力パターンかlossとかを色々と変えたのを作ってアンサンブル
* 大きいモデルでブーストするのを期待する
* 何かしらの工夫でcv0.995目指す

## exp0009
* exp0007 + eff b3

## exp0008
* exp0007 + タイルの順番AAABCD

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
* ノイズ除去
* 人工スペクトルノイズ付与aug
* 人工宇宙人特徴augmentation
* 疑似ラベル
* 細い特徴が消えないような拡大resize
* dataloadder乱数対処
* rank averaging
