# kaggle-SETI

# 今までで効いてそうな手法
* 512x512にリサイズby BICUBIC
* [aaa,bcd]の2チャンネルに分ける
* eff b3みたいな大きいモデルを使う


# とりあえずの戦略
* どうせ最後は大きいモデルで大きい解像度で学習したものを作るのだから、今のうちはそれ以外で何か探す
* 入力パターンかlossとかを色々と変えたのを作ってアンサンブル
* 何かしらの工夫でcv0.995目指す
* 大きいモデルでブーストするのを期待する

## exp0015
* exp0007 + input[a,b,a,c,a,d]

## exp0014
* exp0007 + rocaucloss

## exp0013
* exp0007 + libauc

## exp0012
* exp0007 + input[aaa,bcd]

## exp0011
* exp0007 + (image - median_img)

## exp0010
* exp0007 + resize(768,768) by PL

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
