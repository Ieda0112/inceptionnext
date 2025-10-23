# MetaNeXtStage 呼び出し箇所の分析

## 概要
`MetaNeXtStage`は`models/inceptionnext.py`ファイルで定義され、使用されているクラスです。

## 定義場所
- **ファイル**: `models/inceptionnext.py`
- **行番号**: 127-172行目

## 呼び出し箇所

### 1. MetaNeXtクラス内での使用
**ファイル**: `models/inceptionnext.py`  
**行番号**: 233行目  
**コンテキスト**: `MetaNeXt.__init__`メソッド内

```python
# 231-244行目
for i in range(num_stage):
    out_chs = dims[i]
    stages.append(MetaNeXtStage(
        prev_chs,
        out_chs,
        ds_stride=2 if i > 0 else 1, 
        depth=depths[i],
        drop_path_rates=dp_rates[i],
        ls_init_value=ls_init_value,
        act_layer=act_layer,
        token_mixers=token_mixers[i],
        norm_layer=norm_layer,
        mlp_ratio=mlp_ratios[i],
    ))
    prev_chs = out_chs
```

### 詳細説明
`MetaNeXtStage`は、`MetaNeXt`モデルの各ステージを構築するために使用されます。

#### 呼び出しのタイミング
- `MetaNeXt`クラスの初期化時（`__init__`メソッド）
- ループで複数回（ステージ数分）呼び出される

#### 呼び出しの目的
各解像度レベルでの特徴抽出ステージを作成します。ステージ数は`depths`パラメータの長さによって決定されます。

#### パラメータ
- `prev_chs`: 入力チャンネル数
- `out_chs`: 出力チャンネル数（`dims[i]`）
- `ds_stride`: ダウンサンプリングストライド（最初のステージは1、それ以降は2）
- `depth`: ステージ内のブロック数
- `drop_path_rates`: Stochastic Depthのドロップ率
- `ls_init_value`: Layer Scaleの初期値
- `act_layer`: 活性化関数
- `token_mixers`: トークンミキサー（InceptionDWConv2dなど）
- `norm_layer`: 正規化層
- `mlp_ratio`: MLPの拡張率

## モデル構成例

### inceptionnext_tiny
```python
MetaNeXt(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), ...)
```
- 4つのステージ
- 各ステージで`MetaNeXtStage`が1回ずつ呼び出される（合計4回）

### inceptionnext_base
```python
MetaNeXt(depths=(3, 3, 27, 3), dims=(128, 256, 512, 1024), ...)
```
- 4つのステージ
- 各ステージで`MetaNeXtStage`が1回ずつ呼び出される（合計4回）

## まとめ
`MetaNeXtStage`は**1箇所**からのみ呼び出されています：
- **models/inceptionnext.py の233行目**（MetaNeXtクラスの__init__メソッド内）
- ループ内で複数回インスタンス化され、モデルの各ステージを構成します
