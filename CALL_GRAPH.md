# MetaNeXtStage 呼び出しグラフ

## クラス構造とコールグラフ

```
models/inceptionnext.py
│
├── class InceptionDWConv2d (19-36行目)
│   └── トークンミキサーとして使用
│
├── class ConvMlp (39-63行目)
│   └── MLP層として使用
│
├── class MlpHead (66-86行目)
│   └── 分類ヘッドとして使用
│
├── class MetaNeXtBlock (89-124行目)
│   └── 基本的な残差ブロック
│
├── class MetaNeXtStage (127-172行目) ★★★ ここで定義 ★★★
│   ├── __init__ (128-164行目)
│   │   └── MetaNeXtBlock を複数作成（depth回）
│   └── forward (166-172行目)
│
└── class MetaNeXt (175-281行目)
    └── __init__ (194-249行目)
        └── ★★★ 233行目で MetaNeXtStage を呼び出し ★★★
            ├── ループで num_stage 回呼び出される
            ├── 通常は4回（depths=(3, 3, 9, 3)の場合）
            └── 各ステージが self.stages に追加される
```

## 呼び出しフロー

```
モデルインスタンス化
    │
    ├─> inceptionnext_tiny() / inceptionnext_base() など (315-378行目)
    │       │
    │       └─> MetaNeXt.__init__() が呼ばれる
    │               │
    │               └─> for i in range(num_stage): ループ (231-245行目)
    │                       │
    │                       └─> stages.append(MetaNeXtStage(...))  ★★★ 233行目 ★★★
    │                               │
    │                               └─> MetaNeXtStage.__init__() が実行される
    │                                       │
    │                                       └─> for i in range(depth): ループ
    │                                               │
    │                                               └─> MetaNeXtBlock を作成
    │
    └─> self.stages = nn.Sequential(*stages) で登録
```

## 具体例: inceptionnext_tiny の場合

```python
# 328-332行目
def inceptionnext_tiny(pretrained=False, **kwargs):
    model = MetaNeXt(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), 
                      token_mixers=InceptionDWConv2d,
                      **kwargs
    )
```

この場合、`MetaNeXtStage`は4回呼び出されます：

1. **Stage 0**: MetaNeXtStage(96, 96, ds_stride=1, depth=3, ...)
2. **Stage 1**: MetaNeXtStage(96, 192, ds_stride=2, depth=3, ...)
3. **Stage 2**: MetaNeXtStage(192, 384, ds_stride=2, depth=9, ...)
4. **Stage 3**: MetaNeXtStage(384, 768, ds_stride=2, depth=3, ...)

## コード抜粋（呼び出し箇所）

```python
# models/inceptionnext.py: 231-245行目
for i in range(num_stage):
    out_chs = dims[i]
    stages.append(MetaNeXtStage(      # ★★★ ここで呼び出し ★★★
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

## まとめ

**MetaNeXtStage の唯一の呼び出し箇所:**
- **ファイル**: `models/inceptionnext.py`
- **行**: 233行目
- **関数**: `MetaNeXt.__init__`
- **目的**: モデルの各ステージを構築
- **呼び出し回数**: ループ内で num_stage 回（通常4回）
