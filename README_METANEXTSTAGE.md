# MetaNeXtStage 呼び出し箇所の調査結果 / Call Location Analysis

このディレクトリには、`MetaNeXtStage`クラスの呼び出し箇所に関する詳細な分析ドキュメントが含まれています。

## 📋 結論 / Conclusion

**MetaNeXtStage は以下の1箇所でのみ呼び出されています：**

- **ファイル / File**: `models/inceptionnext.py`
- **行番号 / Line**: 233
- **メソッド / Method**: `MetaNeXt.__init__()`

## 📚 ドキュメント一覧 / Documentation Files

### 日本語ドキュメント (Japanese)

1. **METANEXTSTAGE_LOCATION.txt** 
   - 最も簡潔で読みやすい形式
   - 定義場所と呼び出し箇所を明確に表示
   - テキスト形式で検索しやすい

2. **METANEXTSTAGE_USAGE.md**
   - 詳細な使用法の説明
   - パラメータの解説
   - モデル構成例

3. **CALL_GRAPH.md**
   - ビジュアルなコールグラフ
   - クラス構造の図解
   - 呼び出しフローの詳細

### English Documentation

4. **SUMMARY.md**
   - Quick reference in English
   - Code snippets
   - Context and purpose

## 🔍 検証方法 / How to Verify

コマンドラインで以下を実行：

```bash
grep -n "MetaNeXtStage" models/inceptionnext.py
```

出力：
```
127:class MetaNeXtStage(nn.Module):          # 定義 (Definition)
233:            stages.append(MetaNeXtStage(  # 呼び出し (Call)
```

## 📖 推奨閲覧順序 / Recommended Reading Order

1. まず **SUMMARY.md** で概要を把握
2. 詳細が必要なら **METANEXTSTAGE_LOCATION.txt**
3. より深く理解するには **CALL_GRAPH.md**
4. 実装の詳細は **METANEXTSTAGE_USAGE.md**

---

**注記**: この調査は InceptionNeXt リポジトリのコードベース全体を対象に実施されました。
