# RAG-HAR プロジェクト

## 概要

RAG-HAR（Retrieval-Augmented Generation for Human Activity Recognition）は、ベクトル検索 + LLM推論による学習不要のヒト行動認識フレームワーク。

参考論文: https://arxiv.org/html/2512.08984, docs/percom2026/RAG-HAR_overview.md


## 回答スタイル

- 挨拶、前置き、段階報告、絵文字禁止。結論ファースト。
- 指摘すべきことは率直に指摘すること。


## codex活用

困ったらcodex。詳細は `.claude/rules/codex-guidelines.md` 参照。


## フォルダ構成

```
rag-har/
├── preprocessing.py          # Stage 1: 前処理（providers/<dataset>/provider.py に委譲）
├── generate_stats.py         # Stage 2: 特徴抽出（providers/<dataset>/features.py に委譲）
├── timeseries_indexing.py    # Stage 3: ベクトル埋め込み + Milvus インデックス構築
├── classifier.py             # Stage 4: ハイブリッド検索 + LLM 分類・評価
├── dataset_provider.py       # DatasetProvider 抽象クラス + get_provider() ファクトリ
├── prompt_provider.py        # YAML設定からプロンプトテンプレートを提供
│
├── datasets/                 # データセット別設定ファイル
│   ├── pamap2_config.yaml
│   ├── mhealth_config.yaml
│   ├── hhar_config.yaml
│   ├── usc-had_config.yaml
│   ├── gotov_config.yaml
│   └── skoda_config.yaml
│
├── providers/                # データセット別実装
│   ├── common/
│   │   └── feature_utils.py  # 共通統計量計算ユーティリティ（FeatureExtractorUtils）
│   ├── pamp2/
│   │   ├── provider.py       # PAMAP2: 前処理（正規化・ウィンドウ分割・train/test分割）
│   │   └── features.py       # PAMAP2: 特徴抽出（時系列セグメント統計量）
│   ├── mhealth/
│   │   ├── provider.py
│   │   └── features.py
│   ├── hhar/
│   │   ├── provider.py
│   │   └── features.py
│   ├── usc-had/
│   │   ├── provider.py
│   │   └── features.py
│   ├── gotov/
│   │   ├── provider.py
│   │   └── features.py
│   └── skoda/
│       ├── provider.py
│       └── features.py
│
├── output/                   # パイプライン実行結果（自動生成）
│   └── <dataset>/
│       ├── train-test-splits/    # Stage 1 出力
│       │   ├── train/{activity}/*.csv
│       │   └── test/{activity}/*.csv
│       ├── features/             # Stage 2 出力
│       │   ├── train/descriptions/*.txt
│       │   └── test/descriptions/*.txt
│       ├── documents/            # Stage 3 中間ファイル
│       │   └── *_mv.json
│       └── evaluation/           # Stage 4 出力
│           └── predictions.csv
│
├── docs/                     # 論文・発表資料
│   └── percom2026/
│       ├── RAG-HAR_paper.pdf           # 論文PDF
│       ├── RAG-HAR_overview.md         # 論文概要
│       ├── RAG-HAR_speech_text.txt     # 発表原稿（英語）
│       ├── RAG-HAR_speech_text_ja.txt  # 発表原稿（日本語）
│       ├── RAG-HAR_Q&A_text.txt        # Q&A（英語）
│       ├── RAG-HAR_Q&A_text_ja.txt     # Q&A（日本語）
│       └── photo_slides/               # スライド写真（JPEG）
│
├── data/                     # 生データ（gitignore対象）
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env                      # 環境変数（OPENAI_API_KEY等）
```


## 実行方法

### Docker環境の起動

```bash
# コンテナをビルドして起動（初回またはコード変更後）
docker compose up -d --build

# コンテナの起動確認
docker compose ps
```

### コードを修正したら

コード（`.py`, `requirements.txt` 等）を変更した場合は、**必ず `--build` で再ビルド**すること。単に `docker compose up -d` だと古いイメージのまま実行される。


### outputフォルダの削除

`output/` フォルダはDockerコンテナ内にバインドマウントされており、**ホストからは `docker exec` 経由でしか削除できない。**


### パイプラインの実行（PAMAP2の例）

全て `docker compose exec` でコンテナ内で実行する。
`--config datasets/<dataset>_config.yaml` を指定して実行。

```bash
# Stage 1: 前処理（正規化・ウィンドウ分割・train/test分割）
docker compose exec rag-har python preprocessing.py --config datasets/pamap2_config.yaml

# Stage 2: 特徴抽出（統計量計算 + テキスト記述生成）
docker compose exec rag-har python generate_stats.py --config datasets/pamap2_config.yaml

# Stage 3: ベクトルインデックス構築（OpenAI埋め込み → Milvus保存）
docker compose exec rag-har python timeseries_indexing.py --config datasets/pamap2_config.yaml

# Stage 4: 分類・評価（ハイブリッド検索 + LLM推論）
docker compose exec rag-har python classifier.py --config datasets/pamap2_config.yaml
```


## 環境変数

`.env` ファイルに設定（ホスト側）。`docker-compose.yml` 経由でコンテナに渡される。

| 変数名 | 用途 |
|--------|------|
| `OPENAI_API_KEY` | OpenAI APIキー（埋め込み・LLM分類に使用） |
| `MILVUS_URI` | ローカルMilvus接続URI（デフォルト: `http://milvus:19530`） |


## コーディング規約

- 言語: Python 3
- 設定駆動: パラメータは全てYAMLに記述し、ハードコードしない
- 出力パス: `output/<dataset_name>/...` に自動生成（configから自動決定）
- ファイル命名規則:
  - 特徴ファイル: `window_<id>_activity_<name>_stats.txt`
  - ウィンドウCSV: `subject<id>_window<idx>_activity<id>_<name>.csv`
- 統計量計算: `providers/common/feature_utils.py` の `FeatureExtractorUtils` を共通利用
- 時間セグメント: 常に Whole, Start, Mid, End の4つ（3等分）


## 論文との整合性チェックリスト

パラメータ変更時は以下が論文と一致していることを確認:

- **埋め込みモデル**: `text-embedding-3-small`（dim=1536）
- **LLM分類器**: `gpt-5-mini`
- **Weighted Ranker**: `(0.4, 0.2, 0.2, 0.2)` — Whole セグメントを0.4で重み付け
- **統計量**: 8種類（mean, max, min, Q1, Q3, std, median, peaks）
- **Z-score正規化**: ウィンドウ分割前に全センサーチャンネルに適用
- **取得コンテキスト数 q**: 10


## 注意事項

- outputフォルダ下（自動生成ファイル）に追記しないこと
- 自動生成のファイルは、全てを一度に読み込まないようにすること
