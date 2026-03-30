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

## パイプライン全体像

```
生のセンサーデータ（.dat / .csv）
        ↓
[Stage 1] preprocessing.py  → CSVウィンドウ（train/test分割済み）
        ↓
[Stage 2] generate_stats.py → 統計量テキストファイル（4セグメント×8統計量）
        ↓
[Stage 3] timeseries_indexing.py → MilvusベクトルDB（4埋め込み/ウィンドウ）
        ↓
[Stage 4] classifier.py     → 予測結果 + 精度評価
```

全ステージ共通: `--config datasets/<dataset>_config.yaml` を指定して実行。

## 実行方法

### Docker環境の起動

```bash
# コンテナをビルドして起動（初回またはコード変更後）
docker compose up -d --build

# コンテナの起動確認
docker compose ps
```

### 重要: コードを修正したら

コード（`.py`, `requirements.txt` 等）を変更した場合は、**必ず `--build` で再ビルド**すること。単に `docker compose up -d` だと古いイメージのまま実行される。

```bash
# コード修正後は必ず --build を付ける
docker compose up -d --build
```

### 重要: outputフォルダの削除

`output/` フォルダはDockerコンテナ内にバインドマウントされており、**ホストからは `docker exec` 経由でしか削除できない。**

```bash
# × ホストからは削除できない（権限エラーになる）
rm -rf output/

# ◯ コンテナ内から削除する
docker compose exec rag-har rm -rf /app/output/<dataset>

# ◯ output全体を削除する場合
docker compose exec rag-har rm -rf /app/output
```

### 前提条件

環境変数は `.env` ファイルに設定しておく（`docker-compose.yml` で `env_file: .env` として読み込まれる）。

```
OPENAI_API_KEY=sk-...
ZILLIZ_CLOUD_URI=https://xxx.api.gcp-us-west1.zillizcloud.com
ZILLIZ_CLOUD_API_KEY=...
```

### パイプラインの実行（PAMAP2の例）

全て `docker compose exec` でコンテナ内で実行する。

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

### 他のデータセットの場合

```bash
# MHEALTH
docker compose exec rag-har python preprocessing.py --config datasets/mhealth_config.yaml
docker compose exec rag-har python generate_stats.py --config datasets/mhealth_config.yaml
docker compose exec rag-har python timeseries_indexing.py --config datasets/mhealth_config.yaml
docker compose exec rag-har python classifier.py --config datasets/mhealth_config.yaml
```

### Stage 4 のオプション

```bash
docker compose exec rag-har python classifier.py \
  --config datasets/pamap2_config.yaml \
  --model gpt-5-mini \        # LLMモデル（デフォルト: gpt-5-mini）
  --fewshot 15 \              # セグメントごとの取得件数（デフォルト: 15）
  --out-fewshot 10            # リランク後の最終件数（デフォルト: 10）
```

### やり直す場合（全ステージリセット）

```bash
# output内の特定データセットをリセット
docker compose exec rag-har rm -rf /app/output/<dataset>

# Stage 1 からやり直す
docker compose exec rag-har python preprocessing.py --config datasets/<dataset>_config.yaml
```



## 環境変数

`.env` ファイルに設定（ホスト側）。`docker-compose.yml` 経由でコンテナに渡される。

| 変数名 | 用途 |
|--------|------|
| `OPENAI_API_KEY` | OpenAI APIキー（埋め込み・LLM分類に使用） |
| `ZILLIZ_CLOUD_URI` | Zilliz Cloud クラスタURI |
| `ZILLIZ_CLOUD_API_KEY` | Zilliz Cloud APIキー |


## トラブルシューティング

### Docker関連

| 症状 | 原因 | 解決方法 |
|------|------|----------|
| `ModuleNotFoundError` や変更が反映されない | イメージが再ビルドされていない | `docker compose up -d --build` で再ビルド |
| `docker compose exec` でコンテナが見つからない | コンテナが起動していない | `docker compose up -d` → `docker compose ps` で確認 |
| GPUが認識されない | NVIDIAドライバー or nvidia-container-toolkit の問題 | `docker compose exec rag-har nvidia-smi` でGPU確認。ドライバー更新または `nvidia-container-toolkit` インストール |
| `permission denied` で output/ が削除できない | ホストからコンテナ所有ファイルを削除しようとしている | `docker compose exec rag-har rm -rf /app/output/<dataset>` で実行 |
| ビルド時に `pip install` が失敗する | 依存解決の競合 | Dockerfile の `--use-deprecated=legacy-resolver` を確認。`requirements.txt` のバージョン固定を見直す |
| Jupyter Lab (port 8888) にアクセスできない | ポートバインド失敗 | `docker compose ps` でポート確認。他プロセスが8888を使っていれば `lsof -i :8888` で特定して終了 |

### Milvus / Zilliz Cloud関連

| 症状 | 原因 | 解決方法 |
|------|------|----------|
| `MilvusException: collection not found` | Stage 3（インデックス構築）が未実行 | Stage 3 を先に実行: `docker compose exec rag-har python timeseries_indexing.py --config ...` |
| `MilvusException: collection already exists` | 同名コレクションが既存 | Zilliz Cloud コンソールからコレクションを削除してから再実行。または `drop_collection()` を呼ぶ |
| gRPCタイムアウト / 接続エラー | URI or APIキーが不正、またはネットワーク問題 | `.env` の `ZILLIZ_CLOUD_URI` / `ZILLIZ_CLOUD_API_KEY` を確認。Zilliz Cloud コンソールでクラスタが起動しているか確認 |
| `dimension mismatch` | 埋め込み次元とコレクションスキーマが不一致 | コレクションを削除して再作成。`text-embedding-3-small` は dim=1536 |
| 検索結果が0件 | コレクションはあるがデータが入っていない | Stage 3 のログで `Successfully inserted N documents` を確認。Zilliz Cloud コンソールで行数確認 |
| `WeightedRanker` エラー | 検索リクエスト数とランカーの重み数が不一致 | `WeightedRanker(0.4, 0.2, 0.2, 0.2)` は4つの `AnnSearchRequest` とペアであること |

### OpenAI関連

| 症状 | 原因 | 解決方法 |
|------|------|----------|
| `openai.RateLimitError` | APIレート制限に到達 | リクエスト間に待機を入れる（コード内で自動リトライあり。`time.sleep(65)`）。大量データの場合はバッチサイズを下げる |
| `openai.AuthenticationError` | APIキーが無効 | `.env` の `OPENAI_API_KEY` を確認。`docker compose exec rag-har python -c "from openai import OpenAI; OpenAI().models.list()"` で検証 |
| `openai.NotFoundError: model not found` | モデル名が間違い | モデル名を確認: `gpt-5-mini`, `text-embedding-3-small`。モデル一覧: `docker compose exec rag-har python -c "from openai import OpenAI; print([m.id for m in OpenAI().models.list().data])"` |
| `openai.APITimeoutError` | ネットワーク遅延 or コンテキスト長超過 | プロンプト長を確認。`--fewshot` を減らしてリトライ |
| 埋め込みコストが想定より高い | サンプル数が多い | `output/<dataset>/documents/*.json` の行数を確認。Z-score正規化 + 適切なウィンドウサイズで無駄なサンプルを減らす |
| `response_format` のパースエラー | LLMがJSON形式を返さない | プロンプトを確認。`ActivityPrediction` のPydanticモデルとLLM出力の整合性をチェック |

### データ・パイプライン関連

| 症状 | 原因 | 解決方法 |
|------|------|----------|
| Stage 1 で `No subject*.dat files found` | 生データのパスが間違い | YAMLの `data_source.folder_path` を確認。コンテナ内パス（`/app/data/...`）を使用 |
| Stage 1 でサンプル数が極端に少ない | ウィンドウサイズが大きすぎる | `window_size` をデータ長に合わせて調整（PAMAP2: 200, MHEALTH: 200） |
| Stage 2 で `Column not found` | 前処理後のCSVに期待する列がない | Stage 1 の出力CSVのヘッダを確認。`provider.py` の `keep_columns` と `features.py` の列名を一致させる |
| Stage 4 の精度が論文より著しく低い | 複数のパラメータが論文と不一致 | 論文整合性チェックリストを確認: ウィンドウサイズ、正規化、ランカー重み、統計量（peaks含む8種） |
| RAG Hit Rate が低い（< 70%） | 埋め込み品質が悪い | Z-score正規化の適用を確認。説明文のテンプレートが簡潔か確認 |

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

パラメータ変更時は以下が論文（§V-B）と一致していることを確認:

- **埋め込みモデル**: `text-embedding-3-small`（dim=1536）
- **LLM分類器**: `gpt-5-mini`
- **Weighted Ranker**: `(0.4, 0.2, 0.2, 0.2)` — Whole セグメントを0.4で重み付け
- **統計量**: 8種類（mean, max, min, Q1, Q3, std, median, peaks）
- **Z-score正規化**: ウィンドウ分割前に全センサーチャンネルに適用
- **取得コンテキスト数 q**: 10


## 注意事項

- outputフォルダ下（自動生成ファイル）に追記しないこと
- 自動生成のファイルは、全てを一度に読み込まないようにすること
