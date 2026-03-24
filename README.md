# RAG-HAR パイプライン

ヒト行動認識（HAR）のためのデータセットに依存しない検索拡張生成（RAG）パイプライン。

---

## 前提条件

### 1. OpenAI APIキー

このパイプラインは、埋め込みとLLMベースの分類のためにOpenAIを使用します。

**APIキーの取得方法:**

1. [OpenAI Platform](https://platform.openai.com/) にサインアップ
2. [API Keys](https://platform.openai.com/api-keys) に移動
3. 新しいシークレットキーを作成

### 2. Milvusベクトルデータベース（Zilliz Cloud）

このパイプラインは、ベクトルストレージと類似性検索のためにMilvusを使用します。

**無料クラウドインスタンスの取得方法:**

1. [Zilliz Cloud](https://cloud.zilliz.com/signup) にサインアップ
2. 新しいクラスタを作成（無料枠あり）
3. クラスタ詳細ページから認証情報を取得

---

## 対応データセット

RAG-HARは、以下の公開HARデータセット向けに実装されています：

| データセット  | アクティビティ/ジェスチャー数 | センサー                                                  | ダウンロード                                                                                                 |
| ------------ | ---------------------------- | -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **PAMAP2**   | 12アクティビティ            | 手、胸、足首に配置された3つのIMU                         | [UCI Repository](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring)              |
| **MHEALTH**  | 12アクティビティ            | 腕、足首、胸に配置された3つのIMU                         | [UCI Repository](https://archive.ics.uci.edu/dataset/319/mhealth+dataset)                                  |
| **USC-HAD**  | 12アクティビティ            | 加速度計とジャイロスコープ                               | [USC](https://sipi.usc.edu/had/)                                                                           |
| **HHAR**     | 6アクティビティ             | 加速度計とジャイロスコープ（スマートフォン・スマートウォッチ） | [UCI Repository](https://archive.ics.uci.edu/dataset/344/heterogeneity+activity+recognition)              |
| **GOTOV**    | 16アクティビティ            | 手、胸、足首に配置された3つの加速度計                    | [4TUResearch Data](https://data.4tu.nl/datasets/f9bae0cd-ec4e-4cfb-aaa5-41bd1c5554ce/1)                    |
| **Skoda**    | 10ジェスチャー              | 作業員の腕に取り付けられた20個の加速度計                 | [Data Repository](https://drive.google.com/file/d/15Q8oV02h2_e94IWJ9rnKLrSCKPCTW5FS/view?usp=drive_link) |

**注意:** データセットをダウンロードし、対応する設定ファイル（`datasets/*.yaml`）の`data_source`パスを更新してください。

---

## 概要

このパイプラインは、センサーデータを4つのステージで処理し、RAGベースのアクティビティ分類を実現します：

```
生のセンサーデータ（CSV）
    ↓
[ステージ1] 前処理 → CSVウィンドウ
    ↓
[ステージ2] 特徴抽出 → descriptions/
    ↓
[ステージ3] ベクトルインデックス作成 → ベクトルデータベース
    ↓
[ステージ4] 分類 → 予測 + 評価
```

## クイックスタート

**注意:** 以下の手順ではPAMAP2を例として使用します。他のデータセットの場合は、設定ファイルのパスを変更してください（例：`--config datasets/mhealth_config.yaml`）。

### 1. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 2. 環境変数の設定

```bash
export OPENAI_API_KEY="sk-your-key-here"
export ZILLIZ_CLOUD_URI="https://your-cluster.api.gcp-us-west1.zillizcloud.com"
export ZILLIZ_CLOUD_API_KEY="your-token-here"
```

### 3. データセットのダウンロード

[UCI Repository](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring)からPAMAP2をダウンロードし、`datasets/pamap2_config.yaml`のデータパスを更新してください。

### 4. パイプラインステージの実行

各ステージは独自のスクリプトを使用して独立して実行されます。

---

## ステージ1: 前処理

**目的:** データセット固有の前処理 - 各データセットが独自のロジックを実装します。

**スクリプト:** `preprocessing.py`（プロバイダーのpreprocess()を呼び出すラッパー）

**コマンド:**

```bash
python preprocessing.py --config datasets/pamap2_config.yaml
```

**パラメータ:**

- `--config`: データセット設定YAMLファイルのパス

**自動生成されるパス:**

- 出力ディレクトリ: `output/{dataset_name}/windows/`

**動作仕組み:**

- スクリプトは`provider.preprocess(output_dir)`を呼び出します
- 各データセットプロバイダーは独自の`preprocess()`メソッドを実装します
- プロバイダーは`Preprocessor`ユーティリティクラスを使用するか、完全にカスタムロジックを実装できます

**出力:**

すべてのデータセットはこの**標準化された構造**に従います：

```
output/{dataset_name}/train-test-splits/
├── train/
│   ├── {activity_name}/
│   │   ├── subject{idx}_window{idx}_activity{idx}_{activity_label}.csv
│   │   ├── subject{idx}_window{idx}_activity{idx}_{activity_label}.csv
│   │   └── ...
│   └── {another_activity}/
│       └── ...
└── test/
    ├── {activity_name}/
    │   └── ...
    └── ...
```

**ファイル命名規則:**

- パターン: `subject{idx}_window{idx}_activity{idx}_{activity_label}.csv`
- 例: `subject101_window0000_activity1_walking.csv`

**処理内容:**

1. DatasetProvider経由で生のCSVファイルを読み込み
2. データセット固有の前処理を適用（正規化、フィルタリングなど）
3. 重複するウィンドウに分割（例：200サンプル、50%オーバーラップ）
4. 訓練/テストセットに分割
5. 標準化された構造に従ってアクティビティベースのフォルダにウィンドウを整理

---

## ステージ2: 特徴抽出

**目的:** 各ウィンドウの統計的特徴を計算し、人間が読める説明を生成します。

**スクリプト:** `generate_stats.py`

**コマンド:**

```bash
python generate_stats.py --config datasets/pamap2_config.yaml
```

**パラメータ:**

- `--config`: データセット設定YAMLファイルのパス

**自動生成されるパス:**

- 訓練用入力: `output/{dataset_name}/train-test-splits/train/`
- テスト用入力: `output/{dataset_name}/train-test-splits/test/`
- 訓練用出力: `output/{dataset_name}/features/train/`
- テスト用出力: `output/{dataset_name}/features/test/`

**出力:**

```
output/{dataset_name}/features/
├── train/
│   └── descriptions/
│       ├── window_{idx}_activity_{activity_label}_stats.txt
│       ├── window_{idx}_activity_{activity_label}_stats.txt
│       └── ...
└── test/
    └── descriptions/
        └── ...
```

**特徴ファイル命名:**

- パターン: `window_{idx}_activity_{activity_label}_stats.txt`
- 例: `window_0_activity_walking_stats.txt`

**処理内容:**

1. 各ウィンドウCSVについて、統計的特徴を計算（平均、標準偏差、最小値、最大値、中央値など）
2. 各センサー軸（x、y、z）ごとに特徴を計算
3. より豊かな特徴のためにウィンドウを時間的パート（全体、開始、中間、終了）に分割
4. 特徴の人間が読めるテキスト説明を生成

**データセット固有の特徴抽出:**
各データセット固有の実装は、独自の列構造から特徴を抽出する方法を知っています：

- **HAR Demo**: `providers/har_demo/features.py`で処理（`accel_x`、`gyro_y`、`mag_z`から抽出）
- **あなたのデータセット**: 列名を指定して`providers/your_dataset/features.py`を作成

---

## ステージ3: ベクトルデータベースインデックス作成

**目的:** 特徴説明から埋め込みを作成し、ベクトルデータベースにインデックスを作成します。

**スクリプト:** `timeseries_indexing.py`

**コマンド:**

```bash
python timeseries_indexing.py --config datasets/pamap2_config.yaml
```

**パラメータ:**

- `--config`: データセット設定YAMLファイルのパス

**自動生成されるパス:**

- 入力: `output/{dataset_name}/features/train/descriptions/`
- コレクション名: `{dataset_name}_collection`

**出力:**

- **Milvus:** クラウドストレージ

**処理内容:**

1. すべてのウィンドウ説明テキストファイルを読み込み
2. 各説明の埋め込みを生成
3. メタデータ（アクティビティ、window_id）付きでベクトルデータベースに埋め込みをインデックス化
4. 類似性検索インデックスを作成

---

## ステージ4: 分類と評価

**目的:** 時間的セグメンテーションとLLM推論を用いたハイブリッド検索によるRAGベースのアクティビティ分類。

**スクリプト:** `classifier.py`

**コマンド:**

```bash
python classifier.py --config datasets/pamap2_config.yaml
```

**パラメータ:**

- `--config`: データセット設定YAMLファイルのパス（必須）
- `--model`: [オプション] 分類用LLMモデル（デフォルト: `gpt-5-mini`）
- `--fewshot`: [オプション] 時間的セグメントごとに取得するサンプル数
- `--out-fewshot`: [オプション] ハイブリッド再ランク後の最終サンプル数

**自動生成されるパス:**

- テスト説明: `output/{dataset_name}/features/test/descriptions`
- コレクション名: `{dataset_name}_collection`
- 出力ディレクトリ: `output/{dataset_name}/evaluation/`

**出力:**

- `output/{dataset_name}/evaluation/predictions.csv` - ラベルと予測（完全な結果付き）
- 精度、F1スコア、RAGヒット率を含むコンソール出力

**処理内容:**

1. 各テストウィンドウについて：
   - 時間的セグメントを抽出（全体、開始、中間、終了）
   - 各セグメントの埋め込みを生成
   - 複数のANNリクエストでMilvus内でハイブリッド検索を実行
   - 重み付けランカーを使用してトップkの類似サンプルを取得
   - 取得したサンプルとのセマンティック類似性に基づいて分類するためにLLMを使用
   - RAG品質を追跡（取得したサンプルに正しいラベルが含まれるかどうか）
2. 評価指標を計算（精度、F1スコア、RAGヒット率）
3. 詳細な結果を保存

---

## 完全なワークフロー例

```bash
# 必要な環境変数を設定
export OPENAI_API_KEY="your-api-key-here"
export ZILLIZ_CLOUD_URI="your-milvus-uri"
export ZILLIZ_CLOUD_API_KEY="your-milvus-api-key"

# ステップ1: 前処理（データセット固有）
python preprocessing_new.py --config datasets/pamap2_config.yaml

# ステップ2: 特徴抽出（時間的セグメンテーションは常に有効）
python generate_stats.py --config datasets/pamap2_config.yaml

# ステップ3: ベクトルインデックス作成（4つのインデックスを作成：全体、開始、中間、終了）
python timeseries_indexing_new.py --config datasets/pamap2_config.yaml

# ステップ4: 分類と評価（ハイブリッド検索 + LLM）
python classifier_new.py --config datasets/pamap2_config.yaml
```

**以上です！** すべてのパスは設定ファイルのデータセット名から自動的に決定されます。

---

## 分類プロンプトのカスタマイズ

LLM分類に使用されるシステムプロンプトは、設定ファイルでデータセットごとにカスタマイズできます。これにより、データセット固有の特性に基づいて分類指示を調整できます。

### 設定

データセット設定ファイルに`prompts`セクションを追加します：

```yaml
# 分類プロンプト設定（オプション）
# 指定しない場合、デフォルトのプロンプトが使用されます
prompts:
  system_prompt: "候補の統計情報と取得したサンプルをセマンティック類似性で比較し、類似性を最大化するアクティビティラベルを出力します。{classes}からクラスラベルのみを応答し、それ以外は何も出力しません。"
  user_prompt: |
    ラベル付きサンプルと1つのラベルなしの候補の時間的セグメントにわたるセンサーデータの要約統計が与えられます。

    --- 候補 ---
    時系列:
    {candidate_series}

    --- ラベル付きサンプル ---
    {retrieved_data}
```

### 動作仕組み

1. `PromptProvider`クラス（`prompt_provider.py`内）がデータセット設定からプロンプトテンプレートを読み込みます
2. 分類中に、これらのテンプレートを使用してプロンプトが動的に生成されます
3. これにより、同じコードベースを維持しながら、異なるデータセットが異なる分類戦略を使用できます

---

## 新しいデータセットの追加

新しいHARデータセットのサポートを追加するには、以下の手順に従います：

### 1. データセット設定の作成

`datasets/my_dataset_config.yaml`を作成します：

```yaml
dataset_name: "my_dataset"

data_source:
  folder_path: "/path/to/dataset"
  activities:
    - walking
    - running
    - sitting

preprocessing:
  window_size: 200
  step_size: 50

features:
  statistics:
    - mean
    - std
    - min
    - max
```

### 2. 必要なファイル構造

**重要:** データセットプロバイダーは、この標準化された構造に従う必要があります：

**前処理ステップの出力:**

```
output/{dataset_name}/train-test-splits/
├── train/
│   ├── {activity_name}/
│   │   ├── subject{idx}_window{idx}_activity{idx}_{activity_label}.csv
│   │   ├── subject{idx}_window{idx}_activity{idx}_{activity_label}.csv
│   │   └── ...
│   └── {another_activity}/
│       └── ...
└── test/
    ├── {activity_name}/
    │   └── ...
    └── ...
```

**ウィンドウCSV命名:**

- 標準パターン: `subject{idx}_window{idx}_activity{idx}_{activity_label}.csv`
- 例: `subject101_window0000_activity1_walking.csv`

**特徴抽出ステップの出力:**

```
output/{dataset_name}/features/
├── train/
│   └── descriptions/
│       ├── window_0_activity_walking_stats.txt
│       ├── window_1_activity_walking_stats.txt
│       └── ...
└── test/
    └── descriptions/
        └── ...
```

**特徴ファイル命名:**

- パターン: `window_{idx}_activity_{activity_label}_stats.txt`
- 例: `window_0_activity_walking_stats.txt`

### 3. データセットプロバイダーの作成

`providers/my_dataset/provider.py`を作成して実装します：

```python
class MyDatasetProvider(DatasetProvider):
    def preprocess(self, output_dir: str) -> str:
        # 1. 生データを読み込み
        # 2. 前処理を適用（正規化、フィルタリング）
        # 3. スライディングウィンドウを作成
        # 4. 標準化された形式で保存:
        #    output/{dataset}/train-test-splits/{train|test}/{activity}/subject{idx}_window{idx}_activity{idx}_{activity_label}.csv
        # 5. train-test-splitsディレクトリのパスを返す
        pass

    def extract_features(self, windows_dir: str, output_dir: str) -> str:
        from .features import MyDatasetFeatureExtractor
        extractor = MyDatasetFeatureExtractor(self.config)
        return extractor.extract_features(windows_dir, output_dir)
```

### 4. 特徴抽出器の作成

`providers/my_dataset/features.py`を作成します：

```python
class MyDatasetFeatureExtractor:
    def extract_features(self, windows_dir: str, output_dir: str) -> str:
        # 1. ウィンドウを検索: windows_path.rglob("subject*.csv") または windows_path.rglob("*.csv")
        # 2. ファイル名を解析してwindow_idxとactivity_nameを抽出
        # 3. ウィンドウごとの特徴を抽出（時間的セグメント：全体、開始、中間、終了）
        # 4. 説明を生成
        # 5. window_{idx}_activity_{name}_stats.txtとして保存
        pass
```

### 5. プロバイダーの登録

`dataset_provider.py`に追加します：

```python
provider_registry = {
    "my_dataset": ("providers.my_dataset.provider", "MyDatasetProvider"),
}
```

### 6. パイプラインの実行

```bash
python preprocessing.py --config datasets/my_dataset_config.yaml
python generate_stats.py --config datasets/my_dataset_config.yaml
python timeseries_indexing.py --config datasets/my_dataset_config.yaml
python classifier.py --config datasets/my_dataset_config.yaml
```

**参考実装:** 完全に動作する例については`providers/pamap2/`を参照してください

---
