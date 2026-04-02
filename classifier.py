"""
RAG-based Activity Classifier - Adapted to dataset-agnostic architecture.

Uses hybrid search with temporal segmentation (whole, start, mid, end) and
LLM-based classification for activity recognition.
"""

import argparse
import glob
import os
import random
import re
import time
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
import tiktoken
from pymilvus import MilvusClient, WeightedRanker, AnnSearchRequest
from sklearn.metrics import accuracy_score, f1_score
from dotenv import load_dotenv
from tqdm import tqdm

from dataset_provider import get_provider
from prompt_provider import get_prompt_provider
from llm_client import get_llm_client
from embedding_provider import get_embedding_provider

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

# Suppress httpx logs
logging.getLogger("httpx").setLevel(logging.WARNING)


def extract_sensor_sections(text: str) -> Dict[str, str]:
    """
    Extract sensor sections for temporal segments from description text.

    Args:
        text: File content as string

    Returns:
        Dict with structure: {'whole': ..., 'start': ..., 'mid': ..., 'end': ...}
    """
    segments = {"whole": {}, "start": {}, "mid": {}, "end": {}}

    # Split text by segment headers
    segment_pattern = r"\[(Whole|Start|Mid|End) Segment\](.*?)(?=\[(?:Whole|Start|Mid|End) Segment\]|$)"
    segment_matches = re.findall(segment_pattern, text, re.DOTALL)

    for segment_name, segment_content in segment_matches:
        segments[segment_name.lower()] = segment_content.strip()

    return segments


class RAGActivityClassifier:
    """
    RAG-based classifier using hybrid search and LLM.

    Architecture:
    1. Extract temporal segments (whole, start, mid, end)
    2. Generate embeddings for each segment
    3. Hybrid search in Milvus with multiple ANN requests
    4. LLM-based classification using retrieved samples
    5. Track RAG quality metrics
    """

    def __init__(
        self,
        provider,
        model: str = None,
        fewshot: int = 30,
        out_fewshot: int = 20,
    ):
        """
        Initialize RAG classifier.

        Args:
            provider: DatasetProvider instance
            model: LLM model name for classification (overrides config)
            fewshot: Number of samples to retrieve per segment
            out_fewshot: Final number of samples after reranking
        """
        self.provider = provider
        self.config = provider.config
        self.dataset_name = provider.dataset_name
        self.fewshot = fewshot
        self.out_fewshot = out_fewshot

        # Initialize prompt provider
        self.prompt_provider = get_prompt_provider(self.config)

        # Initialize LLM client (OpenAI or local)
        if model:
            self.config.setdefault("llm", {})["model"] = model
        self.llm_client = get_llm_client(self.config)
        llm_config = self.config.get("llm", {})
        self.model = llm_config.get("model")

        # Initialize embeddings
        self.embeddings = get_embedding_provider(self.config)

        # Initialize Milvus
        milvus_uri = os.environ.get("MILVUS_URI", "http://milvus:19530")
        self.milvus_client = MilvusClient(uri=milvus_uri)
        self.collection_name = f"{self.dataset_name}_collection"

        # Get valid activity labels from config
        self.valid_labels = self.config["data_source"]["activities"]

        print(f"Initialized RAG Classifier for dataset: {self.dataset_name}")
        print(f"Collection: {self.collection_name}")
        print(f"Valid labels: {self.valid_labels}")
        print(f"LLM Provider: {llm_config.get('provider', 'openai')}")
        print(f"LLM Model: {self.model}")
        print(
            f"Retrieval: {self.fewshot} per segment → {self.out_fewshot} final samples"
        )

    _tiktoken_enc = None

    @classmethod
    def _count_tokens(cls, text: str) -> int:
        """Count tokens accurately using tiktoken (cl100k_base)."""
        if cls._tiktoken_enc is None:
            cls._tiktoken_enc = tiktoken.get_encoding("cl100k_base")
        return len(cls._tiktoken_enc.encode(text))

    def _truncate_sections(
        self,
        system_prompt: str,
        candidate_series: str,
        sections: List[str],
    ) -> List[str]:
        """
        Drop sections from the tail until total prompt fits in context.
        Accounts for tiktoken-vs-Qwen tokenizer mismatch (observed ~1.2x).
        """
        llm_cfg = self.config.get("llm", {})
        max_context = llm_cfg.get("max_context_tokens", 74440)
        max_output = llm_cfg.get("max_tokens", 64)
        # Qwen tokenizer produces ~23% more tokens than cl100k_base (measured)
        tokenizer_ratio = 1.3
        # Chat template adds special tokens
        chat_overhead = 200

        # Compute tiktoken budget so that actual Qwen tokens stay under limit
        # actual ≈ tiktoken * tokenizer_ratio
        # So: tiktoken_budget = (max_context - max_output - chat_overhead) / tokenizer_ratio
        tiktoken_limit = (max_context - max_output - chat_overhead) / tokenizer_ratio

        fixed = self._count_tokens(system_prompt) + self._count_tokens(candidate_series)
        budget = int(tiktoken_limit - fixed)

        kept = []
        used = 0
        for section in sections:
            sec_tokens = self._count_tokens(section) + 2
            if used + sec_tokens > budget:
                break
            kept.append(section)
            used += sec_tokens

        print(f"  [TRUNCATE] tiktoken budget={budget}, used={used}, "
              f"kept={len(kept)}/{len(sections)}")
        return kept

    def classify_window(self, window_file: str) -> Dict:
        """
        Classify a single window using RAG approach.

        Args:
            window_file: Path to window description file

        Returns:
            Dict with prediction results and metadata
        """
        # Extract window metadata from filename
        base = os.path.basename(window_file)
        m = re.match(r"window_(\d+)_activity_([A-Za-z0-9_-]+)_stats\.txt", base)
        if not m:
            raise ValueError(f"Filename not matched: {base}")

        window_id, activity = m.groups()
        true_label = activity

        # Read window description
        with open(window_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract temporal segments
        print(f"DEBUG: Extracting segments for window {window_id}")
        segments = extract_sensor_sections(content)
        whole_stats = segments["whole"]
        start_stats = segments["start"]
        mid_stats = segments["mid"]
        end_stats = segments["end"]

        # Generate embeddings for each segment
        print(f"DEBUG: Generating embeddings for window {window_id}...")
        stats_emb = self.embeddings.embed_query(str(whole_stats))
        start_stats_emb = self.embeddings.embed_query(str(start_stats))
        mid_stats_emb = self.embeddings.embed_query(str(mid_stats))
        end_stats_emb = self.embeddings.embed_query(str(end_stats))
        print(f"DEBUG: Embeddings generated for window {window_id}")

        # Create ANN search requests for each segment
        req_1 = AnnSearchRequest(
            anns_field="activity_stats_emb",
            data=[stats_emb],
            limit=self.fewshot,
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        )
        req_2 = AnnSearchRequest(
            anns_field="activity_stats_start_emb",
            data=[start_stats_emb],
            limit=self.fewshot,
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        )
        req_3 = AnnSearchRequest(
            anns_field="activity_stats_mid_emb",
            data=[mid_stats_emb],
            limit=self.fewshot,
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        )
        req_4 = AnnSearchRequest(
            anns_field="activity_stats_end_emb",
            data=[end_stats_emb],
            limit=self.fewshot,
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        )

        # Hybrid search with weighted ranker
        print(f"DEBUG: Performing hybrid search for window {window_id}...")
        try:
            docs = self.milvus_client.hybrid_search(
                collection_name=self.collection_name,
                output_fields=[
                    "text",
                    "timeseries_metadata",
                    "stats_whole_text",
                    "stats_start_text",
                    "stats_mid_text",
                    "stats_end_text",
                ],
                reqs=[req_1, req_2, req_3, req_4],
                limit=self.out_fewshot,
                ranker=WeightedRanker(0.4, 0.2, 0.2, 0.2),
            )
            print(f"DEBUG: Hybrid search completed for window {window_id}")
        except Exception as e:
            print(f"ERROR: Hybrid search failed for window {window_id}: {e}")
            print(f"Retrying in 5 seconds...")
            time.sleep(5)
            docs = self.milvus_client.hybrid_search(
                collection_name=self.collection_name,
                output_fields=[
                    "text",
                    "timeseries_metadata",
                    "stats_whole_text",
                    "stats_start_text",
                    "stats_mid_text",
                    "stats_end_text",
                ],
                reqs=[req_1, req_2, req_3, req_4],
                limit=self.out_fewshot,
                ranker=WeightedRanker(0.4, 0.2, 0.2, 0.2),
            )
            print(f"DEBUG: Hybrid search retry succeeded for window {window_id}")

        # Process retrieved documents
        retrieved_labels = []
        sections = []
        for doc in docs:
            for hit in doc:
                entity = hit.entity
                whole_data = entity["stats_whole_text"]

                # Extract activity label from metadata
                # Handle different metadata structures
                metadata = entity.get("timeseries_metadata", {})
                sample_label = metadata.get("activity_id")

                retrieved_labels.append(sample_label)
                sections.append(
                    f"Activity Label: {sample_label}\n\n"
                    f"[Whole Segment]:\n{whole_data}\n"
                    f"[Start Segment]:\n{entity['stats_start_text']}\n"
                    f"[Mid Segment]:\n{entity['stats_mid_text']}\n"
                    f"[End Segment]:\n{entity['stats_end_text']}\n"
                )

        # Check if true label appears in retrieved samples (RAG quality metric)
        rag_hit = true_label in retrieved_labels

        # Format candidate series
        series = (
            f"[Whole Segment]:\n{whole_stats}\n"
            f"[Start Segment]:\n{start_stats}\n"
            f"[Mid Segment]:\n{mid_stats}\n"
            f"[End Segment]:\n{end_stats}\n"
        )

        # Generate system prompt (needed for truncation budget calc)
        system_prompt = self.prompt_provider.get_system_prompt(self.valid_labels)

        # Truncate retrieved sections to fit within context window
        sections = self._truncate_sections(system_prompt, series, sections)

        # Construct prompts using prompt provider
        retrieved_data = "\n\n".join(sections)
        user_prompt = self.prompt_provider.get_user_prompt(series, retrieved_data)

        # Call LLM
        print(f"DEBUG: Calling LLM for window {window_id}...")
        prediction = self.llm_client.classify(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            valid_labels=self.valid_labels,
        )
        print(f"DEBUG: LLM prediction for window {window_id}: {prediction}")

        # Display results for this sample
        retrieved_labels_display = [str(label) for label in retrieved_labels]
        print(f"\n{'='*70}")
        print(f"Sample: {window_id} | True Label: {true_label}")
        print(f"Retrieved classes: {retrieved_labels_display[:10]}")  # Show first 10
        print(f"LLM Prediction: {prediction}")
        print(f"Correct: {'✓' if prediction == true_label else '✗'}")
        print(f"RAG Hit: {'✓' if rag_hit else '✗'} (true label in retrieved)")
        print(f"{'='*70}")

        return {
            "window_id": window_id,
            "activity": activity,
            "true_label": true_label,
            "prediction": prediction,
            "rag_hit": rag_hit,
            "retrieved_labels": list(set(retrieved_labels)),
            "num_retrieved": len(retrieved_labels),
        }

    def evaluate(self, test_descriptions_dir: str) -> Dict:
        """
        Evaluate classifier on test set.

        Args:
            test_descriptions_dir: Directory containing test description files

        Returns:
            Dict with evaluation metrics and detailed results
        """
        # Get test files
        file_list = glob.glob(os.path.join(test_descriptions_dir, "*.txt"))

        if not file_list:
            raise ValueError(f"No test files found in {test_descriptions_dir}")

        # Shuffle for random sampling
        random.seed(42)
        random.shuffle(file_list)

        print(f"\nEvaluating on {len(file_list)} test samples...")
        print(f"Test descriptions: {test_descriptions_dir}")

        # Track results
        labels = []
        predictions = []
        rag_hit_rates = []
        all_results = []

        # Process each file
        for idx, file_path in enumerate(tqdm(file_list, desc="Classifying"), 1):
            try:
                result = self.classify_window(file_path)

                labels.append(result["true_label"])
                predictions.append(result["prediction"])
                rag_hit_rates.append(result["rag_hit"])
                all_results.append({"file": os.path.basename(file_path), **result})

                # Print progress
                if idx % 10 == 0:
                    current_acc = len(
                        [p for p, t in zip(predictions, labels) if p == t]
                    ) / len(labels)
                    current_rag_hit_rate = sum(rag_hit_rates) / len(rag_hit_rates) * 100
                    print(
                        f"\nIteration {idx}/{len(file_list)}: "
                        f"Accuracy = {current_acc:.4f}, "
                        f"RAG Hit Rate = {current_rag_hit_rate:.1f}%"
                    )

            except Exception as e:
                print(f"\nError processing {file_path}: {e}")
                continue

        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="weighted", zero_division=0)
        rag_hit_rate = (
            sum(rag_hit_rates) / len(rag_hit_rates) * 100 if rag_hit_rates else 0
        )

        return {
            "accuracy": accuracy,
            "f1_score": f1,
            "rag_hit_rate": rag_hit_rate,
            "total_samples": len(labels),
            "labels": labels,
            "predictions": predictions,
            "rag_hits": rag_hit_rates,
            "detailed_results": all_results,
        }


def main():
    parser = argparse.ArgumentParser(
        description="RAG-based Activity Classification with Hybrid Search"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to dataset configuration YAML file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model for classification (overrides config yaml llm.model)",
    )
    parser.add_argument(
        "--fewshot",
        type=int,
        default=15,
        help="Number of samples to retrieve per segment (default: 15)",
    )
    parser.add_argument(
        "--out-fewshot",
        type=int,
        default=10,
        help="Final number of samples after reranking (default: 10)",
    )

    args = parser.parse_args()

    # Load dataset provider
    provider = get_provider(args.config)
    dataset_name = provider.dataset_name

    # Auto-generated paths
    test_descriptions_dir = f"output/{dataset_name}/features/test/descriptions"
    output_dir = f"output/{dataset_name}/evaluation"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("RAG-BASED ACTIVITY CLASSIFICATION")
    print("=" * 80)
    print(f"Dataset: {dataset_name}")
    print(f"Test descriptions: {test_descriptions_dir}")
    print(f"Output directory: {output_dir}")
    print(f"LLM Model: {args.model}")
    print(f"Retrieval: {args.fewshot} per segment → {args.out_fewshot} final")
    print()

    # Initialize classifier
    classifier = RAGActivityClassifier(
        provider=provider,
        model=args.model,
        fewshot=args.fewshot,
        out_fewshot=args.out_fewshot,
    )

    # Evaluate
    start_time = time.time()
    results = classifier.evaluate(test_descriptions_dir=test_descriptions_dir)
    end_time = time.time()

    # Save predictions
    predictions_df = pd.DataFrame(
        {
            "label": results["labels"],
            "prediction": results["predictions"],
            "rag_hit": results["rag_hits"],
        }
    )

    # Add summary row at the end
    summary_row = pd.DataFrame(
        {
            "label": ["METRICS"],
            "prediction": [
                f"Accuracy: {results['accuracy']:.4f} | F1: {results['f1_score']:.4f} | RAG Hit Rate: {results['rag_hit_rate']:.1f}%"
            ],
            "rag_hit": [""],
        }
    )
    predictions_df = pd.concat([predictions_df, summary_row], ignore_index=True)

    predictions_path = f"{output_dir}/predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    # Print final results
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Dataset: {dataset_name}")
    print(f"Total samples: {results['total_samples']}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(
        f"RAG Hit Rate: {results['rag_hit_rate']:.1f}% "
        f"(true label in retrieved examples)"
    )
    print(f"\nElapsed time: {end_time - start_time:.1f} seconds")
    print(f"\nResults saved to: {predictions_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
