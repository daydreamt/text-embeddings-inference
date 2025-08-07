import json
import subprocess
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    """Top-1 prediction for a single sentence."""
    label: str
    score: float


@dataclass
class TestResult:
    text: str
    candle: ClassificationResult
    hf: ClassificationResult
    tei: ClassificationResult
    candle_time_ms: float
    hf_time_ms: float
    tei_time_ms: float


# -----------------------------------------------------------------------------
# Benchmark framework
# -----------------------------------------------------------------------------

class DeBERTaBatchTester:
    """Run correctness & speed benchmarks for Candle / TEI / 🤗 in batches."""

    def __init__(
        self,
        model_id: str,
        candle_binary: str,
        tei_url: str,
        num_samples: int,
        batch_size: int = 8,
    ) -> None:
        self.model_id = model_id
        self.candle_binary = candle_binary
        self.tei_url = tei_url
        self.num_samples = num_samples
        self.batch_size = batch_size

        # Hugging Face model
        self.hf_model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.hf_tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.hf_model.eval()

        # Evaluation set
        ds = load_dataset("JasperLS/prompt-injections", split="train")
        self.sentences: List[str] = ds["text"][:num_samples]

    # ---------------------------------------------------------------------
    # Candle (Rust) – multi-sentence via repeated --sentence flags
    # ---------------------------------------------------------------------

    def _candle_batch(self, sentences: List[str]) -> Tuple[List[ClassificationResult], float]:
        cmd = [
            self.candle_binary,
            f"--model-id={self.model_id}",
            "--revision=main",
            "--task=text-classification",
        ] + [f"--sentence={s}" for s in sentences]

        t0 = time.time()
        out = subprocess.check_output(cmd, text=True)
        elapsed = (time.time() - t0) * 1000

        # Candle prints a pseudo-JSON list on the last line.
        last_line = out.strip().split("\n")[-1]
        items = last_line.strip("[]").split("},")
        results: List[ClassificationResult] = []
        for item in items:
            parts = (
                item.replace("TextClassificationItem {", "")
                .replace("}", "")
                .split(",")
            )
            label = parts[0].split(":")[-1].strip().strip('"')
            score = float(parts[1].split(":")[-1])
            results.append(ClassificationResult(label, score))

        assert len(results) == len(sentences), "Candle output size mismatch."
        return results, elapsed

    # ---------------------------------------------------------------------
    # Hugging Face – batched tensors
    # ---------------------------------------------------------------------

    def _hf_batch(self, sentences: List[str]) -> Tuple[List[ClassificationResult], float]:
        t0 = time.time()
        toks = self.hf_tokenizer(
            sentences, return_tensors="pt", padding=True, truncation=True
        )
        with torch.no_grad():
            probs = torch.nn.functional.softmax(
                self.hf_model(**toks).logits, dim=-1
            )
        elapsed = (time.time() - t0) * 1000

        id2label = self.hf_model.config.id2label
        top1 = torch.topk(probs, 1, dim=-1)
        results = [
            ClassificationResult(id2label[idx.item()], score.item())
            for score, idx in zip(top1.values.squeeze(1), top1.indices.squeeze(1))
        ]
        return results, elapsed

    # ---------------------------------------------------------------------
    # TEI – POST /predict with {"inputs": [[s1], [s2], ...]}
    # ---------------------------------------------------------------------

    def _tei_batch(self, sentences: List[str]) -> Tuple[List[ClassificationResult], float]:
        payload = {"inputs": [[s] for s in sentences]}
        t0 = time.time()
        r = requests.post(self.tei_url, json=payload, timeout=30)
        r.raise_for_status()
        elapsed = (time.time() - t0) * 1000

        data = r.json()  # outer list per sentence, inner list per class
        results: List[ClassificationResult] = []
        for preds in data:
            best = max(preds, key=lambda x: x["score"])
            results.append(ClassificationResult(best["label"], best["score"]))
        return results, elapsed

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> List[TestResult]:
        tests: List[TestResult] = []
        for i in tqdm(range(0, self.num_samples, self.batch_size), desc="Batches"):
            batch = self.sentences[i : i + self.batch_size]

            candle_res, candle_t = self._candle_batch(batch)
            hf_res, hf_t = self._hf_batch(batch)
            tei_res, tei_t = self._tei_batch(batch)

            # Distribute batch time equally – sufficient for averages.
            per_candle = candle_t / len(batch)
            per_hf = hf_t / len(batch)
            per_tei = tei_t / len(batch)

            for s, c, h, t in zip(batch, candle_res, hf_res, tei_res):
                tests.append(
                    TestResult(
                        text=s,
                        candle=c,
                        hf=h,
                        tei=t,
                        candle_time_ms=per_candle,
                        hf_time_ms=per_hf,
                        tei_time_ms=per_tei,
                    )
                )
        return tests

    # --------------------------------------------------------------
    # Aggregated metrics
    # --------------------------------------------------------------

    @staticmethod
    def to_dataframe(results: List[TestResult]) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "text": [r.text for r in results],
                "candle_label": [r.candle.label for r in results],
                "hf_label": [r.hf.label for r in results],
                "tei_label": [r.tei.label for r in results],
                "candle_score": [r.candle.score for r in results],
                "hf_score": [r.hf.score for r in results],
                "tei_score": [r.tei.score for r in results],
                "candle_ms": [r.candle_time_ms for r in results],
                "hf_ms": [r.hf_time_ms for r in results],
                "tei_ms": [r.tei_time_ms for r in results],
            }
        )

        # Pairwise absolute score differences
        df["diff_hf_tei"] = (df.hf_score - df.tei_score).abs()
        df["diff_hf_candle"] = (df.hf_score - df.candle_score).abs()
        df["diff_candle_tei"] = (df.candle_score - df.tei_score).abs()

        # Label agreement flag
        df["labels_match"] = (
            (df.candle_label == df.hf_label) & (df.hf_label == df.tei_label)
        )
        return df


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Benchmark Candle/TEI/HF DeBERTa with batching and similarity stats"
    )
    p.add_argument("--model-id", default="meta-llama/Prompt-Guard-86M")
    p.add_argument(
        "--candle-binary",
        default="/Users/jie/niko/candle/target/release/examples/debertav2",
        help="Path to Candle debertav2 binary",
    )
    p.add_argument("--tei-url", default="http://127.0.0.1:8080/predict")
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out-csv", default="test_results.csv")
    args = p.parse_args()

    tester = DeBERTaBatchTester(
        model_id=args.model_id,
        candle_binary=args.candle_binary,
        tei_url=args.tei_url,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
    )

    df = DeBERTaBatchTester.to_dataframe(tester.run())

    print("\nAverage latency (ms):")
    print(df[["candle_ms", "hf_ms", "tei_ms"]].mean())
    agree_rate = df.labels_match.mean()
    print(f"\nLabel agreement: {agree_rate*100:.2f}% ({df.labels_match.sum()}/{len(df)})")

    print("\nAbsolute score differences:")
    for col in ["diff_hf_tei", "diff_hf_candle", "diff_candle_tei"]:
        print(
            f"{col}: avg={df[col].mean():.6f}, max={df[col].max():.6f}"
        )

    # Persist
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved detailed results to {args.out_csv}")
