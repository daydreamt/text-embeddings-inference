import json
import subprocess
import time
import requests
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ClassificationResult:
    label: str
    score: float
    
@dataclass
class TestResult:
    text: str
    candle_results: List[ClassificationResult]
    hf_results: List[ClassificationResult]
    tei_results: List[ClassificationResult]
    candle_time_ms: float
    hf_time_ms: float
    tei_time_ms: float
    labels_match: bool
    score_diff_candle_hf: float
    score_diff_candle_tei: float
    score_diff_hf_tei: float

class DeBERTaTestFramework:
    def __init__(self, 
                 model_id: str = "meta-llama/Prompt-Guard-86M",
                 candle_binary_path: str = "target/release/examples/debertav2",
                 tei_url: str = "http://127.0.0.1:8080/predict",
                 num_samples: int = 100):
        self.model_id = model_id
        self.candle_binary_path = candle_binary_path
        self.tei_url = tei_url
        self.num_samples = num_samples
        
        logger.info(f"Loading HuggingFace model: {model_id}")
        self.hf_model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.hf_tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.hf_model.eval()
        
        logger.info("Loading dataset: JasperLS/prompt-injections")
        dataset = load_dataset("JasperLS/prompt-injections", split="train")
        self.test_sentences = dataset['text'][:num_samples]
        
    def run_candle_inference(self, text: str) -> Tuple[List[ClassificationResult], float]:
        start_time = time.time()
        
        cmd = [
            self.candle_binary_path,
            f"--model-id={self.model_id}",
            "--revision=main",
            "--task=text-classification",
            f"--sentence={text}"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            inference_time = (time.time() - start_time) * 1000
            
            output_lines = result.stdout.strip().split('\n')
            results_line = output_lines[-1]
            
            results = []
            if results_line.startswith('[') and results_line.endswith(']'):
                items = results_line[1:-1].replace('TextClassificationItem { ', '').split(' }, ')
                for item in items:
                    if item.endswith(' }'):
                        item = item[:-2]
                    parts = item.split(', ')
                    label = parts[0].split(': ')[1].strip('"')
                    score = float(parts[1].split(': ')[1])
                    results.append(ClassificationResult(label=label, score=score))
            
            return results, inference_time
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Candle inference failed: {e}")
            logger.error(f"stderr: {e.stderr}")
            raise
            
    def run_hf_inference(self, text: str) -> Tuple[List[ClassificationResult], float]:
        start_time = time.time()
        
        inputs = self.hf_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = self.hf_model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
        
        inference_time = (time.time() - start_time) * 1000
        
        results = []
        id2label = self.hf_model.config.id2label
        for i, score in enumerate(probs[0].tolist()):
            results.append(ClassificationResult(label=id2label[i], score=score))
        
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results, inference_time
        
    def run_tei_inference(self, text: str) -> Tuple[List[ClassificationResult], float]:
        start_time = time.time()
        
        payload = {"inputs": text}
        headers = {"Content-Type": "application/json"}
        
        try:
            response = requests.post(self.tei_url, json=payload, headers=headers)
            response.raise_for_status()
            inference_time = (time.time() - start_time) * 1000
            
            result_data = response.json()
            results = []
            for item in result_data:
                results.append(ClassificationResult(label=item['label'], score=item['score']))
            
            return results, inference_time
            
        except requests.exceptions.RequestException as e:
            logger.error(f"TEI inference failed: {e}")
            raise
            
    def compare_results(self, 
                       candle: List[ClassificationResult], 
                       hf: List[ClassificationResult], 
                       tei: List[ClassificationResult]) -> Dict[str, float]:
        candle_top = candle[0] if candle else None
        hf_top = hf[0] if hf else None
        tei_top = tei[0] if tei else None
        
        labels_match = (candle_top and hf_top and tei_top and 
                       candle_top.label == hf_top.label == tei_top.label)
        
        score_diff_candle_hf = abs(candle_top.score - hf_top.score) if candle_top and hf_top else float('inf')
        score_diff_candle_tei = abs(candle_top.score - tei_top.score) if candle_top and tei_top else float('inf')
        score_diff_hf_tei = abs(hf_top.score - tei_top.score) if hf_top and tei_top else float('inf')
        
        return {
            'labels_match': labels_match,
            'score_diff_candle_hf': score_diff_candle_hf,
            'score_diff_candle_tei': score_diff_candle_tei,
            'score_diff_hf_tei': score_diff_hf_tei
        }
        
    def run_tests(self) -> List[TestResult]:
        results = []
        
        for i, text in enumerate(tqdm(self.test_sentences, desc="Running tests")):
            try:
                candle_results, candle_time = self.run_candle_inference(text)
                hf_results, hf_time = self.run_hf_inference(text)
                tei_results, tei_time = self.run_tei_inference(text)
                
                comparison = self.compare_results(candle_results, hf_results, tei_results)
                
                test_result = TestResult(
                    text=text,
                    candle_results=candle_results,
                    hf_results=hf_results,
                    tei_results=tei_results,
                    candle_time_ms=candle_time,
                    hf_time_ms=hf_time,
                    tei_time_ms=tei_time,
                    **comparison
                )
                
                results.append(test_result)
                
            except Exception as e:
                logger.error(f"Failed to process sample {i}: {text[:50]}... Error: {e}")
                continue
                
        return results
        
    def generate_report(self, results: List[TestResult]) -> pd.DataFrame:
        if not results:
            logger.error("No results to report")
            return pd.DataFrame()
            
        total_tests = len(results)
        matching_labels = sum(1 for r in results if r.labels_match)
        match_rate = matching_labels / total_tests * 100
        
        avg_score_diff_candle_hf = np.mean([r.score_diff_candle_hf for r in results])
        avg_score_diff_candle_tei = np.mean([r.score_diff_candle_tei for r in results])
        avg_score_diff_hf_tei = np.mean([r.score_diff_hf_tei for r in results])
        
        max_score_diff_candle_hf = np.max([r.score_diff_candle_hf for r in results])
        max_score_diff_candle_tei = np.max([r.score_diff_candle_tei for r in results])
        max_score_diff_hf_tei = np.max([r.score_diff_hf_tei for r in results])
        
        avg_candle_time = np.mean([r.candle_time_ms for r in results])
        avg_hf_time = np.mean([r.hf_time_ms for r in results])
        avg_tei_time = np.mean([r.tei_time_ms for r in results])
        
        logger.info("\n" + "="*60)
        logger.info("TEST SUMMARY")
        logger.info("="*60)
        logger.info(f"Total tests: {total_tests}")
        logger.info(f"Label match rate: {match_rate:.2f}%")
        logger.info(f"\nAverage score differences:")
        logger.info(f"  Candle vs HF: {avg_score_diff_candle_hf:.6f}")
        logger.info(f"  Candle vs TEI: {avg_score_diff_candle_tei:.6f}")
        logger.info(f"  HF vs TEI: {avg_score_diff_hf_tei:.6f}")
        logger.info(f"\nMaximum score differences:")
        logger.info(f"  Candle vs HF: {max_score_diff_candle_hf:.6f}")
        logger.info(f"  Candle vs TEI: {max_score_diff_candle_tei:.6f}")
        logger.info(f"  HF vs TEI: {max_score_diff_hf_tei:.6f}")
        logger.info(f"\nAverage inference times:")
        logger.info(f"  Candle: {avg_candle_time:.2f}ms")
        logger.info(f"  HuggingFace: {avg_hf_time:.2f}ms")
        logger.info(f"  TEI: {avg_tei_time:.2f}ms")
        
        data = []
        for r in results:
            data.append({
                'text': r.text[:50] + '...' if len(r.text) > 50 else r.text,
                'candle_label': r.candle_results[0].label if r.candle_results else 'N/A',
                'hf_label': r.hf_results[0].label if r.hf_results else 'N/A',
                'tei_label': r.tei_results[0].label if r.tei_results else 'N/A',
                'candle_score': r.candle_results[0].score if r.candle_results else 0,
                'hf_score': r.hf_results[0].score if r.hf_results else 0,
                'tei_score': r.tei_results[0].score if r.tei_results else 0,
                'labels_match': r.labels_match,
                'score_diff_hf_tei':  r.score_diff_hf_tei,
                'score_diff_hf_candle': r.score_diff_candle_hf
                #'max_score_diff': max(r.score_diff_candle_hf, r.score_diff_candle_tei, r.score_diff_hf_tei)
            })
            
        df = pd.DataFrame(data)
        
        discrepancies = df[~df['labels_match']]
        if not discrepancies.empty:
            logger.info(f"\n\nFound {len(discrepancies)} label mismatches:")
            for _, row in discrepancies.iterrows():
                logger.info(f"\nText: {row['text']}")
                logger.info(f"  Candle: {row['candle_label']} ({row['candle_score']:.4f})")
                logger.info(f"  HF: {row['hf_label']} ({row['hf_score']:.4f})")
                logger.info(f"  TEI: {row['tei_label']} ({row['tei_score']:.4f})")
                
        return df

def main():
    parser = argparse.ArgumentParser(description='Test DeBERTa implementations')
    parser.add_argument('--model-id', default='meta-llama/Prompt-Guard-86M', help='Model ID')
    parser.add_argument('--candle-binary', default='/Users/jie/niko/candle/target/release/examples/debertav2', help='Path to Candle binary')
    parser.add_argument('--tei-url', default='http://127.0.0.1:8080/predict', help='TEI endpoint URL')
    parser.add_argument('--num-samples', type=int, default=100, help='Number of samples to test')
    parser.add_argument('--output-csv', default='test_results.csv', help='Output CSV file')
    
    args = parser.parse_args()
    
    framework = DeBERTaTestFramework(
        model_id=args.model_id,
        candle_binary_path=args.candle_binary,
        tei_url=args.tei_url,
        num_samples=args.num_samples
    )
    
    logger.info("Starting tests...")
    results = framework.run_tests()
    
    df = framework.generate_report(results)
    print(f"Average distance hf TEI: {df['score_diff_hf_tei'].mean()}")
    print(f"Max distance hf TEI: {df['score_diff_hf_tei'].max()}")

    print(f"Average distance hf candle: {df['score_diff_hf_candle'].mean()}")
    print(f"Max distance hf candle: {df['score_diff_hf_candle'].max()}")
    
    if df is not None and not df.empty:
        df.to_csv(args.output_csv, index=False)
        logger.info(f"\nDetailed results saved to {args.output_csv}")
    
if __name__ == "__main__":
    main()