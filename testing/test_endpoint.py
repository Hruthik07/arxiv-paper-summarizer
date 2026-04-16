"""
test_endpoint.py

Comprehensive testing suite for the arxiv-summarizer SageMaker endpoint.
Runs three test suites:
  1. Quality  — ROUGE + BERTScore across different paper domains/lengths
  2. Performance — Latency percentiles (p50/p95/p99) and throughput
  3. Load — Concurrent request scaling (1/5/10/20 threads)

Usage:
    python testing/test_endpoint.py --suite all
    python testing/test_endpoint.py --suite quality
    python testing/test_endpoint.py --suite performance
    python testing/test_endpoint.py --suite load --concurrency-levels 1,5,10,20
"""

import argparse
import json
import logging
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import boto3
from botocore.config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ENDPOINT_NAME = "arxiv-summarizer-endpoint"
CONTENT_TYPE = "application/json"
BOTO_CONFIG = Config(read_timeout=120, retries={"max_attempts": 0})

# ---------------------------------------------------------------------------
# Test Data — 8 samples across domains, lengths, and edge cases
# ---------------------------------------------------------------------------
SAMPLE_PAPERS = [
    {
        "id": "ml_lora",
        "domain": "ML/NLP",
        "length_category": "short",
        "input_text": (
            "We propose a novel approach to parameter-efficient fine-tuning "
            "of large language models using low-rank adaptation (LoRA). Our method "
            "significantly reduces the number of trainable parameters while maintaining "
            "model quality on downstream NLP tasks. Experiments on GPT-3 show that LoRA "
            "matches or exceeds full fine-tuning performance with up to 10,000x fewer "
            "trainable parameters, enabling fine-tuning on consumer-grade hardware."
        ),
        "reference_summary": (
            "A parameter-efficient fine-tuning method using low-rank adaptation that "
            "matches full fine-tuning performance with far fewer trainable parameters."
        ),
    },
    {
        "id": "cv_detection",
        "domain": "Computer Vision",
        "length_category": "medium",
        "input_text": (
            "Object detection has been significantly advanced by deep learning methods. "
            "We present an improved single-stage detector that addresses the class imbalance "
            "problem during training through a novel focal loss function. Traditional cross-entropy "
            "loss treats all examples equally, but in object detection the vast majority of "
            "candidate locations are easy negatives. Our focal loss focuses training on hard "
            "examples by down-weighting the contribution of easy negatives. We demonstrate that "
            "a simple one-stage detector trained with focal loss achieves state-of-the-art results "
            "on COCO, surpassing all previous one-stage detectors and matching the accuracy of "
            "more complex two-stage detectors like Faster R-CNN. Our best model achieves 40.8 AP "
            "on the COCO test-dev set, running at 5 FPS on a single GPU. Extensive ablation "
            "experiments validate the effectiveness of the focal loss across different backbone "
            "architectures and training configurations."
        ),
        "reference_summary": (
            "A focal loss function for single-stage object detectors that addresses class "
            "imbalance by down-weighting easy negatives, achieving state-of-the-art results on COCO."
        ),
    },
    {
        "id": "physics_cosmicray",
        "domain": "Physics",
        "length_category": "long",
        "input_text": (
            "the hires collaboration has recently announced preliminary measurements of the "
            "energy spectrum of ultra - high energy cosmic rays ( uhecr ), as seen in monocular "
            "analyses from each of the two hires sites. the cosmic ray energy spectrum is nearly "
            "featureless over ten orders of magnitude in energy, from 1e9 ev to 1e20 ev. there "
            "are two types of models describing the sources of ultra - high energy cosmic rays: "
            "astrophysical models (bottom-up), in which cosmic rays are accelerated to very high "
            "energies by magnetic shock fronts moving though plasmas; and cosmological models "
            "(top-down), in which cosmic rays are the result of the decays of super heavy particles "
            "which are relics of the big bang. these sources may plausibly give cosmic rays at "
            "1e9 ev, but in all cases, one is pushing the bounds of plausibility at the highest "
            "energies. this spectrum is consistent with the existence of the gzk cutoff, as well "
            "other aspects of the energy loss processes that cause the gzk cutoff. based on the "
            "analytic energy loss formalism of berezinsky et al., the hires spectra favor a "
            "distribution of extragalactic sources that has a similar distribution to that of "
            "luminous matter in the universe, both in its local over-density and in its "
            "cosmological evolution. uhecrs have a very low flux, so one must have a large "
            "collection area to obtain a reasonable event rate."
        ),
        "reference_summary": (
            "the hires collaboration has recently announced preliminary measurements of the "
            "energy spectrum of ultra - high energy cosmic rays ( uhecr ). this spectrum is "
            "consistent with the existence of the gzk cutoff. the hires spectra favor a "
            "distribution of extragalactic sources similar to luminous matter in the universe."
        ),
    },
    {
        "id": "math_topology",
        "domain": "Mathematics",
        "length_category": "medium",
        "input_text": (
            "We study the topology of the space of smooth embeddings of the circle into "
            "three-dimensional Euclidean space, known as the space of knots. Using techniques "
            "from algebraic topology and homotopy theory, we establish new results on the "
            "homology groups of this embedding space. Our main theorem shows that certain "
            "characteristic classes, derived from configuration space integrals, generate "
            "non-trivial homology classes in every dimension. We also prove that the space "
            "of long knots is rationally equivalent to a product of Eilenberg-MacLane spaces, "
            "confirming a conjecture of Vassiliev. These results have implications for the "
            "classification of knot invariants and provide new computational tools for "
            "understanding the global structure of knot spaces."
        ),
        "reference_summary": (
            "New results on the homology of the space of knots using algebraic topology, "
            "confirming Vassiliev's conjecture on the rational homotopy type of long knot spaces."
        ),
    },
    {
        "id": "ml_transformers",
        "domain": "ML/NLP",
        "length_category": "short",
        "input_text": (
            "We introduce a new pre-training objective for transformer language models that "
            "combines masked language modeling with a sentence-level coherence prediction task. "
            "Unlike BERT which uses next sentence prediction, our approach predicts whether "
            "a set of sentences form a coherent paragraph or have been randomly shuffled. "
            "Pre-training on large unlabeled corpora followed by task-specific fine-tuning "
            "achieves new state-of-the-art results on the GLUE benchmark, SQuAD 2.0, and "
            "several text classification tasks. Ablation studies confirm that the coherence "
            "objective contributes meaningfully to downstream performance."
        ),
        "reference_summary": (
            "A transformer pre-training method combining masked language modeling with coherence "
            "prediction that achieves state-of-the-art results on GLUE, SQuAD, and text classification."
        ),
    },
    {
        "id": "bio_genomics",
        "domain": "Biology",
        "length_category": "medium",
        "input_text": (
            "Genome-wide association studies (GWAS) have identified thousands of genetic variants "
            "associated with complex diseases, but the biological mechanisms remain largely unknown. "
            "We develop a computational framework that integrates GWAS summary statistics with "
            "single-cell RNA sequencing data to identify disease-relevant cell types and gene "
            "regulatory programs. Our method uses a hierarchical Bayesian model to partition "
            "heritability across cell types while accounting for linkage disequilibrium and "
            "gene expression specificity. Applied to 30 complex traits and diseases using "
            "single-cell data from 15 tissues, we identify novel cell-type associations for "
            "autoimmune diseases, neuropsychiatric disorders, and metabolic traits. For example, "
            "we find that heritability for type 2 diabetes is enriched in pancreatic beta cells "
            "and specific subtypes of adipocytes. Our results provide a roadmap for functional "
            "follow-up of GWAS findings at cellular resolution."
        ),
        "reference_summary": (
            "A Bayesian framework integrating GWAS and single-cell RNA-seq data to identify "
            "disease-relevant cell types, revealing novel associations across 30 complex traits."
        ),
    },
    {
        "id": "edge_short",
        "domain": "Edge Case",
        "length_category": "minimal",
        "input_text": (
            "We present a new algorithm for matrix multiplication that runs in O(n^2.37) time, "
            "improving the previous best bound."
        ),
        "reference_summary": (
            "A faster matrix multiplication algorithm with improved time complexity."
        ),
    },
    {
        "id": "edge_latex",
        "domain": "Edge Case",
        "length_category": "medium",
        "input_text": (
            "we describe the design and performance of the medium resolution spectrometer "
            "( mrs ) for the jwst - miri instrument. the mrs incorporates four coaxial spectral "
            "channels in a compact opto - mechanical layout that generates spectral images over "
            "fields of view up to 7.7 x 7.7 arcseconds in extent and at spectral resolving "
            "powers ranging from 1,300 to 3,700. each channel includes an all - reflective "
            "integral field unit ( ifu ) : an image slicer that reformats the input field for "
            "presentation to a grating spectrometer. two 1024 x 1024 focal plane arrays record "
            "the output spectral images with an instantaneous spectral coverage of approximately "
            "one third of the full wavelength range of each channel. the full 5 to 28.5 @xmath0 m "
            "spectrum is then obtained by making three exposures using gratings and pass-band "
            "determining filters that are selected using just two three-position mechanisms."
        ),
        "reference_summary": (
            "we describe the design and performance of the medium resolution spectrometer "
            "( mrs ) for the jwst - miri instrument. the mrs incorporates four coaxial spectral "
            "channels that generate spectral images at resolving powers from 1,300 to 3,700."
        ),
    },
]


# ---------------------------------------------------------------------------
# Endpoint invocation helper
# ---------------------------------------------------------------------------
def invoke_endpoint(
    region: str, endpoint_name: str, text: str
) -> tuple[str, float, int]:
    """Invoke the SageMaker endpoint. Returns (summary, latency_seconds, status_code).

    Creates its own boto3 client — safe to call from any thread.
    """
    client = boto3.client("sagemaker-runtime", region_name=region, config=BOTO_CONFIG)
    payload = json.dumps({"inputs": "summarize: " + text})

    start = time.perf_counter()
    try:
        response = client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType=CONTENT_TYPE,
            Body=payload,
        )
        elapsed = time.perf_counter() - start
        result = json.loads(response["Body"].read().decode())
        summary = result[0]["summary_text"] if isinstance(result, list) else str(result)
        return summary, elapsed, 200
    except client.exceptions.ModelError as e:
        elapsed = time.perf_counter() - start
        return f"ERROR: {e}", elapsed, 500
    except Exception as e:
        elapsed = time.perf_counter() - start
        return f"ERROR: {e}", elapsed, 0


# ---------------------------------------------------------------------------
# Quality Test Suite
# ---------------------------------------------------------------------------
def run_quality_tests(region: str, endpoint_name: str) -> dict:
    """Test summary quality across domains and lengths."""
    print("\n" + "=" * 80)
    print("  QUALITY TEST SUITE")
    print("=" * 80)

    predictions = []
    references = []
    per_sample = []

    for sample in SAMPLE_PAPERS:
        logger.info(f"  Testing: {sample['id']} ({sample['domain']}, {sample['length_category']})")
        summary, latency, status = invoke_endpoint(region, endpoint_name, sample["input_text"])

        predictions.append(summary)
        references.append(sample["reference_summary"])
        per_sample.append({
            "id": sample["id"],
            "domain": sample["domain"],
            "length_category": sample["length_category"],
            "prediction": summary,
            "reference": sample["reference_summary"],
            "latency": round(latency, 2),
            "status": status,
        })

    # Compute ROUGE
    import evaluate as hf_evaluate

    rouge = hf_evaluate.load("rouge")
    rouge_scores = rouge.compute(
        predictions=predictions, references=references, use_stemmer=True
    )
    rouge_pct = {k: round(v * 100, 2) for k, v in rouge_scores.items()}

    # Per-sample ROUGE-L
    for i, sample in enumerate(per_sample):
        sample_rouge = rouge.compute(
            predictions=[predictions[i]], references=[references[i]], use_stemmer=True
        )
        sample["rougeL"] = round(sample_rouge["rougeL"] * 100, 2)

    # Compute BERTScore
    logger.info("  Computing BERTScore ...")
    bertscore = hf_evaluate.load("bertscore")
    bert_result = bertscore.compute(
        predictions=predictions,
        references=references,
        model_type="distilbert-base-uncased",
    )
    bert_f1 = round(
        sum(bert_result["f1"]) / len(bert_result["f1"]) * 100, 2
    )

    # Print per-sample table
    print(f"\n{'ID':<20} {'Domain':<18} {'Length':<10} {'ROUGE-L':<10} {'Latency':<10} Summary Preview")
    print("-" * 120)
    for s in per_sample:
        preview = s["prediction"][:50] + "..." if len(s["prediction"]) > 50 else s["prediction"]
        print(
            f"{s['id']:<20} {s['domain']:<18} {s['length_category']:<10} "
            f"{s['rougeL']:<10} {s['latency']:<10} {preview}"
        )

    # Print aggregate
    print("\n" + "-" * 80)
    print("  AGGREGATE QUALITY SCORES")
    print("-" * 80)
    for k, v in rouge_pct.items():
        print(f"  {k:<20} {v}")
    print(f"  {'bertscore_f1':<20} {bert_f1}")
    print("=" * 80)

    # Pass/fail
    passed = rouge_pct.get("rouge1", 0) > 35
    status_str = "PASS" if passed else "FAIL"
    print(f"  Quality gate (ROUGE-1 > 35): {status_str}")

    return {
        "aggregate_rouge": rouge_pct,
        "bertscore_f1": bert_f1,
        "per_sample": per_sample,
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Performance Test Suite
# ---------------------------------------------------------------------------
def run_performance_tests(
    region: str, endpoint_name: str, num_iterations: int = 20
) -> dict:
    """Measure latency percentiles with sequential requests."""
    print("\n" + "=" * 80)
    print("  PERFORMANCE TEST SUITE")
    print("=" * 80)

    warmup = 2
    latencies = []
    errors = 0

    total_start = time.perf_counter()
    for i in range(num_iterations):
        sample = SAMPLE_PAPERS[i % len(SAMPLE_PAPERS)]
        tag = "warmup" if i < warmup else f"req {i - warmup + 1}"
        logger.info(f"  [{tag}] {sample['id']}")

        _, latency, status = invoke_endpoint(region, endpoint_name, sample["input_text"])

        if i >= warmup:
            latencies.append(latency)
            if status != 200:
                errors += 1

    total_elapsed = time.perf_counter() - total_start
    measured = num_iterations - warmup

    latencies_sorted = sorted(latencies)
    p50_idx = int(len(latencies_sorted) * 0.50)
    p95_idx = min(int(len(latencies_sorted) * 0.95), len(latencies_sorted) - 1)
    p99_idx = min(int(len(latencies_sorted) * 0.99), len(latencies_sorted) - 1)

    results = {
        "total_requests": measured,
        "errors": errors,
        "error_rate_pct": round(errors / measured * 100, 2) if measured else 0,
        "latency_min": round(min(latencies), 3),
        "latency_max": round(max(latencies), 3),
        "latency_mean": round(statistics.mean(latencies), 3),
        "latency_p50": round(latencies_sorted[p50_idx], 3),
        "latency_p95": round(latencies_sorted[p95_idx], 3),
        "latency_p99": round(latencies_sorted[p99_idx], 3),
        "latency_stddev": round(statistics.stdev(latencies), 3) if len(latencies) > 1 else 0,
        "throughput_rps": round(measured / total_elapsed, 3),
    }

    print(f"\n  Total requests:     {results['total_requests']} (after {warmup} warmup)")
    print(f"  Errors:             {results['errors']} ({results['error_rate_pct']}%)")
    print("-" * 50)
    print(f"  Latency (seconds):")
    print(f"    Min:              {results['latency_min']}")
    print(f"    Max:              {results['latency_max']}")
    print(f"    Mean:             {results['latency_mean']}")
    print(f"    P50 (median):     {results['latency_p50']}")
    print(f"    P95:              {results['latency_p95']}")
    print(f"    P99:              {results['latency_p99']}")
    print(f"    Std Dev:          {results['latency_stddev']}")
    print("-" * 50)
    print(f"  Throughput:         {results['throughput_rps']} req/sec (serial)")
    print("=" * 80)

    passed = results["latency_p95"] < 10 and results["error_rate_pct"] < 5
    status_str = "PASS" if passed else "FAIL"
    print(f"  Performance gate (P95 < 10s, errors < 5%): {status_str}")
    results["passed"] = passed

    return results


# ---------------------------------------------------------------------------
# Load Test Suite
# ---------------------------------------------------------------------------
def run_load_tests(
    region: str,
    endpoint_name: str,
    concurrency_levels: list[int],
    requests_per_level: int,
) -> dict:
    """Test endpoint under concurrent load."""
    print("\n" + "=" * 80)
    print("  LOAD TEST SUITE")
    print("=" * 80)

    level_results = []

    for concurrency in concurrency_levels:
        logger.info(f"  Load level: {concurrency} concurrent threads, {requests_per_level} requests")
        latencies = []
        errors = 0

        total_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = []
            for i in range(requests_per_level):
                sample = SAMPLE_PAPERS[i % len(SAMPLE_PAPERS)]
                future = executor.submit(
                    invoke_endpoint, region, endpoint_name, sample["input_text"]
                )
                futures.append(future)

            for future in as_completed(futures):
                summary, latency, status = future.result()
                latencies.append(latency)
                if status != 200:
                    errors += 1

        total_elapsed = time.perf_counter() - total_start
        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)
        p50_idx = int(n * 0.50)
        p95_idx = min(int(n * 0.95), n - 1)
        p99_idx = min(int(n * 0.99), n - 1)

        level_result = {
            "concurrency": concurrency,
            "total_requests": requests_per_level,
            "errors": errors,
            "error_rate_pct": round(errors / requests_per_level * 100, 2),
            "latency_p50": round(latencies_sorted[p50_idx], 3),
            "latency_p95": round(latencies_sorted[p95_idx], 3),
            "latency_p99": round(latencies_sorted[p99_idx], 3),
            "throughput_rps": round(requests_per_level / total_elapsed, 3),
            "total_time": round(total_elapsed, 2),
        }
        level_results.append(level_result)

        # Pause between levels
        if concurrency != concurrency_levels[-1]:
            logger.info("  Pausing 5s for endpoint stabilization ...")
            time.sleep(5)

    # Print comparison table
    print(f"\n{'Concurrency':<14} {'Requests':<10} {'Errors':<8} {'P50(s)':<9} "
          f"{'P95(s)':<9} {'P99(s)':<9} {'Throughput':<12} {'Total(s)'}")
    print("-" * 95)
    for r in level_results:
        print(
            f"{r['concurrency']:<14} {r['total_requests']:<10} {r['errors']:<8} "
            f"{r['latency_p50']:<9} {r['latency_p95']:<9} {r['latency_p99']:<9} "
            f"{r['throughput_rps']:<12} {r['total_time']}"
        )
    print("=" * 80)

    # Pass/fail on max concurrency level
    max_level = level_results[-1]
    passed = max_level["error_rate_pct"] < 10 and max_level["latency_p95"] < 15
    status_str = "PASS" if passed else "FAIL"
    print(f"  Load gate (at max concurrency: errors < 10%, P95 < 15s): {status_str}")

    return {"levels": level_results, "passed": passed}


# ---------------------------------------------------------------------------
# Final Report
# ---------------------------------------------------------------------------
class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types from evaluate/bert_score."""
    def default(self, obj):
        import numpy as np
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)


def save_report(results: dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    logger.info(f"Results saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Test arxiv-summarizer SageMaker endpoint")
    parser.add_argument("--endpoint-name", default=ENDPOINT_NAME)
    parser.add_argument(
        "--region", default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    )
    parser.add_argument(
        "--suite",
        choices=["quality", "performance", "load", "all"],
        default="all",
    )
    parser.add_argument("--concurrency-levels", default="1,5,10,20")
    parser.add_argument("--requests-per-level", type=int, default=20)
    parser.add_argument("--output-file", default=None)
    args = parser.parse_args()

    concurrency_levels = [int(x) for x in args.concurrency_levels.split(",")]

    print("\n" + "#" * 80)
    print("  arXiv Summarizer Endpoint — Comprehensive Test Suite")
    print(f"  Endpoint : {args.endpoint_name}")
    print(f"  Region   : {args.region}")
    print(f"  Suite    : {args.suite}")
    print(f"  Time     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 80)

    # Warmup / connectivity check
    logger.info("Verifying endpoint connectivity ...")
    summary, latency, status = invoke_endpoint(
        args.region, args.endpoint_name, "Test connectivity."
    )
    if status != 200:
        logger.error(f"Endpoint not reachable: {summary}")
        return
    logger.info(f"Endpoint alive (latency: {latency:.2f}s)")

    results = {"endpoint": args.endpoint_name, "timestamp": datetime.now().isoformat()}

    if args.suite in ("quality", "all"):
        results["quality"] = run_quality_tests(args.region, args.endpoint_name)

    if args.suite in ("performance", "all"):
        results["performance"] = run_performance_tests(args.region, args.endpoint_name)

    if args.suite in ("load", "all"):
        results["load"] = run_load_tests(
            args.region, args.endpoint_name, concurrency_levels, args.requests_per_level
        )

    # Final summary
    print("\n" + "#" * 80)
    print("  FINAL SUMMARY")
    print("#" * 80)
    for suite_name in ("quality", "performance", "load"):
        if suite_name in results:
            passed = results[suite_name].get("passed", "N/A")
            icon = "PASS" if passed else "FAIL"
            print(f"  {suite_name:<15} {icon}")
    print("#" * 80 + "\n")

    # Save results
    output_path = args.output_file or (
        f"testing/test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    save_report(results, output_path)


if __name__ == "__main__":
    main()
