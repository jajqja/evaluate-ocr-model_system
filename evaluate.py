import json
import argparse
from pathlib import Path
from dataclasses import asdict
import csv
import time

from models import Sample, ModelResult, ModelSummary

from dataloader import load_dataset
from load_model_custom import load_model, infer
from metrics import compute_metrics


def run_model(samples: list[Sample], output_dir: str) -> ModelSummary:
    print(f"  Loading model...", end="", flush=True)
    t_load = time.time()
    try:
        model_name, model, processor = load_model()

    except Exception as e:
        print(f"\n[ERROR] Không thể load model '{model_name}': {e}")
        return ModelSummary(
            model_name=model_name, avg_wer=1.0, avg_cer=1.0,
            avg_latency_ms=0, p50_latency_ms=0, p95_latency_ms=0,
            total_samples=len(samples), failed_samples=len(samples),
            results=[ModelResult(
                sample_id=s.sample_id,
                prediction="", ground_truth=s.ground_truth,
                wer=1.0, cer=1.0, latency_ms=0,
                error=f"Load model error: {e}"
            ) for s in samples]
        )

    elapsed_load = time.time() - t_load
    print(f" done ({elapsed_load:.1f}s)")

    results = []
    for i, sample in enumerate(samples):
        prefix = f"  [{i+1:>4}/{len(samples)}]"
        try:
            t0 = time.perf_counter()
            pred = infer(model, processor, sample.image_path)
            latency_ms = (time.perf_counter() - t0) * 1000

            metrics = compute_metrics(sample.ground_truth, pred)
            results.append(ModelResult(
                sample_id=sample.sample_id,
                prediction=pred,
                ground_truth=sample.ground_truth,
                wer=metrics["wer"],
                cer=metrics["cer"],
                latency_ms=round(latency_ms, 2),
            ))
            with open(f"{output_dir}/md/{sample.sample_id}.md", "w") as f:
                f.write("-----Prediction-----")
                f.write(pred)
                f.write("\n\n-----Ground Truth-----")
                f.write(sample.ground_truth)
            print(f"{prefix} WER={metrics['wer_pct']:5.1f}%  CER={metrics['cer_pct']:5.1f}%  {latency_ms:.0f}ms  {sample.sample_id}")

        except Exception as e:
            results.append(ModelResult(
                sample_id=sample.sample_id,
                prediction="", ground_truth=sample.ground_truth,
                wer=1.0, cer=1.0, latency_ms=0, error=str(e)
            ))
            print(f"{prefix} [ERROR] {sample.sample_id}: {e}")

    # ── Aggregate ──
    valid = [r for r in results if not r.error]
    failed = len(results) - len(valid)

    if not valid:
        return ModelSummary(
            model_name=model_name, avg_wer=1.0, avg_cer=1.0,
            avg_latency_ms=0, p50_latency_ms=0, p95_latency_ms=0,
            total_samples=len(results), failed_samples=failed,
            results=results
        )

    latencies = sorted(r.latency_ms for r in valid)
    p50 = latencies[int(len(latencies) * 0.50)]
    p95 = latencies[min(int(len(latencies) * 0.95), len(latencies)-1)]

    return ModelSummary(
        model_name=model_name,
        avg_wer=round(sum(r.wer for r in valid) / len(valid), 4),
        avg_cer=round(sum(r.cer for r in valid) / len(valid), 4),
        avg_latency_ms=round(sum(r.latency_ms for r in valid) / len(valid), 2),
        p50_latency_ms=round(p50, 2),
        p95_latency_ms=round(p95, 2),
        total_samples=len(results),
        failed_samples=failed,
        results=results,
    )


def save_result(summary: ModelSummary, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summaries.json"
    data = asdict(summary)

    with open(summary_path, "a", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    safe_name = summary.model_name.replace("/", "__")
    csv_path = output_dir / f"{safe_name}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sample_id", "wer", "cer", "latency_ms",
            "ground_truth", "prediction", "error"
        ])
        writer.writeheader()
        for r in summary.results:
            writer.writerow(asdict(r))


def parse_args():
    parser = argparse.ArgumentParser(
        description="OCR Benchmark: WER, CER, Speed",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--dataset", "-d", default="./data",
                        help="Thư mục chứa dataset (default: ./data)")
    parser.add_argument("--output", "-o", default="./results",
                        help="Thư mục lưu kết quả (default: ./results)")
    parser.add_argument("--max-samples", "-n", type=int, default=None,
                        help="Giới hạn số sample để test nhanh")
    return parser.parse_args()


def main():
    args = parse_args()

    samples = load_dataset(args.dataset)
    if args.max_samples:
        samples = samples[:args.max_samples]

    print(f"[Dataset] {len(samples)} samples found")

    summary = run_model(samples)

    save_result(summary, args.output)


if __name__ == "__main__":
    main()