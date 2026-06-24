import os
import pandas as pd

def main():
    print("Starting Stage 13: Final Benchmark Aggregation")
    print("=" * 60)

    # File paths for our generated results
    compile_path = "results/compile_comparison.csv"
    onnx_path = "results/onnxruntime_benchmark.csv"

    if not os.path.exists(compile_path) or not os.path.exists(onnx_path):
        print("Missing some benchmark CSV files! Make sure you ran previous steps.")
        return

    # 1. Read individual files
    df_compile = pd.read_csv(compile_path)
    df_onnx = pd.read_csv(onnx_path)

    # 2. Reformat Stage 8 (torch.compile) data to match standard schema
    # Stage 8 ran explicitly on ResNet18
    compiled_rows = []
    for _, row in df_compile.iterrows():
        compiled_rows.append({
            "model": "ResNet18",
            "framework/engine": f"PyTorch ({row['mode']})",
            "device": "CUDA",
            "latency_mean_ms": row['latency_mean_ms'],
            "throughput_samples_per_sec": row['throughput_samples_per_sec']
        })
    df_py = pd.DataFrame(compiled_rows)

    # 3. Reformat Stage 11 (ONNX Runtime) data
    onnx_rows = []
    for _, row in df_onnx.iterrows():
        device_type = "CUDA" if "CUDA" in row['provider'] else "CPU"
        onnx_rows.append({
            "model": row['model'],
            "framework/engine": "ONNX Runtime",
            "device": device_type,
            "latency_mean_ms": row['latency_mean_ms'],
            "throughput_samples_per_sec": row['throughput_samples_per_sec']
        })
    df_ort = pd.DataFrame(onnx_rows)

    # 4. Combine into a master summary matrix
    master_df = pd.concat([df_py, df_ort], ignore_index=True)
    
    # Sort nicely for presentation
    master_df = master_df.sort_values(by=["model", "device", "throughput_samples_per_sec"], ascending=[True, True, False])

    # Save master report
    master_df.to_csv("results/master_inference_report.csv", index=False)

    print("\n MASTER INFERENCE BENCHMARK REPORT ")
    print("=" * 70)
    print(master_df.to_string(index=False))
    print("=" * 70)
    print("\nReport saved to: results/master_inference_report.csv")

if __name__ == "__main__":
    main()