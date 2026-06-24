import torch
import pandas as pd
from src.models import build_model
from src.benchmark import run_inference
from src.plotting import plot_batch_scaling

def main():
    # ==========================================
    # STAGE 8: torch.compile Optimization
    # ==========================================
    print("Starting Stage 8: torch.compile Optimization")
    print("=" * 60)

    model_name = "ResNet18"
    batch_size = 32
    input_shape = (batch_size, 3, 224, 224)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("torch.compile benchmarking is best done on CUDA. Please switch to GPU!")
        return
        
    print(f"Testing {model_name} on {device.upper()} with Batch Size {batch_size}...")
    print("Note: The first compilation run will pause for a few seconds. This is normal!\n")
    
    all_results = []
    
    # 1. Run Standard Eager Mode
    print("Running Eager Mode (Standard)...")
    model_eager = build_model(model_name)
    metrics_eager = run_inference(
        model_eager, input_shape=input_shape, num_runs=100, device=device, use_compile=False
    )
    metrics_eager['mode'] = "Eager"
    all_results.append(metrics_eager)

    # 2. Run Compiled Mode
    print("Running Compiled Mode (Wait for it to compile)...")
    model_compiled = build_model(model_name)
    metrics_compiled = run_inference(
        model_compiled, input_shape=input_shape, num_runs=100, device=device, use_compile=True
    )
    metrics_compiled['mode'] = "Compiled"
    all_results.append(metrics_compiled)

    # Display Results
    df = pd.DataFrame(all_results)
    df = df[['mode', 'latency_mean_ms', 'throughput_samples_per_sec', 'peak_memory_mb']]
    
    output_path = "results/compile_comparison.csv"
    df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 60)
    print("Final torch.compile Comparison:")
    print(df.to_string(index=False))
    
    eager_t = df.loc[df['mode'] == 'Eager', 'throughput_samples_per_sec'].values[0]
    comp_t = df.loc[df['mode'] == 'Compiled', 'throughput_samples_per_sec'].values[0]
    print(f"\n🔥 Compiler Speedup: {comp_t / eager_t:.2f}x faster")


    # ==========================================
    # STAGE 9: Summary Tables and Plots
    # ==========================================
    print("\n\nStarting Stage 9: Generating Plots")
    print("=" * 60)
    
    print("Reading CSV and generating Batch Size Scaling Tradeoff Plot...")
    
    # This will read the CSV from Stage 4 and save the PNG to your plots folder
    plot_batch_scaling()
    
    print("=" * 60)
    print("All stages complete! Go check your 'results/plots/' folder for the new graph!")

if __name__ == "__main__":
    main()