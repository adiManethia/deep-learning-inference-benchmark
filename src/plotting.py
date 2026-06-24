import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_batch_scaling():
    """
    Reads the batch size scaling CSV and generates a throughput/latency plot.
    """
    # 1. Make sure our plots folder exists
    os.makedirs("results/plots", exist_ok=True)
    
    # 2. Load the data
    file_path = "results/batch_size_scaling.csv"
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found. Run Stage 4 first!")
        return

    df = pd.read_csv(file_path)

    # 3. Create a figure with two subplots (side-by-side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Plot 1: Throughput ---
    ax1.plot(df['batch_size'], df['throughput_samples_per_sec'], marker='o', color='blue', linewidth=2)
    ax1.set_title('Throughput vs. Batch Size', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Batch Size', fontsize=12)
    ax1.set_ylabel('Samples per Second', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # --- Plot 2: Latency per Sample ---
    ax2.plot(df['batch_size'], df['latency_per_sample_ms'], marker='s', color='red', linewidth=2)
    ax2.set_title('Latency per Sample vs. Batch Size', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Batch Size', fontsize=12)
    ax2.set_ylabel('Milliseconds (ms)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)

    # 4. Save the plot
    plt.tight_layout()
    save_path = "results/plots/batch_scaling_tradeoff.png"
    plt.savefig(save_path, dpi=300) # High resolution for GitHub
    plt.close()
    
    print(f"Plot successfully saved to {save_path}")