import pandas as pd
import onnxruntime as ort
from src.benchmark_onnx import run_onnx_inference

def main():
    print("Starting Stage 11: ONNX Runtime Benchmark")
    print("=" * 60)

    models = ["SmallCNN", "ResNet18", "MobileNetV2"]
    batch_size = 32
    
    # Check if ORT detects your GPU
    available_providers = ort.get_available_providers()
    print(f"ONNX Runtime sees these providers: {available_providers}\n")

    providers_to_test = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available_providers:
        providers_to_test.append("CUDAExecutionProvider")

    all_results = []

    for model_name in models:
        onnx_path = f"exports/{model_name}.onnx"
        
        for provider in providers_to_test:
            print(f"Running {model_name} on {provider}...")
            
            try:
                metrics = run_onnx_inference(
                    onnx_path=onnx_path, 
                    batch_size=batch_size, 
                    num_runs=100, 
                    provider=provider
                )
                
                metrics['model'] = model_name
                metrics['runtime'] = "ONNX Runtime"
                metrics['provider'] = provider.replace("ExecutionProvider", "")
                all_results.append(metrics)
                
            except Exception as e:
                print(f" Failed to run {model_name} on {provider}: {e}")

    # Display Results
    df = pd.DataFrame(all_results)
    df = df[['model', 'provider', 'latency_mean_ms', 'throughput_samples_per_sec']]
    
    output_path = "results/onnxruntime_benchmark.csv"
    df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 60)
    print("Final ONNX Runtime Results:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()