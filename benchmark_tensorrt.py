import pandas as pd
import onnxruntime as ort 
from src.benchmark_onnx import run_onnx_inference 

def main():
    print("Starting TensorRT benchmark ...")
    print("="*80)

    models = ["SmallCNN", "ResNet18", "MobileNetV2"]
    models = ["SmallCNN", "ResNet18", "MobileNetV2"]
    batch_size = 32
    provider = "TensorrtExecutionProvider"
    
    # Check if the provider is actually available
    available = ort.get_available_providers()
    if provider not in available:
        print(f" {provider} not found in ONNX Runtime!")
        print(f"Available: {available}")
        return

    all_results = []

    for model_name in models:
        onnx_path = f"exports/{model_name}.onnx"
        
        print(f"\nBuilding TensorRT Engine for {model_name}...")
        print(" PLEASE WAIT: This might take 30-60 seconds to compile...")
        print("Do not press Ctrl+C!")
        
        try:
            metrics = run_onnx_inference(
                onnx_path=onnx_path, 
                batch_size=batch_size, 
                num_runs=100, 
                provider=provider
            )
            
            metrics['model'] = model_name
            metrics['provider'] = "TensorRT"
            all_results.append(metrics)
            print(f" {model_name} finished successfully!")
            
        except Exception as e:
            print(f" Failed to run {model_name} on TensorRT:")
            print(e)

    # Display Results
    if all_results:
        df = pd.DataFrame(all_results)
        df = df[['model', 'provider', 'latency_mean_ms', 'throughput_samples_per_sec']]
        
        output_path = "results/tensorrt_benchmark.csv"
        df.to_csv(output_path, index=False)
        
        print("\n" + "=" * 60)
        print("Final TensorRT Results:")
        print(df.to_string(index=False))

if __name__ == "__main__":
    main()
