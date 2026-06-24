import time
import numpy as np
import onnxruntime as ort

def run_onnx_inference(onnx_path, batch_size=1, num_runs=100, provider="CPUExecutionProvider"):
    """
    Runs an ONNX model using ONNX Runtime and returns latency and throughput.
    """
    # 1. Start an ONNX Runtime Session and tell it which hardware to use
    session = ort.InferenceSession(onnx_path, providers=[provider])

    # 2. Get the input details dynamically from the ONNX file
    input_name = session.get_inputs()[0].name
    
    # 3. Create dummy data as a NumPy array (Float32)
    dummy_input = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)

    # 4. Warm-up runs
    for _ in range(10):
        _ = session.run(None, {input_name: dummy_input})

    # 5. The timed benchmark
    start_time = time.perf_counter()
    
    for _ in range(num_runs):
        _ = session.run(None, {input_name: dummy_input})
        
    end_time = time.perf_counter()

    # Math
    total_time_sec = end_time - start_time
    time_per_batch_sec = total_time_sec / num_runs
    
    latency_mean_ms = time_per_batch_sec * 1000
    latency_per_sample_ms = latency_mean_ms / batch_size
    throughput_samples_per_sec = batch_size / time_per_batch_sec

    return {
        "latency_mean_ms": latency_mean_ms,
        "latency_per_sample_ms": latency_per_sample_ms,
        "throughput_samples_per_sec": throughput_samples_per_sec
    }