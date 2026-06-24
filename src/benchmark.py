import time 
import torch 


def run_inference(model, input_shape=(1, 3, 224, 224), num_runs=100, device="cpu", precision="fp32", use_compile=False):
    """
    Runs PyTorch model on the specified device and returns metrics.
    """

    # move model and data to correct device 
    model = model.to(device)
    
    # batch_size = input_shape[0]
    dummy_input = torch.randn(input_shape).to(device)   # generate fake data ( a random tensor ) 

    # cast half precisio, fp16
    if precision == "fp16" and device == "cuda":
        model = model.half()
        dummy_input = dummy_input.half()

    # torch.compile for graph optimization
    if use_compile and device == "cuda":
        # mode=reduce-overhead, tells PyTorch to optimze for speed
        model = torch.compile(model, mode="reduce-overhead")

    # put model in evaluation mode 
    model.eval()

    # warm up runs
    # Few dry runs because first few passes are usually slower as PyTorch initiailizes memory.
    with torch.inference_mode():
        for _ in range(10):
            _ = model(dummy_input)

    # wait for gpu to finish warming up before starting clock 
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()  # clear internal memory tracker 
    
    # actual timed benchmark 
    start_time = time.perf_counter()

    with torch.inference_mode():
        for _ in range(num_runs):
            _ = model(dummy_input)


    # grab peak memory right after finish 
    if device == "cuda":
        torch.cuda.synchronize()
        peak_memory_bytes = torch.cuda.max_memory_allocated()

        # convert bytes to Megabytes
        peak_memory_mb = peak_memory_bytes / (1024 * 1024)

    else:
        peak_memory_mb = 0.0

    end_time = time.perf_counter()


    batch_size = input_shape[0]
    total_time_sec = end_time - start_time
    time_per_batch_sec = total_time_sec / num_runs
    
    latency_mean_ms = time_per_batch_sec * 1000
    latency_per_sample_ms = latency_mean_ms / batch_size
    throughput_samples_per_sec = batch_size / time_per_batch_sec

    return {
        "latency_mean_ms": latency_mean_ms,
        "latency_per_sample_ms": latency_per_sample_ms,
        "throughput_samples_per_sec": throughput_samples_per_sec,
        "peak_memory_mb": peak_memory_mb  # Added to our results dictionary!
    }