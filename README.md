## Introduction

**Inference** is the process where a trained Large Language Model (LLM) takes a user’s prompt and generates a response by predicting one token at a time until the output is complete.
It is the stage where the model applies everything in it's training to produce answers without updating its knowledge. Inference speed and efficiency directly impact user experience, making it a critical focus for performance benchmarking.

In this project, there is a focus on 2:

- **Streaming inference** is where model begins sending tokens to the user as soon as they are generated (similar to typing). This improves perceived responsiveness but may require more complex infrastructure.

- **Batch inference** is where the model processes one or more prompts fully before returning results (sending a full prompt).  This can be more efficient for high-throughput workloads but may introduce longer wait times for the first output.

This project benchmarks streaming vs batch LLM inference to understand the trade-offs of metrics in speed, throughput, and memory usage.

### Metrics 

1) Total Latency (ms) – Time from sending the prompt to receiving the complete response.
2) Token Throughput (tokens/sec) – Average rate of token generation.
3) Memory Usage (MB) – Peak memory usage during inference.


### Why This Project Matters
Inference performance directly affects user experience and system efficiency:

1) Low latency = faster responses in interactive applications.
2) High throughput = more users served concurrently.
3) Lower memory footprint = cheaper and more scalable deployments.

By comparing streaming vs batch, this project helps answers; When is streaming better than batch? What trade-offs exist between speed and memory usage? How do inference strategies affect resource allocation?