## Introduction

This project benchmarks LLM inference performance by comparing streaming and batch generation modes across multiple key metrics

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