## Status  
![status](https://img.shields.io/badge/status-actively--developed-brightgreen)  
This project is **currently** being developed and improved with additional features, optimizations, and testing.  

## Introduction

**Inference** is the process where a trained Large Language Model (LLM) takes a user’s prompt and generates a response by predicting one token at a time until the output is complete.
It is the stage where the model applies everything in it's training to produce answers without updating its knowledge. Inference speed and efficiency directly impact user experience, making it a critical focus for performance benchmarking.

In this project, there is a focus on 2:

- **Streaming inference** is where model begins sending tokens to the user as soon as they are generated (similar to typing). Streaming is ideal for interactive, real-time applications where perceived responsiveness matters like chatbots, live translation, or coding assistants.


- **Batch inference** is where the model processes one or more prompts fully before returning results (sending a full prompt).  Batching is better for large-scale, high-volume workloads where efficiency per request is more important than immediate partial output like document processing, report generation, or bulk question answering.

This project benchmarks streaming vs batch LLM inference to understand the trade-offs of metrics in speed, throughput, and memory usage.

### Metrics 

1) Total Latency (ms) – Time from sending the prompt to receiving the complete response.
2) Token Throughput (tokens/sec) – Average rate of token generation.
3) Memory Usage (MB) – Peak memory usage during inference.


### Why This Project Matters
Inference performance directly affects user experience and system efficiency:

1) Low latency → Faster responses in interactive apps like chat or code assistants, improving user satisfaction. Even though the answer is correct, users perceive slow responses as poor quality — especially in real-time tools.

2) High throughput → More requests served per second, enabling scalability without extra hardware. The higher your throughput, the more people you can serve at the same time without slowing everyone down.

3) Lower memory footprint → Fit more models or requests per GPU, reducing infrastructure costs.
Lower memory footprint also reduces the risk of out-of-memory crashes, keeping systems stable under heavy load.

By comparing streaming vs batch, this project helps answers; When is streaming better than batch? What trade-offs exist between speed and memory usage? How do inference strategies affect resource allocation?


### Tech Stack  
**Python** · **Hugging Face Transformers** · **FastAPI** · **Docker** · **PyTorch** · **pandas** · **scikit-learn**  
- **pytest** · **Streamlit** · **GitHub Actions (CI)** · **logging** · **dotenv** · **modular code structure**  


### Set Up
TBD