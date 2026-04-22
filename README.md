# Distributed GPU Load Balancer for LLM Requests

This repository contains the initial implementation scaffold for the CSE354 Distributed Computing project:
"Efficient Load Balancing and GPU Cluster Task Distribution for Handling 1000+ Concurrent LLM Requests."

The project aims to model a distributed AI serving platform that can accept many simultaneous user requests, route them efficiently across multiple GPU-backed workers, enrich prompts with Retrieval-Augmented Generation (RAG), and stay responsive even when the system is under heavy load or partial failure.

## Project Overview

The assignment focuses on designing and implementing a distributed system with the following goals:

- handle 1000+ concurrent LLM requests
- distribute work efficiently across multiple compute nodes
- simulate GPU task scheduling and inference execution
- integrate a RAG pipeline for context-aware responses
- support scalability, observability, and fault tolerance

This repository is currently in the bootstrap stage. It establishes the architecture, directory structure, and stubbed request flow so the team can build the real simulation incrementally without restructuring the project later.

## Assignment Context

According to the project brief, the system should include:

- a client layer that simulates many concurrent users
- a load balancer that distributes requests using scheduling strategies
- a master controller that coordinates task execution
- GPU worker nodes that process inference jobs
- a RAG module that retrieves relevant context
- fault-tolerance logic for node failure and task reassignment

The full target is a distributed prototype that can be evaluated using metrics such as latency, throughput, GPU utilization, scalability under load, and recovery after worker failure.

## Current Repository Status

What is implemented right now:

- a clean Python package layout that matches the assignment architecture
- `Request` and `Response` data models in `common`
- a round-robin load balancer stub in `lb`
- a master scheduler stub in `master`
- GPU worker stubs in `workers`
- a RAG retriever stub in `rag`
- an LLM inference stub in `llm`
- a client-side synthetic request generator in `client`
- a deterministic end-to-end dry run in `main.py`

What is not implemented yet:

- real concurrent execution with `asyncio`, threads, or processes
- real GPU scheduling, batching, or utilization tracking
- real LLM serving or model integration
- real vector store or knowledge-base retrieval
- least-connections or load-aware balancing
- node health checks, heartbeat monitoring, and reassignment
- large-scale benchmarking for 100 to 1000+ concurrent users

## System Architecture

The scaffold follows the architecture described in the assignment.

### 1. Client Layer

The client layer is responsible for generating user requests. In the current scaffold, it creates synthetic requests with stable IDs and sample prompts so the system flow is easy to test and extend.

### 2. Load Balancer

The load balancer is responsible for deciding which worker receives the next request. The current implementation uses round-robin scheduling because it is simple, deterministic, and useful for validating routing behavior before more advanced policies are added.

Future strategies should include:

- least-connections
- load-aware routing
- health-aware routing that skips unavailable workers

### 3. Master Scheduler

The master scheduler acts as the controller of the pipeline. It receives the selected worker, requests additional context from the RAG module, and coordinates the call to the worker so the response can be assembled consistently.

In later phases, this component should also handle:

- scheduling policies
- request prioritization
- retry and reassignment logic
- failure handling and monitoring hooks

### 4. GPU Worker Nodes

Worker nodes simulate compute servers responsible for executing LLM-related jobs. In the current version, workers track active tasks, log request processing, and call the inference stub.

Later, workers can be extended to simulate:

- queue depth
- varying execution times
- hardware capacity differences
- node crashes or timeouts

### 5. RAG Module

The RAG module represents the retrieval layer that supplies supporting context before inference. Right now it returns fixed placeholder context, but the interface is designed so a real retriever can be plugged in later.

Possible future implementations include:

- in-memory document retrieval
- embedding-based search
- vector database integration
- dataset-specific context selection

### 6. LLM Inference Module

The LLM module currently returns a stubbed response that includes the prompt and retrieved context. This keeps the pipeline testable without external dependencies while still showing where real inference logic belongs.

## Request Lifecycle

The current dry-run path is:

1. The client creates a synthetic request.
2. The load balancer selects a worker using round-robin scheduling.
3. The master scheduler asks the RAG module for context.
4. The selected worker processes the request.
5. The worker calls the LLM inference stub.
6. A response object is returned and printed by `main.py`.

This flow is intentionally small and deterministic so each subsystem can be tested independently before large-scale concurrency is added.

## Repository Structure

```text
.
|-- client/
|   `-- load_generator.py
|-- common/
|   `-- models.py
|-- lb/
|   `-- round_robin.py
|-- llm/
|   `-- inference.py
|-- master/
|   `-- scheduler.py
|-- rag/
|   `-- retriever.py
|-- workers/
|   `-- gpu_worker.py
|-- main.py
|-- README.md
`-- requirements.txt
```

Directory responsibilities:

- `client`: generates synthetic incoming requests
- `common`: shared data structures and interfaces
- `lb`: request routing and load-balancing strategies
- `llm`: response generation layer
- `master`: orchestration and scheduling logic
- `rag`: retrieval and context enrichment
- `workers`: worker-node execution logic
- `main.py`: simple entrypoint for the current dry-run scenario

## Setup Instructions

Use Python 3.9 or newer.

### Create a Virtual Environment

```bash
python -m venv .venv
```

### Activate the Environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```bat
.venv\Scripts\activate.bat
```

Linux or macOS:

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

The current scaffold does not require external packages beyond the Python standard library, so `requirements.txt` is intentionally minimal.

## Running the Project

From the project root:

```bash
python main.py
```

The current run is a dry run, not a full distributed simulation. It should:

- generate one synthetic request
- route it to one worker
- retrieve placeholder context
- generate a stub response
- print a summary to the console

Example output shape:

```text
[client] Generating 1 synthetic request(s)
[lb] Routed req-0001 to gpu-worker-1 using round-robin scheduling
[master] Scheduling req-0001 on worker gpu-worker-1
[rag] Retrieved placeholder context for req-0001
[worker:gpu-worker-1] Starting req-0001 on NVIDIA-A100-SIM
[llm] Generating stub response for req-0001
[worker:gpu-worker-1] Finished req-0001
[master] Completed req-0001
```

## Design Decisions in This Bootstrap

This initial version intentionally favors simplicity over realism so the repository stays easy to extend:

- deterministic flow instead of random behavior
- single-process execution instead of distributed networking
- stable stub outputs instead of external service dependencies
- explicit module boundaries so future refactors stay localized

These choices make it easier to validate architecture first, then add concurrency, fault tolerance, and scaling behavior in controlled steps.

## Suggested Next Milestones

The most useful next implementation steps are:

1. add configuration for worker count, request volume, and strategy selection
2. introduce concurrent request handling with `asyncio`
3. add more balancing strategies such as least-connections
4. simulate worker load, latency, and capacity differences
5. track metrics such as throughput and per-worker utilization
6. simulate failures and implement reassignment logic
7. add repeatable load tests for 100, 500, and 1000+ requests

## Testing and Evaluation Plan

The assignment brief emphasizes performance and fault-tolerance evaluation. As the project grows, testing should cover:

- functional correctness of routing and scheduling
- end-to-end request completion
- system behavior under increasing concurrency
- worker failure scenarios and recovery behavior
- fairness of task distribution
- response latency and throughput under load

Useful quantitative metrics include:

- average latency
- p95 latency
- throughput in requests per second
- worker utilization
- request success rate
- recovery time after simulated node failure

## Deliverables This Repository Should Support

Beyond code, the course project also requires broader project artifacts. This repository should eventually support:

- the implementation itself
- testing results and performance analysis
- architecture and design documentation
- report material for methodology, limitations, and references
- presentation/demo preparation

## Notes

- The repository keeps the existing published root commit and builds on top of it.
- The current implementation is a scaffold, not the final distributed system.
- The code is organized so future work can replace stubs gradually without changing the high-level structure.
