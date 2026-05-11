from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str
    title: str
    text: str


DEFAULT_CORPUS: tuple[Document, ...] = (
    Document(
        doc_id="doc-001",
        title="Round-robin load balancing",
        text=(
            "Round-robin load balancing distributes each incoming request to the next "
            "worker in a fixed rotation. It is simple, stateless, and fair when workers "
            "have similar capacity. It can be unfair when workers differ in speed or "
            "when request sizes vary, because slow workers accumulate queue depth."
        ),
    ),
    Document(
        doc_id="doc-002",
        title="Least-connections scheduling",
        text=(
            "Least-connections routing picks the worker with the fewest in-flight "
            "requests. It is well suited to heterogeneous workloads where request "
            "latency varies, because it naturally steers traffic away from workers "
            "that are already saturated."
        ),
    ),
    Document(
        doc_id="doc-003",
        title="Load-aware routing",
        text=(
            "Load-aware routing generalises least-connections by combining active "
            "task count, recent latency, GPU utilisation, and queue depth into a "
            "composite score. It improves tail latency under mixed traffic but "
            "requires the load balancer to observe worker-side metrics."
        ),
    ),
    Document(
        doc_id="doc-004",
        title="GPU cluster task distribution",
        text=(
            "A GPU cluster assigns inference tasks to nodes with free accelerator "
            "capacity. Good distribution minimises idle time on expensive GPUs, "
            "keeps memory fragmentation low, and supports parallel execution by "
            "pinning batches to specific devices."
        ),
    ),
    Document(
        doc_id="doc-005",
        title="LLM inference concurrency",
        text=(
            "High-concurrency LLM serving relies on continuous batching, paged "
            "attention, and asynchronous dispatch. Throughput scales with how many "
            "decode steps a single GPU can interleave across independent requests "
            "without stalling on memory transfers."
        ),
    ),
    Document(
        doc_id="doc-006",
        title="Retrieval-augmented generation",
        text=(
            "Retrieval-augmented generation (RAG) fetches relevant documents from "
            "a knowledge base and concatenates them with the user prompt before "
            "calling the language model. This grounds answers in external data, "
            "reduces hallucinations, and lets the system update without retraining."
        ),
    ),
    Document(
        doc_id="doc-007",
        title="Vector databases and embeddings",
        text=(
            "Vector databases index dense embeddings produced by a sentence encoder. "
            "Similarity search uses cosine distance or inner product to find the "
            "documents that are semantically closest to the query. FAISS is a common "
            "in-process backend for small to medium corpora."
        ),
    ),
    Document(
        doc_id="doc-008",
        title="Fault tolerance in distributed services",
        text=(
            "Fault tolerance keeps a distributed service responsive when nodes fail. "
            "Common techniques include health checks, heartbeats, automatic retry, "
            "task reassignment, and partial-failure degradation modes that keep "
            "healthy nodes serving traffic while failed ones recover."
        ),
    ),
    Document(
        doc_id="doc-009",
        title="Worker node failure detection",
        text=(
            "Worker failure detection uses periodic heartbeats, synthetic probes, "
            "or in-band error signalling to mark a node unavailable. Once a node is "
            "marked failed, the scheduler stops sending it work and reassigns any "
            "outstanding tasks to healthy peers."
        ),
    ),
    Document(
        doc_id="doc-010",
        title="Task reassignment and retries",
        text=(
            "Reassignment replays a failed task on a different worker so that no "
            "request is lost. Idempotent handlers, bounded retry counts, and "
            "exponential backoff prevent retry storms from overloading the cluster "
            "when many workers fail at once."
        ),
    ),
    Document(
        doc_id="doc-011",
        title="Horizontal scaling of inference",
        text=(
            "Horizontal scaling adds more GPU workers to serve additional traffic. "
            "It depends on a stateless request path, a scalable router, and a "
            "knowledge base that can be read concurrently. Each worker should be "
            "interchangeable so the load balancer can freely reshape traffic."
        ),
    ),
    Document(
        doc_id="doc-012",
        title="Latency and throughput metrics",
        text=(
            "Latency measures how long a single request takes end to end. Throughput "
            "measures how many requests complete per second. Reporting p50, p95, and "
            "p99 latency alongside throughput gives a fairer picture than averages "
            "because LLM traffic is often heavy tailed."
        ),
    ),
    Document(
        doc_id="doc-013",
        title="GPU utilisation monitoring",
        text=(
            "GPU utilisation tracks how much of each accelerator's time is spent "
            "doing useful compute. Low utilisation with high queue depth usually "
            "signals a bottleneck elsewhere in the pipeline, such as tokenisation, "
            "retrieval, or network transfer."
        ),
    ),
    Document(
        doc_id="doc-014",
        title="Concurrency with threads",
        text=(
            "Python threads share memory and are a good fit for I/O-bound or "
            "blocking workloads such as simulated GPU calls. Shared counters must "
            "be protected by a lock to avoid torn reads, even though individual "
            "integer reads are atomic in CPython."
        ),
    ),
    Document(
        doc_id="doc-015",
        title="Simulating GPU inference workloads",
        text=(
            "A simulation models GPU inference without an accelerator by sleeping "
            "for a duration that depends on prompt length, context size, and a "
            "little random jitter. This lets a distributed system be exercised end "
            "to end on a laptop before being connected to real hardware."
        ),
    ),
    Document(
        doc_id="doc-016",
        title="Health states of a worker",
        text=(
            "A worker can report healthy, degraded, or failed. Healthy workers "
            "accept new work. Degraded workers still accept work but are over "
            "capacity, so the scheduler should prefer other nodes when possible. "
            "Failed workers reject new work until an operator restores them."
        ),
    ),
    Document(
        doc_id="doc-017",
        title="Transformer architecture",
        text=(
            "The transformer architecture uses self-attention to process input "
            "sequences in parallel rather than sequentially. It consists of an "
            "encoder and decoder, each built from layers of multi-head attention "
            "and feed-forward networks. Transformers are the foundation of modern "
            "LLMs like GPT, LLaMA, and Qwen."
        ),
    ),
    Document(
        doc_id="doc-018",
        title="Attention mechanisms in neural networks",
        text=(
            "Attention allows a model to focus on relevant parts of the input when "
            "producing each output token. Scaled dot-product attention computes "
            "compatibility scores between queries and keys, then uses them to weight "
            "values. Multi-head attention runs several attention functions in parallel "
            "to capture different types of relationships."
        ),
    ),
    Document(
        doc_id="doc-019",
        title="Tokenization and vocabulary",
        text=(
            "Tokenization splits raw text into subword units that a language model "
            "can process. Byte-pair encoding (BPE) and SentencePiece are common "
            "algorithms. A larger vocabulary reduces sequence length but increases "
            "embedding table size. The tokenizer must match the one used during "
            "model training."
        ),
    ),
    Document(
        doc_id="doc-020",
        title="Model quantization for inference",
        text=(
            "Quantization reduces model weights from 32-bit floats to lower "
            "precision formats like float16, bfloat16, or int4. This cuts memory "
            "usage and speeds up inference with minimal accuracy loss. Bfloat16 "
            "preserves the dynamic range of float32 and is natively supported on "
            "NVIDIA Ampere and newer GPU architectures."
        ),
    ),
    Document(
        doc_id="doc-021",
        title="API rate limiting and throttling",
        text=(
            "Rate limiting restricts how many requests a client can make in a "
            "given time window to protect backend services from overload. Common "
            "algorithms include token bucket, sliding window, and fixed window "
            "counters. Rate limiting is essential for public-facing LLM APIs to "
            "prevent abuse and ensure fair access."
        ),
    ),
    Document(
        doc_id="doc-022",
        title="Microservices architecture",
        text=(
            "Microservices decompose an application into small, independently "
            "deployable services that communicate over HTTP or message queues. "
            "Each service owns its data and can be scaled, updated, or restarted "
            "without affecting others. This pattern suits distributed AI systems "
            "where inference, retrieval, and routing are separate concerns."
        ),
    ),
    Document(
        doc_id="doc-023",
        title="Containerization with Docker",
        text=(
            "Docker packages applications and their dependencies into containers "
            "that run consistently across environments. Docker Compose orchestrates "
            "multi-container deployments, linking services like workers, load "
            "balancers, and databases. GPU containers require the NVIDIA Container "
            "Toolkit for CUDA access."
        ),
    ),
    Document(
        doc_id="doc-024",
        title="CUDA programming and GPU memory",
        text=(
            "CUDA is NVIDIA's parallel computing platform for running code on GPU "
            "hardware. GPU memory (VRAM) is limited and shared across all running "
            "processes. Model weights, KV-cache for active requests, and activation "
            "tensors all compete for VRAM. Out-of-memory errors crash the process "
            "and require careful capacity planning."
        ),
    ),
    Document(
        doc_id="doc-025",
        title="Neural network training and fine-tuning",
        text=(
            "Training a neural network adjusts its weights by computing gradients "
            "of a loss function via backpropagation. Fine-tuning adapts a pre-trained "
            "model to a specific task using a smaller, domain-specific dataset. "
            "Techniques like LoRA reduce the number of trainable parameters, making "
            "fine-tuning feasible on consumer GPUs."
        ),
    ),
    Document(
        doc_id="doc-026",
        title="Data pipelines and ETL",
        text=(
            "Data pipelines automate the extraction, transformation, and loading "
            "of data from source systems into analytics or ML platforms. Reliable "
            "pipelines use idempotent operations, schema validation, and dead-letter "
            "queues to handle failures gracefully without data loss."
        ),
    ),
    Document(
        doc_id="doc-027",
        title="Caching strategies for web services",
        text=(
            "Caching stores frequently accessed data closer to the consumer to "
            "reduce latency and backend load. Strategies include write-through, "
            "write-behind, and cache-aside patterns. For LLM services, caching "
            "embeddings or repeated query results can significantly reduce GPU "
            "compute costs."
        ),
    ),
    Document(
        doc_id="doc-028",
        title="Observability and distributed tracing",
        text=(
            "Observability combines metrics, logs, and traces to understand system "
            "behaviour. Prometheus collects time-series metrics, Grafana visualises "
            "dashboards, and distributed tracing tools like Jaeger track requests "
            "across service boundaries. Good observability is critical for debugging "
            "performance issues in multi-node GPU clusters."
        ),
    ),
    Document(
        doc_id="doc-029",
        title="Server-Sent Events (SSE) for LLM Streaming",
        text=(
            "Server-Sent Events (SSE) is a lightweight HTTP standard for pushing "
            "events from the server to the client. It is widely used in LLM chat "
            "interfaces to stream tokens as they are generated by the model, "
            "providing a highly responsive user experience compared to waiting "
            "for the full text to complete."
        ),
    ),
    Document(
        doc_id="doc-030",
        title="FAISS Performance Characteristics",
        text=(
            "FAISS (Facebook AI Similarity Search) is highly optimized for dense "
            "vector operations. For small to medium corpora (up to a few million "
            "vectors), an exact inner-product search (IndexFlatIP) takes only "
            "microseconds. Adding a few dozen or hundred documents to the index "
            "has an imperceptible impact on retrieval latency."
        ),
    ),
    Document(
        doc_id="doc-031",
        title="Kubernetes Orchestration",
        text=(
            "Kubernetes is an open-source container orchestration system that automates "
            "software deployment, scaling, and management. It groups containers that "
            "make up an application into logical units for easy management and discovery, "
            "handling health checks, rolling updates, and dynamic scaling."
        ),
    ),
    Document(
        doc_id="doc-032",
        title="Asynchronous Programming with Asyncio",
        text=(
            "Python's asyncio library provides a foundation for writing concurrent "
            "code using the async/await syntax. By yielding control back to an event "
            "loop during I/O operations, a single thread can handle thousands of "
            "concurrent network connections efficiently without OS-level thread overhead."
        ),
    ),
    Document(
        doc_id="doc-033",
        title="Graph Databases",
        text=(
            "Graph databases like Neo4j represent data as nodes and edges rather than "
            "relational tables. They excel at querying highly interconnected data, "
            "such as social networks or recommendation systems, where traversing "
            "deep relationships in SQL would require prohibitively slow JOIN operations."
        ),
    ),
    Document(
        doc_id="doc-034",
        title="Redis Caching Patterns",
        text=(
            "Redis is an in-memory data structure store often used as a distributed "
            "cache. Common patterns include 'Cache-Aside' where the application checks "
            "the cache before querying the database, and 'Write-Through' where data is "
            "written to both the cache and database simultaneously to prevent stale reads."
        ),
    ),
    Document(
        doc_id="doc-035",
        title="Event-Driven Architecture with Kafka",
        text=(
            "Apache Kafka is a distributed event streaming platform. In an event-driven "
            "architecture, services communicate asynchronously by publishing and "
            "subscribing to streams of events. This decouples microservices, allowing "
            "systems to absorb massive traffic spikes by buffering events in topics."
        ),
    ),
    Document(
        doc_id="doc-036",
        title="Zero Trust Security",
        text=(
            "Zero Trust is a cybersecurity paradigm that discards the traditional 'castle "
            "and moat' approach. It operates on the principle of 'never trust, always "
            "verify,' requiring strict identity verification and least-privilege access "
            "for every person and device, regardless of whether they are on the VPN."
        ),
    ),
    Document(
        doc_id="doc-037",
        title="Multi-Head Attention Mechanism",
        text=(
            "In Transformer models, the Multi-Head Attention mechanism allows the model "
            "to jointly attend to information from different representation subspaces "
            "at different positions. By computing attention multiple times in parallel "
            "and concatenating the results, the model captures complex contextual relationships."
        ),
    ),
    Document(
        doc_id="doc-038",
        title="Gradient Descent Optimizers",
        text=(
            "Optimizers update neural network weights to minimize loss. While basic "
            "Stochastic Gradient Descent (SGD) takes steps proportional to the gradient, "
            "modern optimizers like Adam combine momentum (moving average of past gradients) "
            "and RMSprop (adaptive learning rates) to converge faster and more reliably."
        ),
    ),
    Document(
        doc_id="doc-039",
        title="JSON Web Tokens (JWT)",
        text=(
            "JWT is a compact, URL-safe means of representing claims between two parties. "
            "A token consists of a header, payload, and signature. Because the server "
            "can verify the signature cryptographically without querying a database, JWTs "
            "are highly scalable for stateless API authentication."
        ),
    ),
    Document(
        doc_id="doc-040",
        title="gRPC and Protocol Buffers",
        text=(
            "gRPC is a high-performance RPC framework that uses HTTP/2 for transport and "
            "Protocol Buffers (Protobufs) as its interface description language. Protobufs "
            "serialize structured data into a dense binary format, making gRPC significantly "
            "faster and more bandwidth-efficient than JSON-over-HTTP."
        ),
    ),
    Document(
        doc_id="doc-041",
        title="Serverless Computing",
        text=(
            "Serverless computing allows developers to build and run applications without "
            "managing infrastructure. Platforms like AWS Lambda automatically provision "
            "resources and scale precisely with the workload. Users pay only for the exact "
            "compute time consumed down to the millisecond."
        ),
    ),
    Document(
        doc_id="doc-042",
        title="Consistent Hashing",
        text=(
            "Consistent hashing is a distributed routing technique that minimizes the "
            "number of keys that need to be remapped when a node is added or removed "
            "from a cluster. It maps both data keys and server nodes onto a logical "
            "ring, which is foundational for systems like Cassandra and DynamoDB."
        ),
    ),
    Document(
        doc_id="doc-043",
        title="The CAP Theorem",
        text=(
            "The CAP Theorem states that a distributed data store can only guarantee "
            "two out of three properties simultaneously: Consistency, Availability, "
            "and Partition Tolerance. Since network partitions are inevitable in real "
            "networks, distributed systems must generally trade off between C and A."
        ),
    ),
    Document(
        doc_id="doc-044",
        title="B-Tree Indexes in SQL",
        text=(
            "Relational databases use B-Tree (balanced tree) structures for indexing "
            "columns. B-Trees keep data sorted and allow searches, sequential access, "
            "insertions, and deletions in logarithmic time. They are optimized for "
            "systems that read and write large blocks of data."
        ),
    ),
    Document(
        doc_id="doc-045",
        title="React Virtual DOM",
        text=(
            "React improves UI performance using a Virtual DOM—an in-memory representation "
            "of the actual browser DOM. When component state changes, React computes a "
            "diff against the previous Virtual DOM and applies only the minimal set of "
            "changes required to update the real DOM, avoiding expensive layout thrashing."
        ),
    ),
    Document(
        doc_id="doc-046",
        title="WebAssembly (Wasm)",
        text=(
            "WebAssembly is a binary instruction format designed as a portable compilation "
            "target for high-level languages like C, C++, and Rust. It enables deployment "
            "of high-performance applications on the web, running at near-native speed "
            "alongside standard JavaScript."
        ),
    ),
    Document(
        doc_id="doc-047",
        title="Chaos Engineering",
        text=(
            "Chaos Engineering is the discipline of experimenting on a system in order to "
            "build confidence in its capability to withstand turbulent conditions in "
            "production. By intentionally injecting failures like network latency or "
            "crashed nodes, teams can uncover hidden weaknesses before they cause outages."
        ),
    ),
    Document(
        doc_id="doc-048",
        title="WebSockets for Real-time Communication",
        text=(
            "The WebSocket API provides a persistent, full-duplex communication channel "
            "over a single TCP connection. Unlike HTTP polling, WebSockets allow servers "
            "to push data to the client instantly as events occur, making them ideal for "
            "chat applications, live feeds, and multiplayer games."
        ),
    ),
    Document(
        doc_id="doc-049",
        title="Continuous Integration and Continuous Deployment (CI/CD)",
        text=(
            "CI/CD automates the software release process. Continuous Integration merges "
            "code changes into a central repository, triggering automated builds and tests. "
            "Continuous Deployment automates the release of validated changes directly to "
            "production environments, enabling fast and safe feature delivery."
        ),
    ),
    Document(
        doc_id="doc-050",
        title="Parameter-Efficient Fine-Tuning (PEFT)",
        text=(
            "PEFT methods adapt large pre-trained language models to downstream applications "
            "without fine-tuning all model parameters. Techniques like Low-Rank Adaptation "
            "(LoRA) freeze the original weights and inject trainable rank decomposition "
            "matrices into each layer, drastically reducing memory and compute costs."
        ),
    ),
    Document(
        doc_id="doc-051",
        title="Project Overview: Distributed LLM Load Balancer",
        text=(
            "The current system is a 'Distributed LLM GPU Load Balancer'. It is a "
            "Python-based, containerized cluster designed to serve Large Language Models "
            "(like Qwen2.5-0.5B-Instruct) across multiple worker nodes. It demonstrates "
            "advanced load balancing strategies, RAG integration, and fault tolerance."
        ),
    ),
    Document(
        doc_id="doc-052",
        title="System Architecture and Request Flow",
        text=(
            "The cluster topology flows as follows: A client sends an HTTP request to Nginx "
            "(port 8080), which reverse-proxies to the Load Balancer Service (port 7000). "
            "The LB forwards to a Master Service (port 9000). The Master runs RAG, selects "
            "a worker via an inner LoadBalancer, and sends the request to a Worker Node (port 8000)."
        ),
    ),
    Document(
        doc_id="doc-053",
        title="Load Balancing Strategies",
        text=(
            "The system supports four load balancing strategies: 'round_robin' (strict rotation), "
            "'least_connections' (picks lowest active tasks), 'load_aware' (factors in active "
            "tasks, GPU utilization, and capacity limits), and 'power_of_two' (randomly samples "
            "two nodes and picks the less loaded one to reduce overhead in large clusters)."
        ),
    ),
    Document(
        doc_id="doc-054",
        title="Fault Tolerance and Circuit Breaker",
        text=(
            "The Master node uses an active health monitor that polls workers every second. "
            "If a worker misses 3 consecutive heartbeats (the '3-strike' rule), it is marked "
            "as FAILED and removed from routing. Additionally, if a request fails mid-flight, "
            "the MasterScheduler automatically retries it on a healthy worker."
        ),
    ),
    Document(
        doc_id="doc-055",
        title="Project File Structure",
        text=(
            "The codebase is divided into clear domains: 'deploy/' holds Dockerfiles and Nginx configs. "
            "'services/' holds FastAPI endpoints. 'master/' contains the scheduler and health monitor. "
            "'workers/' contains the GPU node proxy logic. 'lb/' implements routing strategies. "
            "'llm/' implements HuggingFace and simulated inference backends. 'rag/' manages FAISS."
        ),
    ),
    Document(
        doc_id="doc-056",
        title="Docker and Compose Infrastructure",
        text=(
            "The cluster runs entirely in Docker. A base Dockerfile provides CPU support, "
            "while Dockerfile.gpu uses the nvidia/cuda runtime for GPU acceleration. "
            "The docker-compose.yml files define the containers: Nginx, Load Balancer, "
            "Master, multiple Workers, Prometheus, and Grafana. A shared 'hf-cache' "
            "volume prevents downloading the HuggingFace model multiple times."
        ),
    ),
    Document(
        doc_id="doc-057",
        title="Nginx Reverse Proxy Configuration",
        text=(
            "Nginx acts as the single entrypoint on port 8080. It serves the static UI "
            "dashboard at the root path, and proxies API requests like '/request', "
            "'/workers', and '/admin' directly to the Load Balancer service. It uses "
            "the 'least_conn' directive to distribute traffic if multiple LBs exist."
        ),
    ),
    Document(
        doc_id="doc-058",
        title="Dashboard UI and Chat Client",
        text=(
            "The web dashboard (index.html) is a zero-terminal interface that visualizes "
            "cluster health and provides a chat window. When a user sends a chat message, "
            "the UI injects 'max_new_tokens: 512' into the request metadata to override "
            "the backend's default short limit, enabling long-form generation."
        ),
    ),
    Document(
        doc_id="doc-059",
        title="Observability: Prometheus and Grafana",
        text=(
            "Every FastAPI service exposes a '/metrics' endpoint. Prometheus scrapes "
            "these endpoints every 5 seconds to collect data on active tasks, pending "
            "tasks, and worker status. Grafana connects to Prometheus and displays "
            "this data on the 'cse354-overview' dashboard on port 3000."
        ),
    ),
    Document(
        doc_id="doc-060",
        title="Load Testing and Benchmark Scripts",
        text=(
            "The system includes 'scripts/benchmark.py', a locust-like testing harness "
            "that simulates high concurrency (e.g., 100 simultaneous users). It fires "
            "massive batches of requests through the LB to measure how well the routing "
            "strategies handle overwhelming traffic and to validate fault tolerance."
        ),
    ),
    Document(
        doc_id="doc-061",
        title="The Language Model: Qwen2.5-0.5B-Instruct",
        text=(
            "By default, the real GPU workers load 'Qwen/Qwen2.5-0.5B-Instruct' via "
            "the HuggingFace transformers pipeline. It is a highly capable 500-million "
            "parameter model that fits comfortably in consumer GPU VRAM (like an RTX 3060), "
            "leaving ample memory for the KV-cache during concurrent inference."
        ),
    ),
    Document(
        doc_id="doc-062",
        title="Worker Self-Shedding at Capacity",
        text=(
            "When a GPU worker's active tasks reach its 'max_concurrent_tasks' limit, "
            "it refuses new requests with a 503 status code and an 'at-capacity' header. "
            "This 'self-shedding' prevents the GPU queue from exploding and stalling "
            "the PCIe bus, forcing the Master to gracefully route the request elsewhere."
        ),
    ),
    Document(
        doc_id="doc-063",
        title="FastAPI Concurrency Model",
        text=(
            "The project uses synchronous FastAPI endpoints backed by an AnyIO threadpool "
            "expanded to 1000 tokens. This easily handles 1000 concurrent users without "
            "the complexity of 'async def'. The health monitor is the only pure asyncio "
            "background task, sharing the event loop with the web server."
        ),
    ),
    Document(
        doc_id="doc-064",
        title="RemoteWorkerProxy and Duck-Typing",
        text=(
            "Inside the Master Service, workers are represented by the 'RemoteWorkerProxy' "
            "class. It duck-types the local 'GPUWorkerNode' interface, holding an httpx "
            "client connection pool. This allows the MasterScheduler to orchestrate remote "
            "HTTP workers exactly as if they were local Python objects."
        ),
    ),
    Document(
        doc_id="doc-065",
        title="Heterogeneous Cluster Support",
        text=(
            "The cluster can run a mix of fast GPU nodes and slow CPU nodes simultaneously. "
            "In this heterogeneous mode, simple load balancers like 'round_robin' fail "
            "because they overload the slow CPU workers. The 'load_aware' strategy "
            "solves this by dividing queue depth by node capacity, steering most traffic to the GPUs."
        ),
    ),
)
