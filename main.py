from __future__ import annotations

from client import ClientLoadGenerator
from lb import RoundRobinLoadBalancer
from llm import LLMInferenceEngine
from master import MasterScheduler
from rag import RAGRetriever
from workers import GPUWorkerNode


def build_workers() -> list[GPUWorkerNode]:
    return [
        GPUWorkerNode(worker_id="gpu-worker-1", gpu_name="NVIDIA-A100-SIM"),
        GPUWorkerNode(worker_id="gpu-worker-2", gpu_name="NVIDIA-A100-SIM"),
        GPUWorkerNode(worker_id="gpu-worker-3", gpu_name="NVIDIA-A100-SIM"),
    ]


def main() -> None:
    workers = build_workers()
    generator = ClientLoadGenerator()
    load_balancer = RoundRobinLoadBalancer(workers)
    # use_stub=True keeps the dry-run sub-second by skipping the FAISS model download.
    # Scripts that exercise real vector retrieval should instantiate RAGRetriever()
    # with the default arguments.
    retriever = RAGRetriever(use_stub=True)
    inference_engine = LLMInferenceEngine()
    scheduler = MasterScheduler(retriever, inference_engine)

    request = generator.generate_requests(count=1)[0]
    selected_worker = load_balancer.select_worker(request)
    response = scheduler.handle_request(request, selected_worker)

    print("\n=== Dry Run Summary ===")
    print(f"Request ID : {response.request_id}")
    print(f"Worker ID  : {response.worker_id}")
    print(f"Status     : {response.status}")
    print(f"Context    : {response.context}")
    print(f"Answer     : {response.answer}")


if __name__ == "__main__":
    main()
