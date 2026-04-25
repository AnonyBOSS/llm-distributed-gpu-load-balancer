from __future__ import annotations

from common import Request


class ClientLoadGenerator:
    def generate_requests(self, count: int = 1) -> list[Request]:
        print(f"[client] Generating {count} synthetic request(s)")
        requests: list[Request] = []

        for index in range(count):
            request_number = index + 1
            requests.append(
                Request(
                    request_id=f"req-{request_number:04d}",
                    user_id=f"user-{request_number:04d}",
                    prompt=(
                        "Summarize how the distributed LLM cluster handles "
                        f"request {request_number}."
                    ),
                    metadata={"priority": "normal", "source": "load-generator"},
                )
            )

        return requests
