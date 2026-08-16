import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, host: str):
        self.host = host.rstrip("/")

    def check_connection(self) -> None:
        try:
            self._get("/api/tags")
        except OllamaError as exc:
            raise OllamaError(
                "Could not reach Ollama. Start Ollama and confirm it is running "
                f"at {self.host}."
            ) from exc

    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        if not texts:
            return []

        try:
            response = self._post("/api/embed", {"model": model, "input": texts})
            embeddings = response.get("embeddings")
            if embeddings:
                return embeddings
        except OllamaError:
            pass

        return [self._embed_one(text, model) for text in texts]

    def chat(self, prompt: str, model: str, temperature: float = 0.0) -> str:
        response = self._post(
            "/api/chat",
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": temperature},
            },
        )
        message = response.get("message", {})
        content = message.get("content")
        if not content:
            raise OllamaError("Ollama returned an empty chat response.")
        return content.strip()

    def _embed_one(self, text: str, model: str) -> list[float]:
        response = self._post("/api/embeddings", {"model": model, "prompt": text})
        embedding = response.get("embedding")
        if not embedding:
            raise OllamaError("Ollama returned an empty embedding.")
        return embedding

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.host}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=120) as response:
                body = response.read().decode("utf-8")
                return json.loads(body) if body else {}
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OllamaError(f"Ollama HTTP error {exc.code}: {detail}") from exc
        except URLError as exc:
            raise OllamaError(f"Ollama connection error: {exc.reason}") from exc
        except TimeoutError as exc:
            raise OllamaError("Ollama request timed out.") from exc

    def _get(self, path: str) -> dict[str, Any]:
        request = Request(f"{self.host}{path}", method="GET")
        try:
            with urlopen(request, timeout=30) as response:
                body = response.read().decode("utf-8")
                return json.loads(body) if body else {}
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OllamaError(f"Ollama HTTP error {exc.code}: {detail}") from exc
        except URLError as exc:
            raise OllamaError(f"Ollama connection error: {exc.reason}") from exc
        except TimeoutError as exc:
            raise OllamaError("Ollama request timed out.") from exc
