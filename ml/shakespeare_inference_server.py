"""Serve checkpoint-backed Shakespeare completion to the local teaching UI.

The server binds to loopback only. It exposes a health endpoint and one JSON
generation endpoint; it never trains or changes checkpoint files.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import mlx.core as mx

from shakespeare_transformer import (
    DEFAULT_DATA,
    DEFAULT_RUN_DIR,
    TinyTransformerLanguageModel,
    generate_from_prompt,
    load_data,
)


class InferenceRuntime:
    """Own the vocabulary and lazily loaded, read-only checkpoint models."""

    def __init__(self, data_path: Path, run_dir: Path):
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        _, _, self.vocabulary, self.char_to_id = load_data(data_path)
        self.run_dir = run_dir
        self.config = config
        self.models: dict[int, TinyTransformerLanguageModel] = {}
        self.lock = threading.Lock()
        self.steps = sorted(
            int(path.stem.split("-")[1])
            for path in run_dir.glob("checkpoint-*.safetensors")
        )

    def model_for(self, step: int) -> TinyTransformerLanguageModel:
        if step not in self.steps:
            raise ValueError(f"Unknown checkpoint {step}; choose one of {self.steps}")
        if step not in self.models:
            model = TinyTransformerLanguageModel(
                len(self.vocabulary),
                self.config["contextSize"],
                self.config["modelSize"],
                self.config["attentionHeads"],
                self.config["transformerBlocks"],
            )
            model.load_weights(str(self.run_dir / f"checkpoint-{step:04d}.safetensors"))
            mx.eval(model.parameters())
            self.models[step] = model
        return self.models[step]

    def generate(self, request: dict[str, Any]) -> dict[str, Any]:
        prompt = str(request.get("prompt", ""))
        step = int(request.get("checkpoint", 3_000))
        temperature = float(request.get("temperature", 0.8))
        characters = int(request.get("characters", 180))
        seed = int(request.get("seed", 42))
        if not prompt:
            raise ValueError("Enter at least one prompt character")
        if len(prompt) > 2_000:
            raise ValueError("Prompt must contain at most 2,000 characters")
        if not 0.1 <= temperature <= 2.0:
            raise ValueError("Temperature must be between 0.1 and 2.0")
        if not 1 <= characters <= 500:
            raise ValueError("Generated length must be between 1 and 500 characters")
        started_at = time.perf_counter()
        with self.lock:
            continuation = generate_from_prompt(
                self.model_for(step),
                prompt,
                self.char_to_id,
                self.vocabulary,
                seed=seed,
                characters=characters,
                temperature=temperature,
            )
        return {
            "prompt": prompt,
            "continuation": continuation,
            "checkpoint": step,
            "temperature": temperature,
            "seed": seed,
            "generatedCharacters": len(continuation),
            "contextCharactersUsed": min(len(prompt), self.config["contextSize"]),
            "elapsedSeconds": time.perf_counter() - started_at,
        }


def handler_for(runtime: InferenceRuntime) -> type[BaseHTTPRequestHandler]:
    class InferenceHandler(BaseHTTPRequestHandler):
        def end_headers(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "http://localhost:3000")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            super().end_headers()

        def json_response(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self.json_response(HTTPStatus.OK, {
                    "status": "ready",
                    "checkpoints": runtime.steps,
                    "contextSize": runtime.config["contextSize"],
                    "device": str(mx.default_device()),
                })
            else:
                self.json_response(HTTPStatus.NOT_FOUND, {"error": "Not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/generate":
                self.json_response(HTTPStatus.NOT_FOUND, {"error": "Not found"})
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length > 20_000:
                    raise ValueError("Request body is too large")
                request = json.loads(self.rfile.read(content_length))
                self.json_response(HTTPStatus.OK, runtime.generate(request))
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                self.json_response(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            except Exception as error:  # keep the teaching service alive and report safely
                self.json_response(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"Generation failed: {error}"})

        def log_message(self, format: str, *args: object) -> None:
            print(f"{self.address_string()} - {format % args}")

    return InferenceHandler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    runtime = InferenceRuntime(args.data, args.run_dir)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(runtime))
    print(f"Shakespeare inference ready at http://127.0.0.1:{args.port}", flush=True)
    print(f"Available checkpoints: {runtime.steps}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
