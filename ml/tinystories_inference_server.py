"""Serve evaluated TinyStories checkpoints to the local teaching playground.

The service binds to loopback only, verifies the frozen Lesson 10 evidence,
and performs inference without changing or writing model weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import mlx.core as mx
from tokenizers import Tokenizer

from paths import local_result, local_run
from shakespeare_transformer import TinyTransformerLanguageModel
from tinystories_random_baseline import random_completion


DEFAULT_EVIDENCE = local_result("tinystories-story-evaluation.json")
DEFAULT_FINAL_EVIDENCE = local_result("tinystories-final.json")
DEFAULT_TOKENIZER = local_run("tinystories-tokenizer-001") / "tokenizer.json"
DEFAULT_PORT = 8002
ALLOWED_ORIGINS = {"http://localhost:3000", "http://127.0.0.1:3000"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class InferenceRuntime:
    """Validate Lesson 10 and lazily cache read-only checkpoint models."""

    def __init__(self, evidence_path: Path, tokenizer_path: Path, final_evidence_path: Path):
        self.evidence_path = evidence_path
        self.tokenizer_path = tokenizer_path
        self.final_evidence_path = final_evidence_path
        self.lock = threading.Lock()
        self.models: dict[str, TinyTransformerLanguageModel] = {}
        self.error: str | None = None
        self.architecture: dict[str, Any] = {}
        self.candidates: dict[str, dict[str, Any]] = {}
        self.tokenizer: Tokenizer | None = None
        try:
            self._load_contract()
        except (OSError, KeyError, TypeError, ValueError, RuntimeError, json.JSONDecodeError) as error:
            self.error = str(error)

    def _load_contract(self) -> None:
        if not self.evidence_path.is_file():
            raise RuntimeError("Lesson 10 local evaluation evidence is missing; complete story evaluation first")
        if not self.tokenizer_path.is_file():
            raise RuntimeError("The learner-local TinyStories tokenizer is missing; complete Lesson 3 first")

        evidence = json.loads(self.evidence_path.read_text(encoding="utf-8"))
        if evidence.get("status") != "Complete":
            raise RuntimeError("Lesson 10 evaluation is not complete")
        protocol = evidence["protocol"]
        if file_sha256(self.tokenizer_path) != protocol["tokenizerSha256"]:
            raise RuntimeError("Tokenizer checksum differs from the evaluated Lesson 10 contract")

        architecture = protocol["architecture"]
        required = (
            "vocabularySize", "contextSize", "modelSize",
            "attentionHeads", "transformerBlocks", "parameterCount",
        )
        if any(not isinstance(architecture.get(key), int) for key in required):
            raise RuntimeError("Lesson 10 architecture metadata is incomplete")

        candidates: dict[str, dict[str, Any]] = {}
        for candidate in protocol["candidates"]:
            checkpoint = Path(candidate["checkpoint"])
            if not checkpoint.is_file():
                raise RuntimeError(f"Evaluated checkpoint is missing: {checkpoint}")
            if file_sha256(checkpoint) != candidate["checkpointSha256"]:
                raise RuntimeError(f"Evaluated checkpoint checksum failed: {checkpoint}")
            candidates[candidate["id"]] = {**candidate, "checkpoint": checkpoint}

        if set(candidates) != {"random", "inherited-700", "clean-700"}:
            raise RuntimeError("Lesson 10 must identify random, inherited-700, and clean-700 checkpoints")
        if self.final_evidence_path.is_file():
            final_evidence = json.loads(self.final_evidence_path.read_text(encoding="utf-8"))
            if final_evidence.get("status") == "Complete":
                selected = final_evidence["selectedCheckpoint"]
                checkpoint = Path(selected["path"])
                if not checkpoint.is_file() or file_sha256(checkpoint) != selected["sha256"]:
                    raise RuntimeError("Lesson 12 selected checkpoint checksum failed")
                selected_seed = final_evidence["selectedSeed"]
                candidates["final-selected"] = {
                    "id": "final-selected", "label": f"Final selected · seed {selected_seed} · step 700",
                    "history": f"Lesson 12 lowest observed complete validation loss across seeds 42–44; training seed {selected_seed}",
                    "checkpoint": checkpoint, "checkpointSha256": selected["sha256"],
                }
        self.architecture = architecture
        self.candidates = candidates
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))

    @property
    def ready(self) -> bool:
        return self.error is None and self.tokenizer is not None and bool(self.candidates)

    def public_candidates(self) -> list[dict[str, Any]]:
        return [
            {
                "id": candidate["id"],
                "label": candidate["label"],
                "history": candidate["history"],
                "checkpointSha256": candidate["checkpointSha256"],
            }
            for candidate in self.candidates.values()
        ]

    def model_for(self, candidate_id: str) -> TinyTransformerLanguageModel:
        if candidate_id not in self.candidates:
            raise ValueError(f"Unknown checkpoint {candidate_id!r}")
        if candidate_id not in self.models:
            candidate = self.candidates[candidate_id]
            model = TinyTransformerLanguageModel(
                self.architecture["vocabularySize"],
                self.architecture["contextSize"],
                self.architecture["modelSize"],
                self.architecture["attentionHeads"],
                self.architecture["transformerBlocks"],
            )
            model.load_weights(str(candidate["checkpoint"]), strict=True)
            mx.eval(model.parameters())
            model.eval()
            self.models[candidate_id] = model
        return self.models[candidate_id]

    def generate(self, request: dict[str, Any]) -> dict[str, Any]:
        if not self.ready or self.tokenizer is None:
            raise RuntimeError(self.error or "Inference runtime is not ready")
        prompt = request.get("prompt")
        checkpoint = request.get("checkpoint", "clean-700")
        temperature = request.get("temperature", 0.8)
        maximum_tokens = request.get("maximumTokens", 64)
        seed = request.get("seed", 42)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Enter a non-empty story opening")
        if len(prompt) > 2_000:
            raise ValueError("Prompt must contain at most 2,000 characters")
        if not isinstance(checkpoint, str) or checkpoint not in self.candidates:
            raise ValueError(f"Checkpoint must be one of {sorted(self.candidates)}")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0.1 <= float(temperature) <= 1.5:
            raise ValueError("Temperature must be between 0.1 and 1.5")
        if isinstance(maximum_tokens, bool) or not isinstance(maximum_tokens, int) or not 1 <= maximum_tokens <= 160:
            raise ValueError("Maximum tokens must be an integer between 1 and 160")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2_147_483_647:
            raise ValueError("Seed must be an integer between 0 and 2,147,483,647")

        prompt_ids = self.tokenizer.encode(prompt).ids
        started_at = time.perf_counter()
        with self.lock:
            generated = random_completion(
                self.model_for(checkpoint), self.tokenizer, prompt,
                seed=seed, maximum_tokens=maximum_tokens, temperature=float(temperature),
            )
        return {
            **generated,
            "checkpoint": checkpoint,
            "checkpointLabel": self.candidates[checkpoint]["label"],
            "promptTokens": len(prompt_ids),
            "contextTokensUsed": min(len(prompt_ids), self.architecture["contextSize"]),
            "contextSize": self.architecture["contextSize"],
            "elapsedSeconds": time.perf_counter() - started_at,
            "weightsUpdated": False,
        }


def handler_for(runtime: InferenceRuntime) -> type[BaseHTTPRequestHandler]:
    class InferenceHandler(BaseHTTPRequestHandler):
        def end_headers(self) -> None:
            origin = self.headers.get("Origin")
            if origin in ALLOWED_ORIGINS:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
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
            if self.path != "/health":
                self.json_response(HTTPStatus.NOT_FOUND, {"error": "Not found"})
                return
            self.json_response(HTTPStatus.OK, {
                "status": "ready" if runtime.ready else "not-ready",
                "error": runtime.error,
                "checkpoints": runtime.public_candidates(),
                "architecture": runtime.architecture,
                "device": str(mx.default_device()),
            })

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/generate":
                self.json_response(HTTPStatus.NOT_FOUND, {"error": "Not found"})
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if not 1 <= content_length <= 20_000:
                    raise ValueError("Request body must contain between 1 and 20,000 bytes")
                request = json.loads(self.rfile.read(content_length))
                if not isinstance(request, dict):
                    raise ValueError("Request body must be a JSON object")
                self.json_response(HTTPStatus.OK, runtime.generate(request))
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                self.json_response(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            except RuntimeError as error:
                self.json_response(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(error)})
            except Exception as error:  # Keep the local teaching process alive.
                self.json_response(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"Generation failed: {error}"})

        def log_message(self, format: str, *args: object) -> None:
            print(f"{self.address_string()} - {format % args}", flush=True)

    return InferenceHandler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--final-evidence", type=Path, default=DEFAULT_FINAL_EVIDENCE)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    runtime = InferenceRuntime(args.evidence, args.tokenizer, args.final_evidence)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(runtime))
    print(f"TinyStories inference ready at http://127.0.0.1:{args.port}", flush=True)
    print(f"Runtime status: {'ready' if runtime.ready else runtime.error}", flush=True)
    print(f"Available checkpoints: {list(runtime.candidates)}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
