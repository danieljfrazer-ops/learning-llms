#!/usr/bin/env python3
"""Export reviewed Shakespeare checkpoints to ONNX and freeze MLX parity fixtures.

This maintainer command reads the verified corpus plus existing safetensors
checkpoints. It does not train or alter reference evidence. It writes compact
browser models, a provenance manifest, and fixed MLX outputs under
``public/models/shakespeare``.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))

from shakespeare_transformer import TinyTransformerLanguageModel, load_data  # noqa: E402


DATA = ROOT / "data" / "raw" / "tiny-shakespeare.txt"
OUTPUT = ROOT / "public" / "models" / "shakespeare"
OPSET = 18
PARITY_PROMPTS = ["\n", "To be, or not to be", "ROMEO:\n"]
GREEDY_CHARACTERS = 24

EXPORTS = [
    {
        "id": "random",
        "label": "Random weights · step 0",
        "runId": "shakespeare-transformer-001",
        "step": 0,
        "checkpoint": ROOT / "experiments/shakespeare-transformer-001/checkpoint-0000.safetensors",
        "config": ROOT / "experiments/shakespeare-transformer-001/config.json",
    },
    {
        "id": "minimal",
        "label": "Minimally trained · step 1",
        "runId": "shakespeare-transformer-001",
        "step": 1,
        "checkpoint": ROOT / "experiments/shakespeare-transformer-001/checkpoint-0001.safetensors",
        "config": ROOT / "experiments/shakespeare-transformer-001/config.json",
    },
    {
        "id": "baseline",
        "label": "Baseline 112K transformer · step 3,000",
        "runId": "shakespeare-transformer-001",
        "step": 3000,
        "checkpoint": ROOT / "experiments/shakespeare-transformer-001/checkpoint-3000.safetensors",
        "config": ROOT / "experiments/shakespeare-transformer-001/config.json",
    },
    {
        "id": "final",
        "label": "Final 420K transformer · seed 43 · step 3,000",
        "runId": "shakespeare-final-seed-043",
        "step": 3000,
        "checkpoint": ROOT / "experiments/shakespeare-final-seed-043/checkpoint-3000.safetensors",
        "config": ROOT / "experiments/shakespeare-final-seed-043/config.json",
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


class GraphBuilder:
    def __init__(self, weights: dict[str, np.ndarray], model_id: str):
        self.weights = weights
        self.model_id = model_id
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []
        self.parameter_manifest: list[dict[str, Any]] = []
        self.counter = 0

    def unique(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}_{self.counter}"

    def constant(self, name: str, value: np.ndarray) -> str:
        value = np.asarray(value)
        self.initializers.append(numpy_helper.from_array(value, name=name))
        return name

    def parameter(self, source_name: str, *, transpose: bool = False) -> str:
        source = np.asarray(self.weights[source_name], dtype=np.float32)
        exported = np.ascontiguousarray(source.T if transpose else source)
        exported_name = source_name + (".transpose" if transpose else "")
        self.initializers.append(numpy_helper.from_array(exported, name=exported_name))
        self.parameter_manifest.append(
            {
                "sourceName": source_name,
                "sourceShape": list(source.shape),
                "sourceSha256": array_sha256(source),
                "exportedName": exported_name,
                "exportedShape": list(exported.shape),
                "exportedSha256": array_sha256(exported),
                "transform": "transpose(1,0)" if transpose else "identity",
            }
        )
        return exported_name

    def op(self, op_type: str, inputs: list[str], *, prefix: str, **attributes: Any) -> str:
        output = self.unique(prefix)
        self.nodes.append(helper.make_node(op_type, inputs, [output], **attributes))
        return output

    def linear(self, value: str, prefix: str, weight_name: str, bias_name: str | None = None) -> str:
        result = self.op("MatMul", [value, self.parameter(weight_name, transpose=True)], prefix=f"{prefix}_matmul")
        if bias_name:
            result = self.op("Add", [result, self.parameter(bias_name)], prefix=f"{prefix}_bias")
        return result

    def layer_norm(self, value: str, prefix: str, weight_name: str, bias_name: str) -> str:
        return self.op(
            "LayerNormalization",
            [value, self.parameter(weight_name), self.parameter(bias_name)],
            prefix=prefix,
            axis=-1,
            epsilon=1e-5,
        )

    def gelu(self, value: str, prefix: str) -> str:
        root_two = self.constant(self.unique("sqrt_two"), np.array(math.sqrt(2), dtype=np.float32))
        one = self.constant(self.unique("one"), np.array(1, dtype=np.float32))
        half = self.constant(self.unique("half"), np.array(0.5, dtype=np.float32))
        divided = self.op("Div", [value, root_two], prefix=f"{prefix}_divide")
        erf = self.op("Erf", [divided], prefix=f"{prefix}_erf")
        shifted = self.op("Add", [erf, one], prefix=f"{prefix}_shift")
        scaled = self.op("Mul", [value, shifted], prefix=f"{prefix}_scale")
        return self.op("Mul", [scaled, half], prefix=f"{prefix}_half")

    def attention(self, value: str, block: int, model_size: int, heads: int, sequence_size: str) -> str:
        prefix = f"blocks.layers.{block}.attention"
        head_size = model_size // heads
        qkv = self.linear(value, f"block{block}_qkv", f"{prefix}.query_key_value.weight")
        queries, keys, values = [self.unique(f"block{block}_{part}") for part in ("queries", "keys", "values")]
        self.nodes.append(helper.make_node("Split", [qkv], [queries, keys, values], axis=-1, num_outputs=3))
        head_shape = self.constant(
            self.unique(f"block{block}_head_shape"),
            np.array([1, -1, heads, head_size], dtype=np.int64),
        )

        def split_heads(item: str, name: str) -> str:
            reshaped = self.op("Reshape", [item, head_shape], prefix=f"block{block}_{name}_reshape")
            return self.op("Transpose", [reshaped], prefix=f"block{block}_{name}_transpose", perm=[0, 2, 1, 3])

        queries = split_heads(queries, "queries")
        keys = split_heads(keys, "keys")
        values = split_heads(values, "values")
        keys_t = self.op("Transpose", [keys], prefix=f"block{block}_keys_t", perm=[0, 1, 3, 2])
        scores = self.op("MatMul", [queries, keys_t], prefix=f"block{block}_scores")
        scale = self.constant(self.unique("attention_scale"), np.array(math.sqrt(head_size), dtype=np.float32))
        scores = self.op("Div", [scores, scale], prefix=f"block{block}_scaled_scores")

        zero_i64 = self.constant(self.unique("zero_i64"), np.array(0, dtype=np.int64))
        one_i64 = self.constant(self.unique("one_i64"), np.array(1, dtype=np.int64))
        positions = self.op("Range", [zero_i64, sequence_size, one_i64], prefix=f"block{block}_positions")
        axis_zero = self.constant(self.unique("axis_zero"), np.array([0], dtype=np.int64))
        axis_one = self.constant(self.unique("axis_one"), np.array([1], dtype=np.int64))
        rows = self.op("Unsqueeze", [positions, axis_one], prefix=f"block{block}_rows")
        columns = self.op("Unsqueeze", [positions, axis_zero], prefix=f"block{block}_columns")
        future = self.op("Greater", [columns, rows], prefix=f"block{block}_future")
        negative = self.constant(self.unique("negative_mask"), np.array(-1e9, dtype=np.float32))
        zero_f32 = self.constant(self.unique("zero_f32"), np.array(0, dtype=np.float32))
        mask = self.op("Where", [future, negative, zero_f32], prefix=f"block{block}_mask")
        scores = self.op("Add", [scores, mask], prefix=f"block{block}_masked_scores")
        probabilities = self.op("Softmax", [scores], prefix=f"block{block}_softmax", axis=-1)
        attended = self.op("MatMul", [probabilities, values], prefix=f"block{block}_attended")
        combined = self.op("Transpose", [attended], prefix=f"block{block}_combine_t", perm=[0, 2, 1, 3])
        combined_shape = self.constant(
            self.unique(f"block{block}_combined_shape"), np.array([1, -1, model_size], dtype=np.int64)
        )
        combined = self.op("Reshape", [combined, combined_shape], prefix=f"block{block}_combine")
        return self.linear(combined, f"block{block}_projection", f"{prefix}.projection.weight")


def export_onnx(weights: dict[str, np.ndarray], model_id: str, model_size: int, heads: int, blocks: int) -> tuple[onnx.ModelProto, list[dict[str, Any]]]:
    builder = GraphBuilder(weights, model_id)
    token_ids = "token_ids"
    shape = builder.op("Shape", [token_ids], prefix="input_shape")
    sequence_axis = builder.constant("sequence_axis", np.array(1, dtype=np.int64))
    sequence_size = builder.op("Gather", [shape, sequence_axis], prefix="sequence_size", axis=0)
    zero_i64 = builder.constant("position_start", np.array(0, dtype=np.int64))
    one_i64 = builder.constant("position_step", np.array(1, dtype=np.int64))
    positions = builder.op("Range", [zero_i64, sequence_size, one_i64], prefix="positions")
    tokens = builder.op("Gather", [builder.parameter("token_embedding.weight"), token_ids], prefix="token_embedding", axis=0)
    position_vectors = builder.op(
        "Gather", [builder.parameter("position_embedding.weight"), positions], prefix="position_embedding", axis=0
    )
    hidden = builder.op("Add", [tokens, position_vectors], prefix="embedded")

    for block in range(blocks):
        block_prefix = f"blocks.layers.{block}"
        normalized = builder.layer_norm(
            hidden,
            f"block{block}_attention_norm",
            f"{block_prefix}.attention_norm.weight",
            f"{block_prefix}.attention_norm.bias",
        )
        attended = builder.attention(normalized, block, model_size, heads, sequence_size)
        hidden = builder.op("Add", [hidden, attended], prefix=f"block{block}_attention_residual")
        normalized = builder.layer_norm(
            hidden,
            f"block{block}_ff_norm",
            f"{block_prefix}.feed_forward_norm.weight",
            f"{block_prefix}.feed_forward_norm.bias",
        )
        expanded = builder.linear(
            normalized,
            f"block{block}_expand",
            f"{block_prefix}.feed_forward.expand.weight",
            f"{block_prefix}.feed_forward.expand.bias",
        )
        activated = builder.gelu(expanded, f"block{block}_gelu")
        contracted = builder.linear(
            activated,
            f"block{block}_contract",
            f"{block_prefix}.feed_forward.contract.weight",
            f"{block_prefix}.feed_forward.contract.bias",
        )
        hidden = builder.op("Add", [hidden, contracted], prefix=f"block{block}_ff_residual")

    hidden = builder.layer_norm(hidden, "final_norm", "final_norm.weight", "final_norm.bias")
    logits = builder.linear(hidden, "output", "output.weight", "output.bias")
    builder.nodes.append(helper.make_node("Identity", [logits], ["logits"]))
    graph = helper.make_graph(
        builder.nodes,
        f"learning-llms-shakespeare-{model_id}",
        [helper.make_tensor_value_info(token_ids, TensorProto.INT64, [1, "sequence"] )],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, "sequence", 65])],
        builder.initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", OPSET)],
        producer_name="LearningLLMs",
        producer_version="1.0.0",
    )
    onnx.checker.check_model(model)
    return model, builder.parameter_manifest


def mlx_fixture(model: TinyTransformerLanguageModel, prompt: str, char_to_id: dict[str, int], vocabulary: list[str]) -> dict[str, Any]:
    token_ids = [char_to_id[character] for character in prompt]
    logits = model(mx.array([token_ids], dtype=mx.int32))[0, -1]
    mx.eval(logits)
    expected_logits = np.asarray(logits, dtype=np.float32)
    generated = list(token_ids)
    greedy_ids: list[int] = []
    greedy_margins: list[float] = []
    for _ in range(GREEDY_CHARACTERS):
        context = generated[-model.context_size :]
        next_logits = model(mx.array([context], dtype=mx.int32))[0, -1]
        sorted_logits = np.sort(np.asarray(next_logits, dtype=np.float32))
        greedy_margins.append(float(sorted_logits[-1] - sorted_logits[-2]))
        next_id = int(mx.argmax(next_logits).item())
        generated.append(next_id)
        greedy_ids.append(next_id)
    return {
        "prompt": prompt,
        "tokenIds": token_ids,
        "lastLogits": expected_logits.tolist(),
        "greedyTokenIds": greedy_ids,
        "greedyTopTwoMargins": greedy_margins,
        "greedyContinuation": "".join(vocabulary[token_id] for token_id in greedy_ids),
    }


def main() -> None:
    if not DATA.is_file():
        raise SystemExit("Tiny Shakespeare is missing. Run scripts/download_tiny_shakespeare.py first.")
    missing = [item["checkpoint"] for item in EXPORTS if not item["checkpoint"].is_file()]
    if missing:
        raise SystemExit("Missing reviewed source checkpoints: " + ", ".join(str(path) for path in missing))

    _, _, vocabulary, char_to_id = load_data(DATA)
    if len(vocabulary) != 65:
        raise SystemExit(f"Expected the reviewed 65-character vocabulary, found {len(vocabulary)}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest_models = []
    fixtures: dict[str, Any] = {}

    for item in EXPORTS:
        config = json.loads(item["config"].read_text(encoding="utf-8"))
        model_size = int(config["modelSize"])
        heads = int(config["attentionHeads"])
        blocks = int(config["transformerBlocks"])
        context = int(config["contextSize"])
        weights = load_file(item["checkpoint"])
        model_proto, parameter_manifest = export_onnx(weights, item["id"], model_size, heads, blocks)
        output_path = OUTPUT / f"{item['id']}.onnx"
        onnx.save_model(model_proto, output_path)

        mlx_model = TinyTransformerLanguageModel(len(vocabulary), context, model_size, heads, blocks)
        mlx_model.load_weights(str(item["checkpoint"]))
        mx.eval(mlx_model.parameters())
        fixtures[item["id"]] = [mlx_fixture(mlx_model, prompt, char_to_id, vocabulary) for prompt in PARITY_PROMPTS]
        manifest_models.append(
            {
                "id": item["id"],
                "label": item["label"],
                "file": f"{item['id']}.onnx",
                "bytes": output_path.stat().st_size,
                "sha256": sha256(output_path),
                "sourceCheckpoint": str(item["checkpoint"].relative_to(ROOT)),
                "sourceCheckpointSha256": sha256(item["checkpoint"]),
                "sourceRunId": item["runId"],
                "step": item["step"],
                "seed": int(config.get("seed", 42)),
                "contextSize": context,
                "modelSize": model_size,
                "attentionHeads": heads,
                "transformerBlocks": blocks,
                "parameterCount": int(config["parameterCount"]),
                "parameters": parameter_manifest,
            }
        )
        print(f"exported {item['id']}: {output_path.stat().st_size:,} bytes")

    fixture_path = OUTPUT / "parity-fixture.json"
    fixture_payload = {
        "schemaVersion": 1,
        "sourceBackend": "MLX 0.32.0",
        "logitAbsoluteTolerance": 1.5e-2,
        "greedyNearTieTolerance": 1e-3,
        "greedyCharacters": GREEDY_CHARACTERS,
        "models": fixtures,
    }
    fixture_path.write_text(json.dumps(fixture_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schemaVersion": 1,
        "createdAt": datetime.now(UTC).isoformat(),
        "course": "LearningLLMs",
        "modelFamily": "Tiny Shakespeare decoder-only transformers",
        "format": "ONNX",
        "opset": OPSET,
        "browserBackend": "ONNX Runtime Web WebAssembly",
        "runtimeVersion": "1.29.0",
        "promptsStayOnDevice": True,
        "dataset": {
            "name": "Tiny Shakespeare",
            "source": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
            "sha256": sha256(DATA),
            "bytes": DATA.stat().st_size,
        },
        "vocabulary": vocabulary,
        "unknownCharacterPolicy": "reject",
        "parityFixture": {
            "file": fixture_path.name,
            "sha256": sha256(fixture_path),
            "logitAbsoluteTolerance": fixture_payload["logitAbsoluteTolerance"],
        },
        "models": manifest_models,
        "limitations": [
            "WebAssembly inference uses the visitor's CPU and may be slower on low-power devices.",
            "Greedy generation is parity-tested; seeded sampling is not text-identical because MLX and browser random-number generators differ.",
            "The model has a 64-character context and rejects characters outside its frozen 65-character vocabulary.",
        ],
    }
    manifest_path = OUTPUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
