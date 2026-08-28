# Licensing, datasets, and exported models

LearningLLMs uses a deliberate split licence:

| Material | Public licence | Boundary |
|---|---|---|
| Original source code and exported LearningLLMs Shakespeare ONNX files | Apache License 2.0 | See `LICENSE` and `NOTICE`. Third-party packages retain their own licences. |
| Original lesson prose, documentation, and course-native diagrams | Creative Commons Attribution 4.0 | See `LICENSE-CONTENT.md` for attribution guidance. |
| Downloaded datasets | Not redistributed | Downloaders fetch from the recorded upstream source into ignored `data/raw/` paths. Upstream terms continue to apply. |
| Quoted text, names, papers, and third-party documentation | Their original terms | Short quotations and links do not become LearningLLMs-owned material. |

## Dataset decisions

- **Tiny Shakespeare:** the [char-rnn repository](https://github.com/karpathy/char-rnn#license) describes the included subset and states an MIT licence. LearningLLMs does not commit the corpus. Its browser manifest records the exact source URL, 1,115,394-byte size and SHA-256 checksum. Four original LearningLLMs checkpoints derived through the documented training code are distributed as Apache-2.0 ONNX files.
- **TinyStories:** the source host records `cdla-sharing-1.0`. The course downloader stores only a deterministic local sample and its manifest under ignored `data/raw/`; no TinyStories rows or checkpoints are in the public repository.
- **WikiSQL:** the upstream code is BSD-3-Clause, but the archived repository does not explicitly resolve the bundled dataset licence. LearningLLMs therefore publishes aggregate measurements and failure counts only. Questions, queries, tables, raw files, adapters, and base-model files remain excluded.

## Browser-model provenance

`public/models/shakespeare/manifest.json` records, for every ONNX file:

- the source run ID, checkpoint step, seed and architecture;
- parameter count and per-layer source/export checksums;
- source-checkpoint and exported-file checksums;
- runtime, format, context length and vocabulary;
- the frozen MLX parity fixture and numerical tolerances; and
- privacy and browser limitations.

`npm run audit:browser` executes all four public files through the WebAssembly runtime and compares vocabulary IDs, logits and greedy continuations with the frozen MLX outputs. Seeded sampling remains repeatable within each engine, but MLX and JavaScript use different random-number generators and therefore are not expected to produce identical sampled passages.
