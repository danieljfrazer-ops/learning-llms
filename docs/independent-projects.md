# Independent project extensions

These projects remove structure gradually. They are not claims of completed work and must not display fabricated metrics.

## Project 04 · Movie-review sentiment laboratory

**Scaffolding:** complete lesson spine.

**Dataset:** Stanford [Large Movie Review Dataset](https://nlp.stanford.edu/~amaas/data/sentiment/) with 25,000 labelled training reviews and 25,000 labelled test reviews, plus separate unlabelled reviews. Audit the supplied terms and original README before redistribution.

**Question:** how do simple word-count features, zero-shot prompting, and parameter-efficient transformer adaptation differ in accuracy, calibration, memory, and failure modes?

### Lesson plan

1. **Frame classification.** Define positive/negative labels, accuracy, precision, recall, F1, calibration, constraints, and untouched test use.
2. **Audit the dataset.** Verify provenance, terms, class balance, review length, duplicates, HTML, label construction, and split integrity.
3. **Build non-neural baselines.** Majority class, bag-of-words, and logistic regression establish what a transformer must beat.
4. **Measure zero-shot behaviour.** Freeze a prompt format and record invalid responses, accuracy, memory, and latency without training.
5. **Fine-tune efficiently.** Compare a classification head or LoRA adapter while retaining base-model and random-adapter checkpoints.
6. **Evaluate properly.** Add confusion matrix, per-class F1, confidence calibration, length slices, and fixed examples.
7. **Challenge the model.** Test negation, sarcasm, mixed sentiment, domain shortcuts, and explanation/prediction disagreement.
8. **Package and document.** Build a local review UI, publish a model card, state licence and limitations, and preserve reference/local evidence.

**Success is not merely higher accuracy.** A useful conclusion explains which errors each approach makes, how confidence behaves, and whether apparent gains survive controlled slices.

## Project 05 · Dialogue summarisation

**Scaffolding:** design brief for your coding agent.

Use the [SAMSum corpus and paper](https://aclanthology.org/D19-5409/) to adapt a small pretrained model that summarises messenger-style dialogue.

Ask your agent to scaffold lessons that:

- audit provenance, licence, format, split integrity, speaker representation, dialogue length, and summary length;
- define extractive and no-training baselines before fine-tuning;
- select a quantised base model from measured memory, licence, and baseline results;
- freeze input format, decoding settings, ROUGE calculation, and manual review rubric;
- train a LoRA adapter with comparable checkpoints;
- measure loss, ROUGE, factual consistency, speaker/entity errors, omissions, memory, and time;
- maintain a fixed set of qualitative conversations; and
- explain the paper’s warning that automatic summarisation metrics may disagree with human judgement.

You choose the lesson count, base model, subset, manual rubric, and acceptable resource budget. Require the agent to present those decisions before implementation.

## Project 06 · Your own question

**Scaffolding:** themes only.

Choose a subject you care about and one checkable transformation. Possibilities include:

- local history or public-domain archives;
- bilingual or dialect text, designed with community context;
- recipes, music notation, poetry, tabletop settings, or game dialogue;
- extraction and structured output from documents;
- a small retrieval-assisted question-answering system;
- tutoring or feedback with an explicit correctness rubric; or
- experiments in context length, data quantity, synthetic data, distillation, or retrieval.

Define these yourself before asking an agent to build:

1. What exact input and output constitute the task?
2. Where may the data legally and ethically come from?
3. What is the simplest credible baseline?
4. What evidence would show improvement and what might that metric hide?
5. Which one variable will the first experiment change?
6. What harm, privacy issue, or misleading claim must the project prevent?

Retain the course’s only mandatory skeleton:

```text
question → baseline → controlled change → evidence → limitation
```
