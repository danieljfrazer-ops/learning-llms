# Agent-assisted learning

A coding agent can perform setup, run experiments, diagnose failures, and maintain the wiki. The learning value comes from requiring the agent to expose decisions and evidence rather than asking for an unexplained finished model.

## Opening prompt

```text
Help me reproduce this course one lesson at a time. Read AGENTS.md, README.md,
the hardware guide, and the current wiki lesson before acting. Use My Lab paths
only. Before each command, explain what it will read, calculate, change, and
write in beginner-friendly language. After each run, compare my evidence with
the committed reference without overwriting it. Stop for a short lesson recap
before advancing.
```

## Blank-canvas prompt

```text
Assume this is a fresh clone. Inspect my hardware non-destructively, recommend
one supported backend, install only that profile, verify the dataset checksum,
start the wiki, and complete lesson 01. Preserve commands, versions, outputs,
and limitations. Do not publish, promote reference evidence, or skip ahead.
```

## What the agent should do

- explain purpose before mechanics;
- preserve exact technical terms while defining them in context;
- use the script defaults unless hardware requires a recorded change;
- show the untrained checkpoint before trained results;
- separate observed results from expectations;
- write generated evidence only under `public/data/local/` and `work/experiments/`;
- update lessons and glossary when introducing a genuinely new concept;
- run the Beginner Mode audit, lint, build, and relevant Python checks; and
- leave reference promotion and public release to an explicit maintainer request.

## What the learner should ask

- “What information enters this function and what comes out?”
- “Which numbers change during this command?”
- “Why is this comparison fair—or not perfectly fair?”
- “What would the output look like if this component were absent?”
- “Which evidence would disprove our explanation?”
- “What does the metaphor fail to capture?”

## Starting an independent project

For project 05, give the agent the dialogue-summarisation brief from [independent projects](independent-projects.md) and ask it to propose a lesson plan before implementing. Require dataset provenance, a baseline, frozen evaluation, resource budgets, failure analysis, reference/local separation, and Beginner Mode coverage.

For project 06, do not ask the agent to choose everything. Choose the domain and question yourself, then ask it to challenge the evaluation design and feasibility.
