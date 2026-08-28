# Security policy

LearningLLMs is an educational, local-first repository. The public wiki is intended to serve course pages and reviewed static evidence. It is not a hosted training service and must not receive learner prompts, datasets, credentials, or database connections.

## Report a vulnerability privately

Please use GitHub's **Report a vulnerability** form in the repository Security tab. Do not open a public issue for a suspected secret, unsafe archive path, dependency compromise, prompt-privacy problem, or SQL execution escape.

If private vulnerability reporting is not enabled on the public repository, contact the maintainer through the private contact method listed on their GitHub profile. Do not include live credentials or personal data in the first message.

## Supported version

Until the first tagged release, only the latest commit on the default branch is maintained. After release, the latest published tag and the default branch receive security fixes; old course snapshots are retained for reproducibility but are not patched in place.

## Security boundaries

- Local inference binds to `127.0.0.1`; never expose ports `8001` or `8002` to a network.
- Generated SQL is untrusted. Only the restricted logical form may be rebuilt as parameterized SQL and executed against the supplied read-only benchmark database. Do not point the evaluator at production data.
- Downloaders pin expected archives or source selections and verify hashes before replacing files. Datasets remain under ignored `data/raw/`.
- Safetensors is used for course weights; do not add pickle-based checkpoints from untrusted sources.
- Learner prompts and My Lab evidence stay on the learner's machine. A future browser playground must preserve that boundary and must not add analytics or remote inference without a separate, visible privacy decision.
- GitHub Actions must use GitHub-hosted runners. Never attach a personal self-hosted runner to this public repository.

These controls reduce risk but do not make generated text or SQL suitable for production use.
