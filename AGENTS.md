# Agent instructions

Read README.md, docs/agent-playbook.md, docs/adapter-protocol.md and SECURITY.md before changes. Work in a scoped branch and preserve other work. Use your host repository's task/session rules when applicable; this public repository does not depend on an internal workflow service.

- Measure before optimizing. Freeze workload/model/scene/controller/oracle inputs and distinguish process count from vectorized environment count.
- Keep the core standard-library-only. Optional GPU/SDK integration is behind explicit task adapters/examples; no private assets or environment paths in source.
- Default CLI behavior must remain dry planning. No automatic cloud creation, selected retries or assumption that failed/unknown workers are closed.
- Do not convert synthetic fixtures, state capacity diagnostics or all-failure policies into real task success evidence.
- Test original acceptance and resource/identity rejection paths. Run `python -m unittest discover -s tests -v` and a fresh synthetic demo before delivery.
- Use feature branches and reviewable changes. Never force-push, bypass verification hooks or merge without authority.
- Performance reports identify real hardware/SDK, timing method, full denominator, output acceptance, startup/drain and uncertainty. Fidelity changes require a separate scientific configuration.
