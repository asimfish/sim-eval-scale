# Execution and data boundaries

This is a local orchestration library, not a sandbox or a multi-tenant service. A task adapter is trusted executable code selected by the operator; never use an unreviewed model response or downloaded script as an adapter.

- The core creates no cloud jobs and does not cancel other resources. Execution is bounded by the existing authorized allocation, policy GPU/world/trial/time limits and ownership inspection.
- Commands are structured argv without shell interpolation. Each adapter runs in a new private trial directory and process group. Unknown worker closure stops the sweep; terminating the adapter group does not prove detached simulation descendants exited.
- Actual task adapters remain responsible for provider ownership verification, original scientific validators, PID-generation/namespace-aware descendants and their closure receipts. A Boolean claim alone cannot make an incorrect validator trustworthy.
- Reports and audit evidence are bound to the trial, fixed scene roster, scientific fingerprint and physical GPU identity. Audit paths resolve inside the trial; content hashes detect accidental changes. Hashes are integrity checks, not authentication of an adversarial adapter.
- Credentials are not forwarded by default. Authorized workloads can explicitly allow variable names through `environment_allowlist`; values stay in the secure process environment and must not be logged or written to policy/command data.
- Default dry planning creates no simulator/GPU workload. Synthetic examples cannot qualify as actual-workload reports. Unknown/infrastructure cases are kept separate from model failures.
- Atomic artifact publication assumes operator-approved own-task source/destination roots. It rejects source symlinks and changed snapshots but is not an arbitrary-path authorization layer.

When reporting a security issue, provide a minimal redacted reproduction through the repository's GitHub Security reporting facilities if enabled. Do not post credentials, private datasets or infrastructure details in public issues. No security claim should rely only on a clean pattern scan.
