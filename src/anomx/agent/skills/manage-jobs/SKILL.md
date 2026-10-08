---
name: manage-jobs
description: Configure, start, pause, stop and diagnose Anomx acquisition and analysis jobs using actual worker capabilities and run evidence.
metadata:
    title: Manage Jobs
    hidden: true
    system: true
---

# Manage jobs

A Job is persisted configuration; a JobRun is an execution with input/config snapshots and outcomes. Job channel bindings connect live input channels, data bindings describe recordings/files, and component bindings select model/scorer/detector implementations. Do not confuse these with `/system/jobs`, which are platform maintenance tasks.

Read [job contracts](references/jobs.md) and `GET /jobs/build-options` before creating or restructuring a job. This resolves real DAQ services, stores, component definitions and protocol limits. Retrieve an existing job before patching it and preserve unrelated configuration and bindings. Choose actual capabilities and validate units, timing and channel shape before enabling recording or analysis.

Use the current `start`, `resume`, `pause`, and `stop` actions under `/jobs/<id>/...`; the historical `/run` endpoint is not implemented by the current platform. Creating a job with `start_immediately=true` can also dispatch work. Configure an inactive job with `start_immediately=false` when the request only authorizes setup; a request to start acquisition authorizes the appropriate action under the active runtime policy.

Retrieve the job after an action, then inspect its run, errors and timestamps. An accepted action is not evidence that live acquisition has produced samples. Verify bounded history/storage and correlate run IDs. Keep component/code versions, configuration snapshots and model artifacts in explanations. Use `inspect-platform` when available for worker/bus problems, and `investigate-findings` for anomaly evidence.
