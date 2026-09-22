---
command: manage-jobs
title: Manage Jobs
description: Inspect, configure, run, and diagnose Anomx jobs through auditable platform APIs.
hidden: true
system: true
---

# Manage Anomx Jobs

Read `GET /jobs/build-options` and the relevant component catalogs before
creating or restructuring a job. Useful catalogs include `/models`,
`/algorithms`, `/scorers`, `/detectors`, and `/components`.
Retrieve the current job before patching it and preserve bindings or configuration not
named by the user.

Use `POST /jobs/<id>/run` and `/stop` only when the active mode and the user's
request authorize execution. Diagnose outcomes through `/job-runs`,
`/findings`, and `/model-artifacts`; retain run IDs, component versions,
timestamps, configuration snapshots, metrics, and errors in conclusions. After any
write or action, retrieve the job or run again and verify the resulting state.
