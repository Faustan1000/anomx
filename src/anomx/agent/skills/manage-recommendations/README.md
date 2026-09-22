---
command: manage-recommendations
title: Manage Recommendations
description: Inspect existing recommendations and create evidence-backed proposals without changing target objects.
hidden: true
system: true
---

# Manage Anomx Recommendations

Use recommendations to propose a change for human review. A recommendation is not
permission to apply the change. In Background mode, `POST /recommendations` is the
default permitted write. Direct create/update/delete is allowed only for model types explicitly granted in the platform Background policy; accepting, rejecting, superseding, updating, or deleting any
record is blocked.

## Required workflow

1. For a proposal about an existing object, resolve and read the target first. Prefer
   `get_anomx_object_details` when you already have its canonical object reference.
2. Always check the target's existing recommendations before creating a new one:

   ```text
   GET /recommendations?target=<object-reference>&ordering=-created_at
   ```

   Add `status=pending` when checking for an active duplicate, but also inspect recent
   accepted, rejected, and superseded recommendations when they affect the decision.
3. Do not create a duplicate. Reuse the evidence from an existing pending
   recommendation in your answer, or create a new recommendation only when the target,
   proposed change, evidence, or operational context is materially different.
4. Verify current values immediately before writing so `current_values` is an accurate,
   minimal snapshot of the fields the proposal would change.
5. Create the proposal with `POST /recommendations`, then inspect the returned record
   and report its object reference.

## Create payload

```json
{
  "name": "Short action-oriented title",
  "description": "Why this change is useful and what a reviewer should consider.",
  "kind": "update",
  "proposed_by": "Anomx agent",
  "proposed_changes": {"field": "new value"},
  "current_values": {"field": "current value"},
  "evidence": {
    "summary": "Observed facts supporting the proposal",
    "sources": ["data_channel-...", "jobs_job_run-..."],
    "observed_at": "ISO-8601 timestamp when available"
  },
  "confidence": 0.85,
  "target_object_reference": "data_channel-..."
}
```

`kind` is one of `create`, `update`, `delete`, `connect`, `info`, `warning`, or `chat`. Use canonical Anomx
object references. A target is required except for `create` and `chat`; for a create proposal,
search for an equivalent object before omitting the target. `proposed_changes`,
`current_values`, and `evidence` must be JSON objects; confidence is optional and
ranges from 0 to 1. Omit `status`: new recommendations are always pending. Existing
recommendations are immutable except for review status (`accepted`, `rejected`, or
`superseded`), and the platform records `reviewed_by` and `reviewed_at`; Background
mode must leave that review action to a human.

Keep proposals small, inspectable, and reversible. Put facts and source references in
`evidence`, uncertainty in `confidence` and `description`, and never include API keys,
credentials, or unrelated personal data.

## Informational and chat recommendations

Use `info` for a useful observation and `warning` for a risk that needs attention.
These display as dismissible alerts above the target object's details; include a
short `name`, a clear `description`, and evidence. They need no `proposed_changes`.
Use `chat` for a suggested follow-up conversation: include `prompt` with the exact
starting prompt. Chat suggestions appear before the default home suggestions and
can be dismissed or inserted into the prompt bar. Do not use the retired `review`
kind. Creation and user interactions are audited. Create proposals are retained
for future workflows and do not currently appear in object alert stacks.

Before generating new alerts or chats, inspect `get_background_runs` and existing
recommendations. Prefer a small number of useful findings over repeated alerts.
The configured direct-change allowlists never authorize accepting your own
recommendations or changing automation permissions.
