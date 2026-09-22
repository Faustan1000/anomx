---
command: manage-systems
title: Manage Systems
description: Discover physical systems, build verified hierarchies, and connect data channels to systems.
hidden: true
system: true
---

# Manage Anomx Systems

Use this skill for system discovery, hierarchical systems, and assigning channels to
machines or subsystems. `/systems` contains the domain systems. `/system/nodes` and
`/system/services` describe the platform's compute infrastructure.

Read `retrieve-data` and `use-anomx-api` for channel discovery and API conventions.
For recurring discovery, inspect one page of `get_background_runs` once per run and
retain the chosen channel and progress after context compression. Use catalog pages
with explicit limits and `ordering=id`; vary the selected page/channel across runs.
Do not repeatedly load the full catalog or run history. An empty catalog page ends
that traversal; a failed endpoint is not an empty result.

Before writing, search `/systems?query=<name>` or filter `external_ref` to reuse
existing systems. Inspect the channel's identifier, description, units, and existing
connections. Treat identifier segments as evidence to investigate, not proof of a
physical hierarchy. Preserve uncertain relationships as notes or recommendations.

Hierarchy is represented by `/connections` with `kind=part_of`: the source is the
child system and the target is its parent. Names, `external_ref` prefixes, and JSON
properties do not create hierarchy edges. Read `GET /systems/explore` for roots and
`GET /systems/explore?focus=<system-uuid>&offset=0&limit=9` for a branch. The response
includes `selected`, `parents`, `nodes`, `child_counts`, `channels`, `channel_total`,
`total`, and `has_more`. Use the UUID from a retrieved system for `focus`.

When authorized to create a system, use `POST /systems` with supported fields such as
`name`, `description`, `kind`, `classification`, `external_ref`, `tags`, `properties`,
and `spatial`. `parent` is a creation-only convenience field; do not PATCH it to
reparent an existing system. For existing systems, create the actual connection:

```json
{
  "kind": "part_of",
  "source_object_reference": "systems_system-<child-uuid>",
  "target_object_reference": "systems_system-<parent-uuid>"
}
```

Send that body to `POST /connections`. The same source/target/kind is deduplicated
by the server. Self-links and cycles are invalid. To attach a channel measuring a
system, use `kind=observes`, the concrete `data_channel-<uuid>` as source, and the
`systems_system-<uuid>` as target. `contains` is for folders, not system hierarchy.
Always use references returned by the API. `GET /connections` is not a supported
list operation; verify hierarchy and attached channels through `/systems/explore`.

After each write, verify the returned fields and inspect the affected branch. Keep
connection IDs for any specifically authorized correction. Read existing JSON
properties before PATCHing: replacing that field replaces the JSON object. Complete
one evidenced channel-to-system chain per discovery run, then report the channel,
reused/created systems, verified connections, uncertainties, and any blocked writes.
Do not claim completion when only a recommendation was created. Background tasks
must obey their configured create/update/delete permissions for both systems and
connections; use recommendations for writes the server denies.

For infrastructure diagnostics, use `/system/health`, `/system/nodes`,
`/system/services`, and `/system/jobs`. Correlate identity, health, heartbeat,
timestamps, and error details before making operational recommendations.

System actions have broad impact. Do not cancel `/system/jobs/<id>/cancel`, change
service configuration, or issue an operational write unless it is explicitly requested
and permitted by the active mode. In Recommend mode, describe such an action in a
`POST /recommendations` proposal instead of applying it.
