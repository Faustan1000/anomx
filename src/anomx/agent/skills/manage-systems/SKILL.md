---
name: manage-systems
description: Discover physical systems, build verified hierarchies, and connect data channels to systems.
metadata:
    title: Manage Systems
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
properties before PATCHing: replacing that field replaces the JSON object. Complete the requested, evidenced branch in bounded steps and retain a continuation
checkpoint for larger catalogs. Report the covered channels, reused/created systems,
verified connections, uncertainties, and any blocked writes.
Do not claim completion when only a recommendation was created. Background tasks
must obey their configured create/update/delete permissions for both systems and
connections; use recommendations for writes the server denies.

Use the platform's `use-icons` skill when available to select and validate the top-level
`icon` for a new system or an icon change. Choose consistent icons for sibling roles;
never invent names. Preserve an existing valid user choice. A physical hierarchy should
reflect facilities, machines, subsystems and measured components with meaningful names,
not one artificial System for every identifier segment. Search and reuse each evidenced
ancestor; keep provenance in descriptions/properties without making it a hierarchy edge.

A channel's catalog identity, live value, historical availability and recording state
are distinct. An `observes` edge classifies the signal's meaning; it does not start
recording or configure its DAQ service. Verify current value/history via `retrieve-data`
and use `manage-jobs` for acquisition changes.

Platform nodes and services are separate from the domain hierarchy. Use `inspect-platform`
when available for a multi-host overview and supported health/info probes. A system
organization request does not authorize restarting workers or changing compute capacity.
