---
command: retrieve-data
title: Retrieve Data
description: Discover Anomx data sources and retrieve bounded, inspectable channel data.
hidden: true
system: true
---

# Retrieve Anomx Data

Use the dedicated data tools first. `search_anomx_data_channels` discovers live
channels from leading identifier segments and returns continuation hints;
`get_anomx_data_channel_history` retrieves a bounded history window for a concrete
`data_channel-...` reference. Use `use_anomx_api` for endpoints not covered by those
tools.

For a broad topic such as "gun", search the persisted catalog with
`GET /channels?identifier__icontains=gun&limit=10` to find concrete prefixes, then
use live discovery on those prefixes. An empty live result can mean discovery is
still pending or the prefix is incomplete; follow returned hints. HTTP failures
must be reported as failures, not empty results, and a collection 404 is not repaired
by trying more prefixes. Use `manage-systems` to build relationships after discovery.

Useful reads include:

- `GET /datasets`
- `GET /channels`
- `GET /channels/live-search?query=<leading-prefix>&limit=<n>`
- `GET /channels/live-hints?query=<leading-prefix>&limit=<n>`
- `GET /channels/<object-reference>/value`
- `GET /channels/<object-reference>/history?range=1h&max_points=100`

The API tool returns a bounded parsed response directly and writes the complete JSON
payload to the response path it reports. Analyze the returned JSON directly; a missing
shell or Python tool is not a blocker. If the preview is truncated, use `read` on the
reported response path and continue in chunks.

Start with narrow queries and explicit limits. Follow pagination metadata (`count`,
`next`, `previous`, `limit`, and `offset`) rather than assuming the first page is
complete. Preserve timestamps, units, quality/status fields, channel identifiers, and
object references when comparing values. State when data is sampled, aggregated,
missing, stale, or outside the requested time range.

For `/channels`, use `limit` and `offset` with stable `ordering=id`; older deployments
support `page` and `size` instead. Verify that pages contain different IDs. For
`search_anomx_data_channels`, follow `pagination.has_more` with the next `page`.
