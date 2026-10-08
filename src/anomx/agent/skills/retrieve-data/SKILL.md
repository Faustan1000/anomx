---
name: retrieve-data
description: Discover Anomx data sources, request current channel values from DAQ services, and retrieve bounded history with explicit freshness.
metadata:
    title: Retrieve Data
    hidden: true
    system: true
---

# Retrieve Anomx Data

Use the dedicated data tools first. `search_anomx_data_channels` uses the platform
channel search bar's matching and discovery: an empty query browses roots, a name
or prefix searches the catalog, a trailing slash browses children, and wildcards
or full addresses narrow the results. It returns compact `matches` with object
references, completion hints, pagination and discovery status. Full responses are
saved at `request.response_path`; waveforms stay out of the search context.
If `loading` is true, wait briefly and repeat with `refresh=false`. A typed
suggestion marked `unverified` does not establish that a channel exists. Partial
results or `catalog_limited` results cannot prove absence; narrow the query.

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
- `GET /channels/search?query=<name-or-pattern>&limit=<n>&page=1`
- `GET /channels/search?query=<name-or-pattern>&refresh=false` to poll discovery
- `GET /channels/<object-reference>/value`
- `GET /channels/<object-reference>/history?range=1h&max_points=100`

## Current values and freshness

For "latest", "current", "now" or a live channel reading, resolve the concrete
channel reference, then call `use_anomx_api` with `method="GET"` and
`path="/channels/<object-reference>/value"`. This endpoint is excluded from the
platform response cache: it resolves the channel's DAQ service, sends
`daq.channel.value` over the service bus (NATS), and returns the source response.
The API handles configured/fallback service selection and integration scope, and
updates the persisted channel snapshot after a successful read. A one-off value
read does not require creating a recording job or starting a broadcast.

`last_value` and `last_value_at` on channel list/detail records are persisted
snapshots, often old when the channel has not been read recently. They are not a
continuously refreshed live feed. `last_seen`, `updated_at`, a recent service
heartbeat, and the newest point in stored history do not prove a current sample.
If the user asks for the latest **recorded** value instead, use recorded history
and label its sample time; a fresh source read answers a different question.

Use the value endpoint's returned `value` (including valid zero/false values),
`timestamp`, `unit`, shape, source metadata and actual `service`. State the sample
time and age relative to the request and expected update rate. A fresh request can
still return an old source sample. Some connectors omit a source timestamp and the
API can substitute the observation time in `timestamp`/`last_value_at`; do not
claim independently verified sensor freshness when timestamp provenance is unknown.

HTTP 409 means no DAQ service was resolved; HTTP 502 indicates a failed source/service
read. Report failures and permission denials explicitly. A last-known DB value may
be useful as a labeled fallback with its timestamp/age, never silently as "current".
For routing, health or advanced read parameters, use the platform `inspect-platform`
skill and its channel-value diagnostics. Read only the requested channels; do not
refresh the entire catalog to answer a single-value question.

The API tool returns a bounded parsed response directly and writes the complete JSON
payload to the response path it reports. Analyze the returned JSON directly; a missing
shell or Python tool is not a blocker. If the preview is truncated, use `read` on the
reported response path and continue in chunks.

Start with narrow queries and explicit limits. Follow pagination metadata (`count`,
`next`, `previous`, `limit`, and `offset`) rather than assuming the first page is
complete. Preserve timestamps, units, quality/status fields, channel identifiers, and
object references when comparing values. State when data is sampled, aggregated,
missing, stale, or outside the requested time range.

For `/channels`, the response is a bare array with no total-count wrapper. Use
`limit` (maximum 100) and `offset` with stable `ordering=id`; advance the offset by
the number returned until an empty page, and verify that pages contain different
IDs. The tool's `result_count` counts only the returned page. For the accessible
catalog total, use `GET /channels/overview` and read `stats.known_channels`
(`stats.recorded_channels` counts recordings separately). This is not a filtered
search count or the count of every live signal available from external systems. For
`search_anomx_data_channels`, follow `has_more` and `next_page`. Older platforms
use the compatibility result's `pagination.has_more` with the next `page`.
