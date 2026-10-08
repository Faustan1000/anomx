---
name: use-anomx-api
description: Internal platform API instructions for connected Anomx agents.
metadata:
    title: Use Anomx API
    hidden: true
    system: true
---

# Connected Anomx Platform API

This agent is connected to an Anomx Platform instance. Use the `use_anomx_api`
tool or the helper file at `~/.anomx/skills/use-anomx-api/api.py` whenever the
task involves platform data, projects, pages, files, jobs, channels, object
search, system state, or user-owned platform records.

The runtime exports these environment variables when a platform is connected:

- `ANOMX_PLATFORM_API_URL`: Base API URL, usually ending in `/api`.
- `ANOMX_PLATFORM_URL`: Same base URL for compatibility.
- `ANOMX_PLATFORM_API_KEY`: Bearer token for platform API calls.
- `ANOMX_PLATFORM_TOKEN`: Same bearer token for compatibility.
- `ANOMX_API_KEY`: Same bearer token for compatibility.
- `ANOMX_RESPONSES_DIR`: Directory for raw API responses, normally `~/.anomx/responses`.

Prefer the `use_anomx_api` tool for simple calls because it automatically stores
the response body as JSON and returns a bounded parsed response plus the response
file path. Analyze that response directly. For a successful JSON response marked
truncated, use `read` on the response path; missing shell or Python execution is
not a blocker. For failed non-JSON responses, follow the returned diagnostic;
HTML debug-page route listings are not the API contract.
For custom local scripts, import or execute `api.py`; it reads the environment
variables above and writes raw payloads into the responses directory.

Examples:

```bash
python ~/.anomx/skills/use-anomx-api/api.py GET /objects --query '{"query":"xfel","limit":10}'
python ~/.anomx/skills/use-anomx-api/api.py GET /channels --query '{"query":"temperature","limit":10}'
python ~/.anomx/skills/use-anomx-api/api.py POST /folders --body '{"name":"Analysis","description":""}'
```

The helper output is intentionally short: HTTP status, response length,
detected result count, and the JSON file path. Read that file when the payload
matters.

Paths here are relative to `ANOMX_PLATFORM_API_URL`; do not prepend `/api` again.
For example, `/channels` resolves to `<base>/channels` and `/openapi.json` to
`<base>/openapi.json`. The schema documents the same unversioned paths, relative to
this API base. Do not add `/v1`, `/api/v1`, or module prefixes such as `/data/channels`
or `/jobs/jobs`. These belong to a separate compatibility API, not this tool's
documented routes.

Check `ok`, `status_code`, and error details before interpreting results. A 404 from
a collection endpoint is not evidence of no matching data. Do not retry that same
endpoint with different search terms; inspect the documented path once, then report
the blocker if it still fails. A detail 404 may indicate an inaccessible or stale
object reference. Send `query`, `body`, and `headers` as JSON objects, not JSON strings.
HTTP 200 alone does not prove a write changed the intended fields: verify them in
the returned object or a bounded follow-up read.

All standard list endpoints automatically support filters for serialized fields.
Pass exact field values directly, or use suffixes such as `__icontains`, `__in`,
`__gte`, `__lte`, `__isnull`, `__not`, and `__not_in` where appropriate. Use
`ordering=field` or `ordering=-field`, `query=<text>` for the model's configured
search fields, and `limit` plus `offset` for pagination. Invalid fields and lookups
return a validation error instead of being silently ignored. Relationship-aware
lists may also support `for_object=<object-reference>` and `for_object_kind=<kind>`.

List response shapes vary by endpoint. `result_count` is the number of records in
the returned response, not the total across pages. `/channels` returns a bare array
and caps `limit` at 100. To count the accessible catalog, call
`GET /channels/overview` and read `stats.known_channels`; `stats.recorded_channels`
counts recordings separately. These are catalog counts, not the number of all
possible live signals or a filtered search total. For a filtered channel list,
use `limit` and `offset` with stable `ordering=id`, advancing by the number returned
until an empty page. Do not try ever-larger limits to obtain a total.

Important platform endpoints:

- `GET /objects`: Unified object search. Useful query params include `query`,
  `model_reference`, `limit`, and `offset`.
- `GET/POST /systems`, `GET/PATCH /systems/<id>`, `GET /systems/explore`,
  `POST/PATCH/DELETE /connections`: domain systems and their relationships.
  Read `manage-systems` before constructing or correcting hierarchy.
- `GET/POST /recommendations`, `GET/PATCH /recommendations/<id>`:
  recommendation proposals and review status. Filter an object's recommendations
  with `target=<object-reference>` and optionally `status=pending`.
- `GET /account`, `GET/PATCH/DELETE /account/profile`,
  `PATCH /account/preferences`, `PATCH /account/password`:
  user account and preferences.
- `GET/PATCH /account/organization`, `GET /account/workspace`,
  `GET/POST /account/tokens`: account organization, workspace summary, and API
  tokens.
- `GET /account/trash`, `POST /account/trash/<resource_type>/<object_id>/restore`:
  trash.
- `GET/POST /folders`, `GET/PATCH/DELETE /folders/<id>`: folders/projects.
- `GET/POST /pages`, `GET/PATCH/DELETE /pages/<id>`: pages and dashboards.
- `POST /pages/<id>/update-component-positions`: reorder page components.
- `GET/POST /files`, `GET/PATCH/DELETE /files/<id>`, `POST /files/upload`:
  files and uploads.
- `GET/POST /integrations`, `GET /integrations/connector-catalog`:
  integrations and connector metadata.
- `GET /datasets`, `GET /channels`, `GET /channels/overview`,
  `GET /channels/search?query=<name-or-pattern>&limit=10&page=1`:
  data catalog and live channel discovery using the platform search bar's matching.
  Poll pending discovery with `refresh=false`; search results are partial and omit
  stored channel values. Prefer `search_anomx_data_channels` for compact results.
  Older platforms expose `GET /channels/live-hints` and `GET /channels/live-search`.
- `GET /channels/<id>/history`: bounded channel time series; the newest stored
  point may be old.
- `GET /channels/<id>/value`: request a current value from the DAQ service over
  NATS, bypassing the platform response cache. Channel list/detail `last_value`
  fields are persisted snapshots and can be stale. Follow `retrieve-data` for
  timestamps and source freshness, and `inspect-platform` for service diagnostics.
- `GET /jobs/build-options`, `GET/POST /jobs`,
  `GET/PATCH/DELETE /jobs/<id>`: job configuration and orchestration
  objects.
- `POST /jobs/<id>/start`, `POST /jobs/<id>/resume`,
  `POST /jobs/<id>/pause`, `POST /jobs/<id>/stop`,
  `POST /jobs/<id>/archive`, `POST /jobs/<id>/restore`: job actions.
- `GET /models/featured`, `GET /models`, `GET /algorithms`,
  `GET /scorers`, `GET /detectors`, `GET /components`:
  component catalogs.
- `GET /findings`, `GET /model-artifacts`, `GET /job-runs`:
  run outputs.
- `GET /agents/chats`, `GET /agents/turns`, `GET /agents/runs`,
  `GET/PATCH /agents/settings/me`: agent state.
- `get_background_runs`: bounded summaries of past unattended runs. Read the most
  recent page once and request older pages only for a specific unresolved question.
  Full details are available through `/agents/chats/<id>`; do not load stored
  compression metadata into the current conversation.
- `POST /agents/turns/<id>/approval`, `POST /agents/turns/<id>/question`:
  human-in-the-loop agent responses.
- `GET /system/health`, `GET /system/nodes`, `GET /system/services`,
  `GET /system/jobs`, `POST /system/jobs/<id>/stop`,
  `POST /system/jobs/<id>/enable`: operator system state.
- `GET /openapi.json`: full OpenAPI schema for exact request and response shapes.

Write requests use the same approval pipeline as command execution. The active runtime policy determines write permissions. A recommendations-only
policy permits proposals through `POST /recommendations`; background policies may
also grant specific create/update/delete model operations. Inspect the current policy
and obey denials rather than assuming a mode name grants access. When creating or updating records, inspect the relevant schema first through
`/openapi.json` or by retrieving a similar object. Keep writes scoped to the
user request and report the response file path in your final summary when it
contains important details.


For platform-native documents and widget dashboards, read `edit-pages` when available;
for custom coded Pages read `create-anomx-apps`. Use `navigate-platform` for built-in
screens and settings; UI navigation paths are not API endpoints. `/agents/skills`
provides editable account/organization skills, and `/agents/skills/bundled` exposes the
read-only catalog. Detail reads include supporting files; list reads omit their content.
`SKILL.md` is the canonical entry point; old README-based local skills remain supported.
