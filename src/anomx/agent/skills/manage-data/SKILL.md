---
name: manage-data
description: Safely inspect and manage Anomx datasets, channels, files, integrations, and data bindings.
metadata:
    title: Manage Data
    hidden: true
    system: true
---

# Manage Anomx Data

Before creating or changing data objects, retrieve the current object and inspect the
relevant request schema in `GET /openapi.json`. Prefer canonical object references and
small `PATCH` payloads over replacing whole records. Never infer identifiers, units,
connector settings, or destructive intent.

Common endpoints include `/datasets`, `/channels`, `/files`,
`/integrations`, and `/integrations/connector-catalog`. Search for an existing object
before creating one. After a write, read the returned or updated object and verify the
requested fields, relationships, ownership scope, and timestamps.

List endpoints automatically accept filters for serialized model fields. Use exact
filters such as `status=active`, lookup suffixes such as `name__icontains=temperature`,
`created_at__gte=<timestamp>`, `id__in=<id1,id2>`, and negation with `__not` or
`__not_in`. Use `ordering=<field>` or `ordering=-<field>`, `query=<text>` for model
search fields, and `limit`/`offset` for pagination. Invalid or non-serialized fields are
rejected instead of being silently ignored.


For discovered filesystem data, use `manage_data_adapters` to inspect schemas, list candidates, and propose explicit file-to-channel recipes. Exact Anomx archive identities are linked automatically by the platform. Never infer a sampling interval from array length alone. Background runs can propose recipes for owner review; activation and channel creation obey the current approval policy.


Keep data concepts separate: an Integration configures authenticated external access;
a Channel identifies a signal with shape, type, units and transport capabilities; a
Dataset groups materialized observations; a Job orchestrates acquisition/analysis; a
System describes the equipment/context. Use `manage-systems` for verified `part_of`
and channel→system `observes` edges instead of encoding hierarchy in names alone.

Read connector catalog fields and each integration's health/access mode before using it.
A discovered catalog entry does not prove live reads or historical storage are working.
The channel record's `last_value`/`last_value_at` is a persisted snapshot and is often
stale; list/detail reads do not acquire a fresh measurement. For a current value,
follow `retrieve-data` and request `GET /channels/<object-reference>/value`, which
asks the DAQ node service for the source value over NATS. Report the returned sample
time and freshness; do not substitute `last_seen` or `updated_at` for sample time.
Use the platform `inspect-platform` skill if service routing or a live read fails.
Preserve timestamps, source identities, units, sample shape and adapter provenance when
importing data. Use the platform `use-icons` skill for supported object icon fields.
