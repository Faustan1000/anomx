# Current job contracts

`GET /jobs/build-options` supplies `daq_services`, `timeseries_stores`, `sample_stores`, `components`, and the legacy protocol sampling-frequency limit. `/jobs/component-options` and `/models`, `/algorithms`, `/scorers`, `/detectors`, `/components` expose further catalogs. Inspect component `config_schema`, defaults, capabilities, signature/parameters and code version instead of selecting by name alone.

The current create serializer supports `job_type=data_acquisition`; standalone `anomaly_detection` job creation is explicitly rejected. Acquisition jobs can enable supported detection or forecasting via flags and validated `runtime_config`; this does not imply that every algorithm or automatic selection mode is available.

- `work_mode=continuous` requires no duration; `fixed_duration` and `sliding_window` require a positive `duration_seconds`.
- Sampling frequency must be positive and supported by the chosen protocol/service; do not request a frequency beyond the published legacy limit.
- DAQ jobs require at least one recordable channel and a DAQ service. Current channel validation supports single/array bool/float/int channels.
- Time-series and independent-sample channels cannot be mixed in one job. Select the appropriate time-series or sample store from build-options. Forecasting is limited to time-series data.
- Detection's current configuration is manual and requires a positive detection period. Inspect the current schema/validation for required training intervals, models, scorers and detectors; do not invent a config from future product plans.
- `record_data`, `detect_anomalies`, and `forecast_data` are separate concerns. Describe which outputs the user requested and the storage/compute consequences.

Related endpoints: `/job-channel-bindings` (live channel roles, sampling, transport, read parameters), `/job-data-bindings` (recordings/files), `/job-component-bindings` (model/scorer/detector roles, selection and config). Use canonical references and serializer-supported relation fields. Read existing bindings before updates; a replacement array must retain unrelated bindings.

Execution evidence is available through `/job-runs`, `/run-component-usages`, `/run-metric-points`, `/model-artifacts`, and `/findings`. A job's current configuration may differ from the frozen configuration of an old run. For a failure, connect its run timestamps/error to the specific DAQ/compute service and storage identity rather than assuming the API host did the work.

Maintainer sources: platform `modules/jobs/models.py`, `serializers.py`, `viewsets.py`, worker runtimes and job utilities. Expand this reference as execution contracts evolve; preserve the runtime-verified distinctions above.
