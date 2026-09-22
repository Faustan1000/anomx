<p align="center">
  <a href="https://anomx.io">
    <picture>
      <img src="assets/logo.png" alt="Anomx logo">
    </picture>
  </a>
</p>

<p align="center">The AI agent for coding, data analysis and anomaly detection.</p>

<p align="center">
    <a href="https://pypi.org/project/anomx/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/anomx?style=flat-square&label=pypi" /></a>
    <a href="https://pypi.org/project/anomx/"><img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/anomx?style=flat-square"/></a> 
</p>

[![Anomx Terminal UI](assets/anomx-shot-03.png)](https://anomx.io)

# Anomx

`anomx` is the installable Python foundation of Anomx. It ships both reusable
time-series and anomaly-detection primitives and the `anomx` CLI agent for
interactive anomaly investigation, data analysis, and platform-connected edge
workflows.

The wider Anomx platform handles orchestration, workers, storage, connectors,
audit trails, and human-in-the-loop review. This repository contains the
portable modeling layer plus the operator-facing CLI agent that can run on any
server, workstation, or laptop.

## Goals

- Provide clean primitives for time-series datasets, scorers, detectors, and models.
- Support both batch and online anomaly detection workflows.
- Keep outputs interpretable: scores, thresholds, timestamps, labels, and metadata.
- Ship a practical CLI agent that is tuned for anomaly detection, time-series inspection,
  and data-quality analysis tasks.
- Stay modular enough to power the Anomx platform without coupling the library to the
  platform's storage or orchestration layer.
- Integrate with the broader time-series ecosystem, especially Darts, through optional
  adapters.

## Installation

The package is structured for PyPI distribution:

```bash
pip install anomx
```

For local development:

```bash
git clone https://github.com/anomx/anomx.git
cd anomx
python -m pip install -e ".[dev]"
```

Optional extras:

```bash
pip install "anomx[darts]"     # Darts forecasting model adapters
pip install "anomx[docs]"      # Documentation tooling
pip install "anomx[platform]"  # Platform-facing IO and connector dependencies
pip install "anomx[release]"   # Build and PyPI publishing tooling
```

## Python Quick Start

```python
from anomx.datasets import make_sine_anomaly_dataset
from anomx.detectors import MovingAverageDetector
from anomx.models import NaiveSeasonalModel

dataset = make_sine_anomaly_dataset()

detector = MovingAverageDetector(window=24, threshold=3.0)
result = detector.fit_predict(dataset)
print(result.to_dataframe().tail())

model = NaiveSeasonalModel(season_length=24).fit(dataset)
forecast = model.predict(12)
print(forecast.to_dataframe())
```

## Anomx CLI Agent

Installing the package also installs the `anomx` console command:

```bash
anomx
```

The CLI agent is meant to feel like a modern coding/data agent, but tuned for
anomaly detection, time-series investigation, data quality problems, and
operational analysis.

Useful startup examples:

```bash
anomx --version
anomx --print-home
anomx --provider openai --model gpt-5.5
anomx --provider anthropic --model claude-sonnet-4-6
anomx --provider kimi --model kimi-k3
anomx --ollama --model qwen3-coder:30b
```

Current CLI capabilities include:

- A full-screen terminal UI with persisted transcripts and session history.
- Multiple model backends: OpenAI, Anthropic, Kimi, DESY Assistant, JSC
  Blablador, and local Ollama.
- Five execution modes: Plan, Recommend (with a connected platform), Standard,
  Automatic, and Autonomous.
- Connected-platform skills for the Anomx API, recommendations, data, jobs, and
  systems, plus user-defined slash-command skills.
- Tool-backed repository and data inspection, plus background Worker agents for
  focused parallel tasks.
- A small inspectable home directory at `~/.anomx` (or `ANOMX_HOME`) that stores
  config, auth metadata, skills, and session transcripts.

The CLI home structure looks like this:

```text
~/.anomx/
  config.toml
  auth.json
  skills/<command>.md
  session_index.jsonl
  sessions/YYYY/MM/DD/rollout-<timestamp>-<id>.jsonl
```

### Platform Connection

When an Anomx Platform instance is available, the CLI can be connected directly
from the agent UI using normal platform credentials. The platform issues a
dedicated CLI-agent token, tracks the CLI host name and client version, and
keeps the machine attached to the same organization context as the platform.

That makes `anomx` the practical edge entry point for working where the data
lives first, then carrying that context back into the platform for visualization,
findings, follow-up jobs, and broader analysis workflows.

## Package Layout

```text
src/anomx/
  agent/         Full-screen CLI agent, providers, skills, platform client
  data/          Connectors and sequence containers for local and edge workflows
  datasets/      TimeSeriesDataset, metadata, dataset loaders, transforms
  scorers/       Anomaly score contracts and scoring implementations
  detectors/     Batch and online anomaly detector contracts and implementations
  models/        Forecasting model contracts and baseline forecasters
  components/    Reusable component and offline pipeline abstractions
  integrations/  Optional adapters, including Darts
```

## Darts Integration

The default install stays lightweight. To use Darts models, install the optional
extra and wrap any compatible Darts forecasting model:

```python
from darts.models import ExponentialSmoothing

from anomx.datasets import make_sine_anomaly_dataset
from anomx.integrations import DartsForecastingModel

dataset = make_sine_anomaly_dataset()
model = DartsForecastingModel(ExponentialSmoothing())
forecast = model.fit(dataset).predict(24)
```

The intent is to let Anomx expose Darts-backed forecasting and anomaly workflows
without making Darts a required dependency for every user.

## Development

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
mypy src/anomx
python -m pip install -e ".[release]"
python -m build
```

## Platform Context

Anomx prioritizes:

- Signal over noise: surface only relevant insights.
- Clarity over complexity: outputs should be interpretable.
- Actionability: every insight should support a decision.

The platform is designed around modular pipelines, real-time and batch workers,
heterogeneous data sources, and versioned entities such as datasets, channels,
jobs, runs, findings, model artifacts, and node services. This package is the
focused modeling and agent layer used by that larger system.

## Status

This repository is in pre-alpha stage. Public APIs and CLI ergonomics should be
treated as provisional until the first stable release.
