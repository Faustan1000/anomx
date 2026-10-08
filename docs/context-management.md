# Agent context management

`maximum_context_tokens` bounds input context. A known model window also reserves
space for output. The CLI and platform display that effective maximum. Legacy
`context_compression_target_percent` settings are ignored.

## Two reduction levels

1. **Tool-result reduction** saves results above 16,000 characters to files before
   returning them to the model. The context receives a bounded structural preview,
   failure/status fields, and a `result_path`. Files retain the exact UTF-8 payload;
   the artifact's byte count and SHA-256 identify the complete saved content.
   This applies across tools, including extensions. Final answers are exempt.
   Command output also moves to a file before its middle rows would be discarded.
   The `read` tool returns bounded character pages, including for very long lines.
   Channel search returns compact metadata and references without stored waveforms;
   its complete API response is available at `request.response_path`.

   Smaller historical results can use the easy-work model one result at a time.
   Calls, arguments, call IDs, message roles and result ordering remain intact.
   The model receives the original and recent user requests, a bounded work
   outline, prior working memory, and the selected call's arguments and result.
   JSON remains JSON: objects retain their keys and types, lists retain selected
   items in source order, and scalar evidence is copied exactly. Long text fields
   can retain original lines. Invalid, empty or insufficiently smaller reductions
   are rejected. Under pressure or on follow-up, an unavailable or unsuccessful
   optimizer falls back to a file with an explicit partial preview. This does not
   require a model call. Large results in older sessions take that path directly,
   including individual records too large to fit the optimizer's batch budget.
2. **History summarization** starts at 48 context messages, independently of
   token utilization. It replaces an old prefix with first-person working memory
   while preferring 24 recent messages. At 80% token utilization, short histories
   can also be summarized, preferring four recent messages. The tail can shrink
   further to fit its budget while preserving the current complete exchange.
   The boundary must be durable and
   must not split an assistant's tool-call/result group. Each summary incorporates
   the previous summary and new history, retaining goals, decisions, verified
   outcomes, references, constraints and unfinished work. It must not imitate
   tool-call syntax. The summary is a separate system message (or a separate
   system text block for Messages APIs), explicitly labelled historical memory.

## Evaluation and limits

A new user message, reaching 48 messages, 24 additional entries, or crossing 50%, 65%, 80%, 90% or 99%
utilization triggers evaluation. A follow-up can reduce older results of at least
2,048 tokens even below 50% utilization. Under pressure, results of at least
2,048 tokens are candidates. The immediate file limit applies regardless of
utilization. Smaller immediate results may use model reduction above the smaller
of 8,192 tokens and one eighth of capacity when projected utilization reaches 50%.
Projection includes
provider input/output usage and the current batch's results.

The optimizer targets at most one third of a historical result, capped at 4,096
tokens. Accepted reductions save at least 25%. Model context budgets and request
counts bound background work. Final output never starts an optimizer request.

At 99%, or after a provider rejects its context, a safe reduction is required
before retrying. If tool reduction and eligible history summarization cannot
make room, the runtime returns a typed failure and preserves the evidence.
It never silently drops recent messages or repeats an oversized request.
The platform gives background reductions an 8,192-token output budget and rejects
responses that report output truncation. A failed artifact write stops an oversized
new result from entering context; original evidence is never replaced by a fake success.

## Persistence and provider replay

Full results are saved before optimization. Version 2 `context_optimization`
events replace only individual result bodies by stable storage ID. They never
merge or delete results. Legacy merged-digest replacements are ignored so their
original tool evidence can be replayed safely. Existing history-summary boundaries
remain readable. The raw session log is not rewritten.

All backends rebuild native calls and results: Responses function calls/outputs,
Chat Completions assistant calls/tool messages, Messages tool-use/result blocks,
and Ollama calls/tool messages. Provider reasoning is retained through in-turn
rebuilds where available. Local storage and optimization metadata are excluded
from provider wire messages. New persisted executions retain provider call IDs;
old executions receive stable replay IDs. A context reset does not execute tools.

## Activity display

One activity ID moves from running `check` to running optimization and then a
terminal result. The platform updates the same turn in place. Completed checks
and attempts without an accepted reduction disappear before tool-group creation,
so they do not leave a row or split adjacent tool groups. Accepted optimization
remains as “Context optimized”; subsequent tools form the next group. Errors
remain visible. The CLI also omits completed no-op checks.

The platform and CLI both use the easy-work model for these operations. Package
and platform changes must be released together; a platform deployment must update
its pinned/vendored package. This rework requires no database migration.
