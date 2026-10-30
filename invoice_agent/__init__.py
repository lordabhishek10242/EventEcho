"""EventEcho Invoice Action Agent.

A dependency-free A2A 1.0 (HTTP+JSON) service that reads invoice claim batches,
proposes exactly one typed business action per package with cited evidence, and
executes only the actions the grader/reviewer accepts.

The service is deliberately split into layers:

    documents.py  -> document flattening, reference extraction, fact extraction
    decisions.py  -> the decision model (evidence-scored, negation aware)
    ai.py         -> model-provider boundary (local model by default)
    store.py      -> SQLite persistence: tasks, idempotency, decision cache
    protocol.py   -> A2A envelopes, Agent Card, task serialisation
    server.py     -> HTTP surface (stdlib only)

Run with:  python3 -m invoice_agent.server
"""

__version__ = "1.0.0"
