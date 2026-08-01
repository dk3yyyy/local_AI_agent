# Security policy

## Supported version

Security fixes target the latest release and the `main` branch.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Use GitHub's private
security-advisory flow for this repository so the report can be investigated
before details are disclosed.

Do not include API keys, model credentials, private review data, or other
secrets in a report.

## Runtime boundary

Local AI Agent runs Chroma in embedded, process-local mode. It does not start
or expose a Chroma HTTP server. Ollama is expected to bind to localhost unless
the operator deliberately configures a different endpoint.

Treat imported CSV files and model output as untrusted. The application parses
CSV data, validates its schema, escapes it into structured documents, validates
model citations against retrieved source IDs, and does not execute model or CSV
content as code.

## Accepted dependency advisory

CI ignores **CVE-2026-45829 / PYSEC-2026-311** for `chromadb` because the
advisory applies to Chroma's unauthenticated HTTP collection endpoint with
`trust_remote_code=true`. This application uses embedded Chroma and never
exposes that endpoint. As of 2026-07-31, pip-audit reports no fixed release.

This is a scoped risk acceptance, not a claim that the dependency is generally
safe. The exception must be removed when either:

1. a fixed compatible Chroma release becomes available; or
2. the application begins exposing a Chroma server or accepting remote Chroma
   endpoints.

The dependency audit still fails CI for every other known vulnerability.