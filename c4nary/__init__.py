"""c4nary — deterministic, offline, read-only security auditor for GGUF files.

The tool inspects a GGUF model's embedded Jinja2 chat template and metadata for
known-dangerous constructs (SSTI / sandbox-escape primitives) and can diff a
model against a known-good reference to detect tampering.

Hard invariants (see spec §7):
  1. Never render or execute a template or model. AST parse only.
  2. No network access anywhere.
  3. Read-only: never write to or modify input files.
  4. Deterministic: identical input -> identical output bytes.
  5. Every finding maps to a registered rule with a stable id.

It detects *risk indicators*; it does not prove a model is safe or malicious.
"""

__version__ = "0.2.0"

__all__ = ["__version__"]
