# The Lookout and the Captain public reproduction

Execution issue: https://github.com/kortexa-ai/legolm.paper/issues/3

The manuscript and evidence audit are prepared under
https://github.com/kortexa-ai/legolm-paper/issues/10. This work publishes an
independent `reproductions/lookout-captain/` package on `main` and the
matching PDF and complete LaTeX archive in `lookout-captain-paper/`.

The package contains the 232 original result records, their hashes, the
exact archived experiment scripts and dependencies, recovered input and
checkpoint identities, analysis, tests, and a locked environment. Its
README separates a CPU rebuild from new model training or inference and
lists the complete recorded suite. Checkpoint tensors are served outside
Git by the research site; upstream models and detectors retain their
upstream distribution paths and licenses.

Validate the exported package from a clean environment. Recompute its
analysis, rebuild the PDF, compare its bytes with the private release, and
build the source archive in a fresh directory. The release manifest binds
these files and the evidence inventory. Record delivery commits and
validation evidence in the issue.

Production publication is tracked separately in
https://github.com/kortexa-ai/research/issues/3.
