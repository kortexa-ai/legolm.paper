# Paper build

This directory contains the short paper for the frozen Qwen 3.6 causal
expression reproduction.

Build from this directory:

```bash
tectonic main.tex
```

or with a standard TeX installation:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The paper reads its figures from the bundled confirmatory artifact under
`../reproductions/causal-expression/results/`.
