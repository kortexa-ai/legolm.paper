# Paper build

This directory contains the research note for the standalone Qwen 3.6
J-space-spectrum reproduction.

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

Figures are read from the bundled confirmatory artifact under
`../reproductions/jspace-spectrum/results/`. The dense extension and both HTML
replays are in the same results tree.
