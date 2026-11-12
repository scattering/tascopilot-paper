# Diagram figures 1-7

The seven plain `.tikz` files in `diagrams/` are the exact graphic-code blocks
from Overleaf revision `a9173895c687129310797ec93b44e6e7b5f89dc0`. They
preserve the original labels, coordinates, styles, scales, and arrows. They
contain no figure floats, manuscript captions, body paragraphs, citations,
bibliography, or manuscript document class. Diagram labels are naturally
part of the graphics and are retained.

`diagrams/figure_sources.json` contains metadata, not encoded source code:
figure numbers, original source files/line ranges, full source-file hashes,
individual graphic-code hashes, and required font/library settings. The five
external figure inputs precede the two inline workflow graphics in the
manuscript's inclusion order.

| Figure | Drawing source | Original source at the pinned revision |
| --- | --- | --- |
| 1, TAS schematic | `diagrams/figure_01_triple_axis_schematic.tikz` | `figures/triple_axis_schematic_jeff_review_20261103.tex`, lines 3-53 |
| 2, semantic decision stack | `diagrams/figure_02_semantic_decision_stack.tikz` | `figures/semantic_decision_stack.tex`, lines 3-31 |
| 3, federated architecture | `diagrams/figure_03_federated_architecture.tikz` | `figures/federated_architecture.tex`, lines 3-33 |
| 4, MCP tool layer | `diagrams/figure_04_mcp_tool_layer.tikz` | `figures/mcp_tool_layer.tex`, lines 3-36 |
| 5, claim lifecycle | `diagrams/figure_05_claim_lifecycle.tikz` | `figures/claim_lifecycle.tex`, lines 3-35 |
| 6, source linkage | `diagrams/figure_06_paper_source_linkage.tikz` | `sections/03_workflows.tex`, lines 23-52 |
| 7, reduction graph | `diagrams/figure_07_reduction_dag.tikz` | `sections/03_workflows.tex`, lines 86-108 |

## Optional TeX build

This is separate from the Python plotting environment for Figures 8-11.
Only Python's standard library is used by the diagram build script. Install
a TeX distribution providing `pdflatex`, `kpsewhich`, the `standalone` class,
TikZ/PGF, `stix2`, `fontenc`, `anyfontsize`, and `xcolor`. Required TikZ
libraries are `arrows.meta`, `positioning`, `shapes.geometric`, `angles`,
`quotes`, and `calc`. Poppler's `pdftoppm` is optional for PNG previews.
The script does not install packages or access the network.

From this `reproducibility` directory:

```sh
python3 -B scripts/build_diagrams.py --check-dependencies
python3 -B scripts/build_diagrams.py --out outputs/diagrams
```

To build the seven figure-only PDFs and PNG previews:

```sh
python3 -B scripts/build_diagrams.py --render --out outputs/diagrams-with-previews
```

The output directory must be new. Paths are independent of the working
directory. Missing dependencies produce `status: not_available` and exit code
2 without creating output. Generated `.tex`, logs, auxiliary files, PDFs,
PNGs, and `diagram_build_result.json` remain under ignored `outputs/` by
default. No source checkout, Git history, manuscript, bibliography, TAS
plugins, or scientific model runtime is required.

The script verifies the package and graphic hashes before invoking TeX with
shell escape disabled. It accepts only the audited graphic-command set and
one bounded TikZ block per source. This is a regression boundary for the
known code, not a general-purpose security sandbox for arbitrary TeX. Do not
compile code supplied by retrieved documents or private corpus text.

## Rendering fidelity

The original journal class uses STIX2 with T1 encoding and unusual size
definitions: normal text 9/11.5 pt, `footnotesize` 7/9 pt, and `scriptsize`
9/9 pt (font size/baseline). The standalone wrapper preserves the settings
needed by these graphics; using standard article-class sizes would visibly
change the diagrams. All in-graphic styles/macros are retained in the exact
`.tikz` blocks. Only a two-point border and a standalone document wrapper are
added. The original `USG.cls` is not distributed.

`VERIFICATION.json` records the actual build/render checks and any visual
comparison performed. All seven were built with pdfTeX 1.40.29 (TeX Live
2026), rendered at 120 dpi, and visually inspected. Independently extracted
source blocks were also compiled with the original journal class in a
figure-only preview wrapper, without manuscript text. Figures 2-7 were
pixel-identical to those renders. Figure 1 retained the same drawing and
labels but differed by approximately 0.012 PDF points in horizontal placement
and crop width (same 557-by-207 raster dimensions; 879 differing pixels out
of 115,299). It is not claimed pixel-identical. The original graphic code is
unchanged; no coordinate adjustment was made to force raster equality.
This checks the figure graphics, not complete manuscript-page rendering.

These diagrams have no numerical calculation to
re-execute, and rebuilding them does not validate the system behaviors they
illustrate. Figure 1 is an illustrative instrument layout, not measured
BT-7 geometry or operating instructions. No scientific calculations or
Figure 8-11 inputs/scripts are changed by this addition.

## Review and edit

The drawing code is directly readable in the `.tikz` files. For an exploratory
edit, work on a separate copy rather than overwriting the verified source.
Intentional source changes require new graphic hashes, package checksums,
provenance, and tests; the strict build is designed to reject unrecorded
changes. Keep generated TeX and PDFs out of the delivered source inventory.
