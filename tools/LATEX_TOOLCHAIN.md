# Report compilation toolchain

The report uses the **unmodified IEEEtran v1.8b class**, dated 2015-08-26, in `conference` mode. Do not override the class margins or font sizes. The official CTAN package identifies v1.8b and LPPL 1.3: <https://ctan.org/pkg/ieeetran>.

`report/IEEEtran.cls` was downloaded from <https://mirrors.ctan.org/macros/latex/contrib/IEEEtran/IEEEtran.cls>. SHA-256: `da751920a317ed318b7b5cd7fa585a6cc7d28502d457856382e9be24b10a3bd7`. Its complete original contribution and license notices remain intact. `tools/IEEEtran-LPPL.txt` contains the LPPL text downloaded from <https://mirrors.ctan.org/macros/latex/base/lppl.txt>.

## Preserve IEEEtran typography under Tectonic

Begin the source with:

```tex
\RequirePackage[T1]{fontenc}
\documentclass[conference]{IEEEtran}
```

Tectonic uses XeTeX, whose default TU encoding does not provide `TUptm.fd`; an unqualified IEEEtran document then silently substitutes Latin Modern for its intended Times family. Loading the standard T1 encoding before the class ensures IEEEtran's own Times (`ptm`) font selection is available while the class calculates its dimensions. This sets an encoding, not a custom font or margin. The smoke PDF font list and log verify the resulting Times-compatible fonts and absence of font substitutions. Use the same source prefix with conventional pdfLaTeX.

## Compile from the project root

Install [Tectonic 0.17.0](https://github.com/tectonic-typesetting/tectonic/releases/tag/tectonic%400.17.0) for your operating system, or download a project-local binary on macOS/Linux:

```sh
python3 tools/setup_tectonic.py
sh tools/compile_latex.sh report/report.tex output/pdf
```

The setup script verifies the SHA-256 digest published in GitHub's official release metadata before extracting only the compiler binary. It supports Apple Silicon/Intel macOS and x86_64/aarch64 Linux. On Windows, install the official release and run `tectonic --keep-logs --keep-intermediates report.tex` from the `report` directory.

Tectonic automatically repeats LaTeX/BibTeX passes as necessary. It downloads required TeX support files on the first run, so Internet access is required until packages are cached. The wrapper keeps caches under `tools/tectonic-cache` inside this project. See the [official cache documentation](https://tectonic-typesetting.github.io/book/latest/getting-started/first-document.html). Additional packages used by the report may require further downloads. Tectonic is MIT-licensed; `tools/TECTONIC-LICENSE` is the upstream notice.

The original report can also be compiled with a normal TeX Live/MacTeX installation using `pdflatex` (and `bibtex` if external bibliography files are used), without altering the source's standard IEEEtran settings. That alternative is documented for portability but was not the locally tested engine.

## Local verification

The environment check source is `tools/latex-smoke/minimal.tex`, compiled using `tools/tectonic` version 0.17.0 for aarch64-apple-darwin. The compiler was downloaded from <https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-aarch64-apple-darwin.tar.gz>; verified archive SHA-256: `a3f1cac7c5678f01661a92212f58480ae3b0634115d880dbc59e2953ded45667`.

The smoke document checks IEEEtran conference mode, graphicx, amsmath, and booktabs. It is infrastructure evidence, not a submission report. Exclude the downloaded compiler, archive, support-file cache, release metadata, and smoke-test files from the submission ZIP; retain the setup script, wrapper, toolchain documentation, licenses, and IEEEtran class.
