# Slides verification

The local slides PDF contains 17 pages at 960 × 540 points (16:9). Every slide was rendered and visually reviewed, including individual checks of the dense architecture, learning-curve, confusion-matrix, and reference slides. No clipping, overlapping text, missing glyphs, or out-of-page text was found in the final version.

The main mean/SD table, all six neural seed scores, paired changes, class-level results, and selected confusion counts derive from the verified CSV/JSON files. The build script checks layout bounds and the final PDF text was checked for the reported central values. Scientific charts use saved histories/class metrics. The four error images are original test examples, with their saved true and predicted classes. Their descriptions are marked exploratory.

No frozen experiment source, configuration, model, or numerical result was modified. Run timings are explicitly qualified as sequential observations that also reflect system load. Team names and actual contribution details remain for the team to complete. Recording, individual participation evidence, viewing access, and submission remain pending.

Rebuild using `.venv/bin/python scripts/build_slides.py` after the dataset is available. Slide titles are indexed in `docs/slides_index.md`.
