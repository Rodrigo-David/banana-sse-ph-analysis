# Climatic and Biotic Associations with Source–Sink Efficiency in Philippine Banana (Musa spp.)

## Overview
Complete dataset, analysis scripts, environment locks, and sensitivity outputs for the manuscript:  
*David, R. S. (2025). Climatic and Biotic Associations with Source–Sink Efficiency in Philippine Banana (Musa spp.) Production: A National-Scale Synthesis Using Mixed-Effects Models and Machine Learning.*

## Quick Start
1. Install dependencies:
   - R: `renv::restore()`
   - Python: `pip install -r scripts/Python/requirements.txt`
2. Run verification pipelines:
   - R: `Rscript scripts/R/verify_lmm_sem.R`
   - Python: `python scripts/Python/verify_banana_sse.py`
3. Outputs reproduce Tables 1–2, SHAP thresholds, and sensitivity diagnostics.

## Data Dictionary
See `data/data_dictionary.md` for variable construction, sources, and transformation notes.

## Integrity Verification
All critical files include SHA-256 hashes in `checksums.sha256`. Verify with:  
`shasum -a 256 -c checksums.sha256` (macOS/Linux) or `Get-FileHash` (Windows).

## Licensing
Data & outputs: CC BY 4.0 | Code: MIT License

## Citation
David, R. S. (2025). [Manuscript Title]. [Journal Name]. https://doi.org/[MANUSCRIPT DOI]