# Clinical Oncology Neural Triage

Official research implementation and benchmark data for:
**Calibrated Deep Neural Networks for High-Stakes Clinical Oncology Triage: Mitigating Asymmetric False-Negative Costs in Diagnostic Cytopathology**

Author: **Abdul Wasiu Adebayo** (Techspance & Nexford University)  
ORCID: [0009-0002-1681-5784](https://orcid.org/0009-0002-1681-5784) | [Google Scholar](https://scholar.google.com/citations?user=630XJ2IAAAAJ)


[![SSRN Preprint](https://img.shields.io/badge/SSRN-Abstract_7744859-blue.svg)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7744859)
[![DOI](https://img.shields.io/badge/DOI-10.2139%2Fssrn.7744859-blue.svg)](https://doi.org/10.2139/ssrn.7744859)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## Core Results
* **Sensitivity (Zero False Negatives):** 100.0% at optimal clinical threshold tau* = 0.34.
* **Clinical Loss Reduction:** -44.9% ($245.83 vs $445.83 per patient) relative to standard tau = 0.50.
* **Calibration:** Platt sigmoid scaling achieves Brier loss of 0.0209 and PR-AUC of 0.9981.

## Quick Start
```bash
pip install -r requirements.txt
python scripts/run_clinical_experiment.py
```


## Citation

```bibtex
@article{adebayo2026clinical_oncology,
  title={Calibrated Deep Neural Networks for High-Stakes Clinical Oncology Triage: Mitigating Asymmetric False-Negative Costs in Diagnostic Cytopathology},
  author={Adebayo, Abdul Wasiu},
  journal={SSRN Electronic Journal},
  year={2026},
  doi={10.2139/ssrn.7744859},
  url={https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7744859}
}
```
