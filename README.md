# Clinical Oncology Neural Triage

Official research implementation and benchmark data for:
**Calibrated Deep Neural Networks for High-Stakes Clinical Oncology Triage: Mitigating Asymmetric False-Negative Costs in Diagnostic Cytopathology**

Author: **Abdul Wasiu Adebayo** (Techspance & Nexford University)  
ORCID: [0009-0002-1681-5784](https://orcid.org/0009-0002-1681-5784) | [Google Scholar](https://scholar.google.com/citations?user=630XJ2IAAAAJ)

## Core Results
* **Sensitivity (Zero False Negatives):** 100.0% at optimal clinical threshold tau* = 0.34.
* **Clinical Loss Reduction:** -44.9% ($245.83 vs $445.83 per patient) relative to standard tau = 0.50.
* **Calibration:** Platt sigmoid scaling achieves Brier loss of 0.0209 and PR-AUC of 0.9981.

## Quick Start
```bash
pip install -r requirements.txt
python scripts/run_clinical_experiment.py
```
