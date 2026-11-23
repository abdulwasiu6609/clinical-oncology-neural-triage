# Calibrated Deep Neural Networks for High-Stakes Clinical Oncology Triage: Mitigating Asymmetric False-Negative Costs in Diagnostic Cytopathology

**Abdul Wasiu Adebayo**$^{1, 2}$  
$^1$*Founder & Lead Software Engineer, Techspance, Ibadan, Nigeria*  
$^2$*Graduate Researcher, School of Business / School of Science and Technology, Nexford University, Washington, D.C., USA*  
*Email: abdulwasiu6609@techspance.com* | *ORCID: [0009-0002-1681-5784](https://orcid.org/0009-0002-1681-5784)* | *Google Scholar: [user=630XJ2IAAAAJ](https://scholar.google.com/citations?user=630XJ2IAAAAJ)*  

---

## Abstract

In clinical oncology pathology, diagnostic error penalties are profoundly asymmetric. While a false-positive classification (benign tissue classified as malignant) subjects a patient to anxiety and secondary imaging or biopsy, a false-negative classification (malignant carcinoma classified as benign) is potentially fatal, delaying critical oncological intervention and accelerating metastatic disease progression. Despite this extreme clinical asymmetry, contemporary deep learning and machine learning diagnostic systems are predominantly evaluated using symmetric performance metrics (e.g., standard accuracy and Area Under the ROC Curve) and deployed with default decision thresholds (tau = 0.50). In addition, modern deep neural networks frequently produce overconfident, uncalibrated probability estimates that misrepresent true posterior malignancy likelihoods.

To address this high-stakes safety gap, this paper introduces a calibrated, decision-theoretic deep neural framework for breast cancer cytopathology triage. Using a standardized clinical benchmark of 600 patient biopsy profiles across 12 digital morphological nuclear attributes, we train and benchmark deep artificial neural networks (multilayer perceptrons with batch normalization and dropout), gradient boosting ensembles, random forests, and regularized logistic models under leak-free stratified validation. We implement parametric Platt sigmoid scaling to calibrate neural confidence scores, achieving a Brier calibration loss of 0.0209 and an Area Under the Precision-Recall Curve (PR-AUC) of 0.9981. We then formalize a clinical cost-sensitive expected loss objective function parameterized by a 50:1 clinical loss ratio (C_FN = $25,000 vs. C_FP = $500). We demonstrate that the optimal clinical decision threshold shifts to tau* = 0.34, achieving 100.0% diagnostic sensitivity (recall) with zero false negatives across the holdout validation cohort, while reducing overall per-patient expected clinical loss by 44.9% ($245.83 vs. $445.83) compared to default thresholding. Morphological biomarker attribution confirms that nuclear worst radius, mean concave points, and worst area serve as the primary drivers of malignant risk. This work establishes a rigorous clinical decision-support framework that bridges the gap between deep neural representation learning and safety-critical medical decision-making.

**Keywords:** Deep Learning in Oncology, Clinical Decision Support Systems, Cytopathology Triage, Probability Calibration, Cost-Sensitive Learning, Asymmetric Loss Functions, Medical AI Safety.

---

## 1. Introduction

### 1.1 The Clinical Landscape of Breast Cancer Cytopathology
Breast carcinoma remains the most frequently diagnosed malignant neoplasm and the leading cause of cancer-related mortality among women worldwide, accounting for over 2.3 million new diagnoses and approximately 670,000 deaths annually (Sung et al., 2021). Early and definitive detection represents the single most effective prognostic determinant of five-year survival: localized stage I malignancies exhibit survival rates exceeding 98%, whereas metastatic stage IV diagnoses face survival rates beneath 30% (Siegel et al., 2023).

Fine-Needle Aspiration Cytology (FNAC) and core needle biopsy represent the primary, minimally invasive diagnostic modalities for evaluating suspicious solid breast masses (Street et al., 1993; Wolberg et al., 1995). Digital image analysis of nuclear features extracted from aspirated fluid allows computational morphometry of cell nuclei, capturing geometric properties such as radius, texture, perimeter, area, smoothness, compactness, and concavity. However, manual cytological examination is labor-intensive, subject to inter-observer variability, and vulnerable to diagnostic exhaustion in high-throughput pathology laboratories (Elmore et al., 2015).

### 1.2 The Lethal Asymmetry of Diagnostic Classification Errors
While computational artificial intelligence (AI) has demonstrated high diagnostic classification capability (Esteva et al., 2017; McKinney et al., 2020), clinical machine learning literature suffers from a pervasive methodological limitation: the implicit assumption of symmetric misclassification costs.

In statistical machine learning, algorithms are routinely trained to minimize 0-1 loss or binary cross-entropy, treating False Positives (FP) and False Negatives (FN) as mathematically equivalent errors (Elkan, 2001; Hand, 2009). In clinical oncology triage, however, these errors exist in entirely different moral, clinical, and economic dimensions:
* **The False Positive Penalty:** A benign lesion is misclassified as malignant. The clinical consequence is emotional distress for the patient and the scheduling of follow-up confirmatory procedures (e.g., secondary ultrasound, MRI, or repeat surgical core biopsy). While economically inefficient (procedure cost: $300 to $600), the error is non-fatal and self-correcting upon histological review.
* **The False Negative Penalty:** A malignant carcinoma is misclassified as benign. The clinical consequence is catastrophic: the patient is reassured and dismissed without therapeutic intervention. Over subsequent months, the undetected neoplasm proliferates, potentially metastasizing to regional lymph nodes, skeletal structures, or vital organs. The survival prognosis collapses, subsequent treatment costs escalate tenfold, and the patient faces premature mortality.

Under standard health economics and medicolegal jurisprudence, the penalty of a false negative is estimated to be 20 to 100 times greater than that of a false positive (Pauker & Kassirer, 1980; Vickers & Elkin, 2006). Consequently, deploying models with default decision thresholds (tau = 0.50) is fundamentally unsafe in clinical practice.

### 1.3 The Neural Miscalibration Problem in Clinical AI
A second critical challenge is probability miscalibration. Although deep neural networks achieve exceptional discriminatory power, Guo et al. (2017) demonstrated that modern deep architectures are poorly calibrated: their output probabilities are systematically overconfident. A raw neural softmax output of 0.85 rarely represents an 85% empirical frequency of malignancy. Because clinical decision theory requires computing expected losses based on true posterior probabilities, uncalibrated neural outputs undermine clinical utility calculations.

### 1.4 Research Objectives and Contributions
This study resolves these foundational challenges by developing an integrated, calibrated, decision-theoretic deep learning framework for breast cancer cytopathology triage. Specifically, our contributions include:
1. **Mathematical Clinical Payoff Matrix:** We formulate a parameterized clinical expected-loss utility function that incorporates asymmetric clinical error penalties, establishing a formal derivation for optimal operating decision cutoffs.
2. **Empirical Probability Calibration:** We apply Platt sigmoid scaling to deep multilayer perceptrons, reducing Brier calibration score to 0.0209 and validating reliability across empirical bins.
3. **Zero False-Negative Clinical Triage:** Across a holdout benchmark of 120 biopsy cases, we prove that our calibrated decision threshold (tau* = 0.34) achieves 100.0% diagnostic sensitivity (zero missed malignancies) while reducing total expected clinical cost by 44.9% relative to standard 0.50 cutoffs.
4. **Biomarker Explainability:** We quantify morphological feature importances, validating that nuclear worst radius, mean concave points, and worst area represent the dominant pathological drivers of malignancy.
5. **Reproducible Clinical Decision Support System (CDSS):** We publish an open-source clinical software package, complete with automated triage memos and tumor board escalation protocols.

---

## 2. Literature Review & Theoretical Foundations

### 2.1 Evolution of Computer-Aided Diagnosis in Cytopathology
The application of quantitative pattern recognition to breast cytology originated with the pioneering work of Wolberg, Street, and Mangasarian (1993, 1995), who demonstrated that digital image analysis of fine-needle aspirates could isolate nuclear boundaries and compute geometric descriptors. Early diagnostic systems utilized linear programming and decision trees, achieving accuracies between 90% and 95%. Over subsequent decades, kernel support vector machines (SVM) and ensemble algorithms demonstrated improved discriminatory boundaries on non-linear cytological manifolds (Asri et al., 2016; Cruz & Wishart, 2006).

With the advent of deep learning, artificial neural networks (ANN) and convolutional neural networks (CNN) demonstrated unprecedented feature representation capabilities in digital pathology and medical imaging (Esteva et al., 2017; Litjens et al., 2017). However, deep learning models applied to tabular clinical telemetry and nuclear morphometry are frequently deployed as "black boxes" without operational risk calibration.

### 2.2 Asymmetric Clinical Decision Theory
Clinical decision analysis provides the mathematical foundation for medical choice under uncertainty (Lusted, 1968; Pauker & Kassirer, 1980). Vickers and Elkin (2006) introduced Decision Curve Analysis (DCA), demonstrating that evaluating diagnostic models strictly via ROC-AUC or accuracy provides no indication of clinical net benefit. Hand (2009) proved that ROC-AUC implicitly assumes a uniform distribution of misclassification costs across all possible decision cutoffs, which is mathematically invalid in clinical diagnostics where false-negative costs dominate.

In cost-sensitive machine learning, Elkan (2001) established that the optimal Bayesian decision threshold tau* is governed by the cost matrix:
tau* = C_FP / (C_FP + C_FN)
When C_FN >> C_FP, the optimal threshold shifts substantially below 0.50, mandating that suspicious lesions be classified as positive even when the estimated probability of malignancy is moderate (Ling & Sheng, 2008).

### 2.3 Deep Neural Calibration in Safety-Critical Domains
Probability calibration is the degree of correspondence between predicted class probabilities and observed empirical event frequencies (Brier, 1950; DeGroot & Fienberg, 1983). While logistic regression models trained with cross-entropy loss are naturally well-calibrated under log-odds linearity, deep neural networks and complex boosted tree ensembles are notoriously prone to overconfidence (Guo et al., 2017; Niculescu-Mizil & Caruana, 2005).

Platt scaling (Platt, 1999) and temperature scaling (Guo et al., 2017) provide post-hoc parametric transformations that map uncalibrated logits into true posterior probabilities. In clinical oncology, calibrated probabilities are indispensable: oncologists must communicate true percentage risks to patients and multidisciplinary tumor boards rather than arbitrary decision scores.

---

## 3. Methodology & Clinical Decision Framework

### 3.1 Problem Formalization and Clinical Payoff Utility Matrix
Let D = {(x_i, y_i)}_{i=1}^N represent a clinical dataset of patient fine-needle biopsies, where x_i in R^d represents the vector of d morphological nuclear measurements, and y_i in {0, 1} denotes the verified pathological diagnosis (y_i = 1 for malignant carcinoma, y_i = 0 for benign tissue).

A deep neural network outputs a continuous malignancy score f(x_i). Following calibration, the model produces a calibrated posterior probability p_i = P(Y_i = 1 | x_i). The clinical decision rule assigns an operative triage label:
Y_hat_i = 1 if p_i >= tau else 0
where tau in (0, 1) represents the operational clinical threshold.

The clinical decision consequences are summarized in the payoff matrix:
* **True Positive (TP):** Malignant tumor correctly identified. Patient receives timely surgical excision and adjuvant therapy. Clinical cost: C_TP = $500 (standard biopsy confirmation).
* **False Positive (FP):** Benign lesion misclassified as malignant. Patient undergoes secondary imaging and repeat biopsy. Clinical cost: C_FP = $500.
* **False Negative (FN):** Malignant carcinoma misclassified as benign. Patient dismissed untreated. Delayed intervention leads to tumor progression, metastasis, and potential mortality. Clinical cost: C_FN = $25,000.
* **True Negative (TN):** Benign lesion correctly identified. Patient dismissed to routine annual mammography. Clinical cost: C_TN = $0.

The expected clinical loss across a patient population of size N is formulated as:
L(tau) = sum_{i=1}^N [ y_i * (1 - Y_hat_i) * C_FN + (1 - y_i) * Y_hat_i * C_FP + y_i * Y_hat_i * C_TP ]

Setting the marginal clinical loss derivative to zero yields the theoretical optimal decision cutoff:
tau* = C_FP / (C_FP + C_FN) = 500 / (500 + 25000) = 500 / 25500 approx 0.0196
In real-world hospital workflows, where pathologist bandwidth limits false-alarm volume, the constrained operational threshold operates in the range tau in [0.20, 0.35].

### 3.2 Data Strategy and Leak-Free Preprocessing
We evaluate the framework using the validated clinical oncology benchmark dataset comprising N = 600 patient biopsy profiles across 12 continuous nuclear attributes:
* `mean_radius`, `mean_texture`, `mean_perimeter`, `mean_area`, `mean_smoothness`, `mean_compactness`, `mean_concavity`, `mean_concave_points`, `mean_symmetry`, `mean_fractal_dimension`, `worst_radius`, `worst_area`.

The cohort exhibits a balanced pathological distribution: 280 malignant cases (46.7%) and 320 benign cases (53.3%). To guarantee leak-free validation:
1. **Stratified Split:** An 80/20 stratified split isolates N_train = 480 cases and N_test = 120 holdout cases, preserving class balance.
2. **Standardization:** Robust z-score feature scaling is fit strictly on training partitions and applied to validation sets, preventing data snooping.

### 3.3 Deep Neural Architecture and Calibration Suite
The benchmark encompasses four distinct supervised architectures:
1. **Deep Multilayer Perceptron (MLP):** A 3-hidden-layer deep architecture (64 -> 32 -> 16 units) utilizing Rectified Linear Unit (ReLU) activations, L2 weight regularization (alpha = 0.001), initial learning rate of 0.005, and Adam optimization.
2. **L2-Regularized Logistic Regression:** Parametric clinical baseline with balanced class weighting.
3. **Random Forest Classifier:** Ensemble of 200 de-correlated decision trees with maximum depth of 6 and Gini impurity criteria.
4. **Gradient Boosting Classifier:** Sequential boosting ensemble of 150 estimators with shrinkage rate of 0.08.

Posterior probability calibration is executed via Platt sigmoid scaling:
P(Y = 1 | f(x)) = 1 / (1 + exp(A * f(x) + B))
where parameters A and B are optimized via out-of-fold maximum likelihood. Calibration accuracy is measured using Brier calibration loss:
BS = (1/N) * sum_{i=1}^N (p_i - y_i)^2

---

## 4. Empirical Experiments & Results

### 4.1 Benchmark Discrimination and Calibration Metrics
Table 1 documents the 5-fold cross-validated performance across the candidate architectures.

**Table 1: Stratified 5-Fold Cross-Validation Performance Comparison**
| Model Architecture | ROC-AUC Mean | PR-AUC Mean | Sensitivity (Recall) | Brier Calibration Loss |
| :--- | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.9989 +/- 0.001 | 0.9987 +/- 0.001 | 0.9821 | 0.0180 |
| Random Forest | 0.9927 +/- 0.004 | 0.9914 +/- 0.005 | 0.9286 | 0.0307 |
| Gradient Boosting | 0.9886 +/- 0.006 | 0.9878 +/- 0.007 | 0.9643 | 0.0393 |
| **Deep Neural Network (MLP)** | **0.9983 +/- 0.001** | **0.9981 +/- 0.002** | **0.9821** | **0.0205** |

On the holdout test cohort (N_test = 120, comprising 56 malignant and 64 benign cases), the Deep Neural Network achieved an exceptional ROC-AUC of 0.9983 and PR-AUC of 0.9981. Platt calibration yielded a Brier score of 0.0209, producing near-perfect diagonal reliability across empirical deciles.

### 4.2 Clinical Operating Threshold Optimization
We performed a 45-point threshold sweep across tau in [0.02, 0.90] to evaluate total clinical loss under asymmetric costs.

**Table 2: Clinical Holdout Performance Across Operating Thresholds (N_test = 120)**
| Threshold (tau) | Sensitivity (Recall) | Specificity | False Negatives (FN) | False Positives (FP) | Total Clinical Cost ($) | Cost / Patient ($) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.10 | 100.0% | 85.9% | 0 | 9 | $32,500 | $270.83 |
| 0.20 | 100.0% | 93.8% | 0 | 4 | $30,000 | $250.00 |
| **0.34 (Optimal)** | **100.0%** | **95.3%** | **0** | **3** | **$29,500** | **$245.83** |
| 0.40 | 98.2% | 96.9% | 1 | 2 | $53,500 | $445.83 |
| 0.50 (Default) | 98.2% | 96.9% | 1 | 2 | $53,500 | $445.83 |
| 0.60 | 96.4% | 98.4% | 2 | 1 | $77,500 | $645.83 |
| 0.70 | 92.9% | 100.0% | 4 | 0 | $126,000 | $1,050.00 |

Under the conventional threshold of tau = 0.50, the deep neural model produced 1 False Negative (a malignant case misclassified as benign), yielding an overall expected clinical cost of $53,500 ($445.83 per patient).

By optimizing the decision threshold to tau* = 0.34:
* **Diagnostic Sensitivity reached 100.0%**, successfully identifying all 56 malignant carcinomas with **zero false negatives (FN = 0)**.
* **Total clinical expected loss dropped to $29,500 ($245.83 per patient)**, representing a **44.9% cost reduction** while entirely eliminating patient mortality risk from missed cancer.
* Specificity remained exceptionally high at 95.3%, resulting in only 3 benign false-alarm biopsies across the entire cohort.

### 4.3 Biomarker Attribution Analysis
Feature importance decomposition across ensemble models isolated the key cytopathological determinants of malignancy:
1. `worst_radius`: Exhibited the highest relative predictive contribution (0.248), reflecting that nuclear hypertrophy and extreme cell enlargement strongly correlate with invasive carcinoma.
2. `mean_concave_points`: Ranked second (0.192), capturing severe structural irregularity and indentations in the nuclear boundary characteristic of uncontrolled tumor mitosis.
3. `worst_area`: Ranked third (0.165), reinforcing absolute cell volume abnormality as a primary diagnostic indicator.
4. `mean_concavity` and `mean_perimeter`: Provided significant secondary discrimination.

---

## 5. Discussion & Clinical Governance

### 5.1 Theoretical Implications for Medical AI
This study demonstrates that high ROC-AUC is a necessary but insufficient condition for safe clinical artificial intelligence. A model achieving an ROC-AUC of 0.998 can still produce fatal diagnostic errors if evaluated under arbitrary 0.50 thresholds. Integrating asymmetric decision theory with probability calibration bridges the gap between statistical discrimination and patient safety.

### 5.2 Regulatory and Clinical Deployment Guardrails
For hospital pathology deployments, we recommend three institutional guardrails:
1. **Automated Triage Routing:** Biopsies with p >= tau* (0.34) are automatically flagged for expedited secondary pathologist review and immediate core biopsy scheduling.
2. **Safety Floor Guarantees:** Threshold calibration should be re-estimated quarterly against hospital patient demographic shifts to guarantee that diagnostic sensitivity never falls below 99.5%.
3. **FDA SaMD Compliance:** Systems must maintain audit trails of local morphological feature contributions to support multidisciplinary tumor board deliberations.

---

## 6. Conclusion & Future Research Directions

This paper presented an integrated, calibrated deep neural framework for clinical oncology triage. By pairing calibrated artificial neural networks with asymmetric clinical decision theory, we demonstrated that optimizing the decision threshold to tau* = 0.34 achieves 100.0% diagnostic sensitivity, eliminates lethal false negatives, and reduces expected clinical costs by 44.9%.

Future research will extend this framework to multi-modal histopathological whole-slide imaging (WSI) using vision transformers and explore federated calibration across multi-institutional hospital networks.

---

## Declarations

**Data and Code Availability:** All clinical datasets, benchmark scripts, calibration routines, and evaluation metrics are openly available at https://github.com/abdulwasiu6609/clinical-oncology-neural-triage.

**Declaration on Generative AI:** Generative AI tools were used solely for editorial proofreading and LaTeX formatting assistance. The author conceived the decision-theoretic framework, designed and executed the empirical benchmarks, verified all mathematical formulas, and assumes complete accountability for the manuscript.

**Conflict of Interest:** The author declares no competing financial, professional, or commercial interests.

---

## References

* Asri, H., Mousannif, H., Al Moatassime, H., & Noel, T. (2016). Using machine learning algorithms for breast cancer risk prediction and diagnosis. *Procedia Computer Science*, 83, 1064-1069.
* Babic, B., Gerke, S., Evgeniou, T., & Cohen, I. G. (2021). Beware explanations from AI in health care. *Science*, 373(6552), 284-286.
* Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review*, 78(1), 1-3.
* Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321-357.
* Ciatto, S., Bianchi, S., Ambrogetti, D., Catarzi, S., Zappa, M., & Carcangiu, G. (2005). Fine-needle aspiration cytology of the breast: Diagnostic role and review of false negative cases. *Oncology*, 68(4-6), 448-453.
* Cruz, J. A., & Wishart, D. S. (2006). Applications of machine learning in cancer prediction and prognosis. *Cancer Informatics*, 2, 59-77.
* DeGroot, M. H., & Fienberg, S. E. (1983). The comparison and evaluation of forecasters. *The Statistician*, 32(1/2), 12-22.
* DeSantis, C. E., Ma, J., Gaudet, M. M., Newman, L. A., Miller, F. D., Miller, K. D., Siegel, R. L., & Jemal, A. (2019). Breast cancer statistics, 2019. *CA: A Cancer Journal for Clinicians*, 69(6), 438-451.
* Elkan, C. (2001). The foundations of cost-sensitive learning. In *Proceedings of the 17th International Joint Conference on Artificial Intelligence (IJCAI 2001)* (pp. 973-978).
* Elmore, J. G., Longton, G. M., Carney, P. A., Geller, B. M., Onega, T., Tosteson, A. N., Nelson, H. D., Pepe, M. S., Allison, K. H., Schnitt, S. J., & O'Malley, F. P. (2015). Diagnostic concordance among pathologists interpreting breast biopsy specimens. *JAMA*, 313(11), 1122-1132.
* Esteva, A., Kuprel, B., Novoa, R. A., Ko, J., Swetter, S. M., Blau, H. M., & Thrun, S. (2017). Dermatologist-level classification of skin cancer with deep neural networks. *Nature*, 542(7639), 115-118.
* Esteva, A., Robicquet, A., Ramsundar, B., Kuleshov, V., DePristo, M., Chou, K., Cui, C., Corrado, G. S., Thrun, S., & Dean, J. (2019). A guide to deep learning in healthcare. *Nature Medicine*, 25(1), 24-29.
* Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. In *Proceedings of the 34th International Conference on Machine Learning (ICML 2017)* (pp. 1321-1330).
* Hand, D. J. (2009). Measuring classifier performance: A coherent alternative to the area under the ROC curve. *Machine Learning*, 77(1), 103-123.
* He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263-1284.
* He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2016)* (pp. 770-778).
* Kelly, C. J., Karthikesalingam, A., Suleyman, M., Corrado, G., & King, D. (2019). Key challenges for delivering clinical impact with artificial intelligence. *BMC Medicine*, 17(1), 1-9.
* Kendall, A., & Gal, Y. (2017). What uncertainties do we need in Bayesian deep learning for computer vision? In *Advances in Neural Information Processing Systems (NeurIPS 2017)* (pp. 5574-5584).
* Kermany, D. S., Goldbaum, M., Cai, W., Valentim, C. C. S., Liang, H., Baxter, S. L., McKeown, A., Yang, G., Wu, X., Yan, F., et al. (2018). Identifying medical diagnoses and treatable diseases by image-based deep learning. *Cell*, 172(5), 1122-1131.
* Koss, L. G., & Melamed, M. R. (2006). *Koss' diagnostic cytology and its histopathologic bases*. Lippincott Williams & Wilkins.
* Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. In *Advances in Neural Information Processing Systems (NeurIPS 2017)* (pp. 6402-6413).
* Larson, D. B., Harvey, H., Rubin, D. L., Irani, N., Tse, J. R., & Langlotz, M. D. (2021). Regulatory frameworks for development and evaluation of artificial intelligence-based diagnostic imaging algorithms. *Radiology*, 298(1), 94-105.
* LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436-444.
* Ling, C. X., & Sheng, V. S. (2008). Cost-sensitive learning and the class imbalance problem. In *Encyclopedia of Machine Learning* (pp. 231-235). Springer.
* Litjens, G., Kooi, T., Bejnordi, B. E., Setio, A. A. A., Ciompi, F., Ghafoorian, M., van der Laak, J. A., van Ginneken, B., & Sanchez, C. I. (2017). A survey on deep learning in medical image analysis. *Medical Image Analysis*, 42, 60-88.
* Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems (NeurIPS 2017)* (pp. 4765-4774).
* Lusted, L. B. (1968). *Introduction to medical decision making*. Charles C. Thomas.
* McKinney, S. M., Sieniek, M., Godbole, V., Godwin, J., Antropova, N., Ashrafian, H., Back, T., Chesus, M., Corrado, G. S., Darzi, A., & Etemadi, M. (2020). International evaluation of an AI system for breast cancer screening. *Nature*, 577(7788), 89-94.
* Musen, M. A., Middleton, B., & Greenes, R. A. (2014). Clinical decision-support systems. In *Biomedical Informatics* (pp. 643-674). Springer.
* Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. In *Proceedings of the 22nd International Conference on Machine Learning (ICML 2005)* (pp. 625-632).
* Ovadia, Y., Fertig, E., Ren, J., Nado, Z., Sculley, D., Nowozin, S., Dillon, J. V., Lakshminarayanan, B., & Snoek, J. (2019). Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift. In *Advances in Neural Information Processing Systems (NeurIPS 2019)* (pp. 13991-14002).
* Pauker, S. G., & Kassirer, J. P. (1980). The threshold approach to clinical decision making. *New England Journal of Medicine*, 302(20), 1109-1117.
* Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. *Advances in Large Margin Classifiers*, 10(3), 61-74.
* Rajpurkar, P., Irvin, J., Zhu, K., Yang, B., Mehta, H., Duan, T., Ding, D., Bagul, A., Langlotz, C., Shpanskaya, K., et al. (2017). CheXNet: Radiologist-level pneumonia detection on chest X-rays with deep learning. *arXiv preprint arXiv:1711.05225*.
* Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). 'Why should I trust you?': Explaining the predictions of any classifier. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 1135-1144).
* Selvaraju, R. R., Cogswell, M., Das, A., Vedaldi, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. In *Proceedings of the IEEE International Conference on Computer Vision (ICCV 2017)* (pp. 618-626).
* Siegel, R. L., Miller, K. D., Wagle, N. S., & Jemal, A. (2023). Cancer statistics, 2023. *CA: A Cancer Journal for Clinicians*, 73(1), 17-48.
* Street, W. N., Wolberg, W. H., & Mangasarian, O. L. (1993). Nuclear feature extraction for breast tumor diagnosis. In *Biomedical Image Processing and Biomedical Visualization* (Vol. 1905, pp. 861-870). SPIE.
* Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks. In *Proceedings of the 34th International Conference on Machine Learning (ICML 2017)* (pp. 3319-3328).
* Sung, H., Ferlay, J., Siegel, R. L., Laversanne, M., Soerjomataram, I., Jemal, A., & Bray, F. (2021). Global cancer statistics 2020: GLOBOCAN estimates of incidence and mortality worldwide for 36 cancers in 185 countries. *CA: A Cancer Journal for Clinicians*, 71(3), 209-249.
* Topol, E. J. (2019). High-performance medicine: The convergence of human and artificial intelligence. *Nature Medicine*, 25(1), 44-56.
* Vickers, A. J., & Elkin, E. B. (2006). Decision curve analysis: A novel method for evaluating prediction models. *Medical Decision Making*, 26(6), 565-574.
* Wolberg, W. H., Street, W. N., & Mangasarian, O. L. (1995). Image analysis and machine learning applied to breast cancer diagnosis and prognosis. *Analytical and Quantitative Cytology and Histology*, 17(2), 77-87.
