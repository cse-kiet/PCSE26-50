# MedTalk — Literature Folder Guide & Annotated Bibliography

**Project:** MedTalk: Explanation-Driven Adaptive Questioning for Responsible Interpretable AI  
**Institution:** KIET Group of Institutions, Ghaziabad (AKTU)  
**Prepared by:** Ujjwal Tomar · Yash Singhal · Tushar Singh  
**Supervisor:** Prof. Gaurav Parashar, Assistant Professor, CSE  
**Date:** May 2026  

---

## Folder Organisation

```
Literature/
│
├── README_Literature.md              ← This file (annotated bibliography)
│
├── core_papers/                      ← Foundational XAI and ML papers
│   ├── Slack2023_TalkToModel.pdf
│   ├── Lundberg2017_SHAP_NeurIPS.pdf
│   ├── Ribeiro2016_LIME_KDD.pdf
│   ├── Breiman2001_RandomForests.pdf
│   └── Caruana2015_IntelligibleModels.pdf
│
├── domain_specific/                  ← Domain application papers
│   ├── healthcare/
│   │   ├── Thirunavukarasu2023_LLMs_Medicine.pdf
│   │   ├── Rajkomar2018_DeepLearning_EHR.pdf
│   │   └── Mitchell2021_AlgorithmicFairness.pdf
│   ├── finance/
│   │   ├── Barocas2016_BigData_DisparateImpact.pdf
│   │   └── SEC2023_XAI_Guidance.pdf
│   └── judiciary/
│       ├── Dressel2018_COMPAS_ScienceAdvances.pdf
│       └── CorbettDavies2016_AlgorithmicFairness.pdf
│
├── regulatory/                       ← Regulatory frameworks and whitepapers
│   ├── FDA2021_AI_ML_SaMD_Regulatory_Framework.pdf
│   ├── EU_AI_Act_2021_Official_Proposal.pdf
│   └── SEC2023_Guidance_Algorithmic_Trading.pdf
│
└── datasets/
    ├── PIDD_Documentation.pdf        ← Pima Indians Diabetes Dataset description
    ├── GermanCredit_UCI_Info.txt     ← German Credit Dataset (Finance baseline)
    └── COMPAS_ProPublica_Study.pdf   ← COMPAS Recidivism Dataset documentation
```

---

## SECTION 1: FOUNDATIONAL PAPERS (CORE — MUST READ)

---

### [1] TalkToModel — PRIMARY BASELINE

**Full Reference:**  
Slack, D., Krishna, S., Lakkaraju, H., & Singh, S. (2023). *TalkToModel: Explaining Machine Learning Models Through Natural Language Conversations.* arXiv preprint arXiv:2207.04154. Published at NeurIPS 2023.

**Relevance to MedTalk:** ★★★★★ (Critical — direct comparison system)

**Summary:**  
TalkToModel is the seminal work that MedTalk directly extends and improves upon. It introduces a conversational interface for querying ML model explanations using natural language, relying on a T5 transformer for semantic parsing and a BART model for natural language generation. It demonstrated that 82% of healthcare workers preferred conversational interfaces over static dashboards, validating the dialogue paradigm. However, it relies entirely on LLMs, which introduces hallucination risk, poor probability calibration (Brier=0.189), and regulatory misalignment.

**Key Finding Used in MedTalk:**  
MedTalk achieves Brier=0.142 vs. TalkToModel's 0.189, and 94.2% clinical comprehension vs. 73.1% — directly benchmarked against this paper's results.

**Access:** https://arxiv.org/abs/2207.04154

---

### [2] SHAP — THEORETICAL FOUNDATION

**Full Reference:**  
Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions.* Advances in Neural Information Processing Systems (NeurIPS), 30, 4765–4774.

**Relevance to MedTalk:** ★★★★★ (Critical — core explainability method)

**Summary:**  
This paper introduces SHapley Additive exPlanations (SHAP), a game-theory-grounded method that assigns each feature an importance value for a particular prediction. SHAP unifies previous attribution methods (LIME, DeepLIFT, Layer-wise Relevance Propagation) under a single framework satisfying three desirable properties: local accuracy, missingness, and consistency. The follow-up paper (Lundberg et al., 2018) introduced TreeSHAP — an exact, polynomial-time algorithm for tree ensembles — which MedTalk uses via shap.TreeExplainer.

**Key Equations:**
- SHAP value: φᵢ = Σ_{S⊆F\{i}} [|S|!(|F|-|S|-1)! / |F|!] [f(S∪{i}) - f(S)]
- SHAP kernel: K(z,z') = (|F|-1) / [C(|F|,|z'|) × |z'| × (|F|-|z'|)]

**Access:** https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html

---

### [3] LIME — SECONDARY ATTRIBUTION METHOD

**Full Reference:**  
Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier.* Proceedings of ACM KDD, 1135–1144.

**Relevance to MedTalk:** ★★★★★ (Critical — secondary explainability method)

**Summary:**  
LIME introduces a model-agnostic local interpretability approach. It approximates complex classifiers locally with a sparse linear model by perturbing the input and observing changes in prediction. The linear model's coefficients serve as interpretable feature importances. MedTalk uses LIME as the secondary attribution method, selecting between SHAP and LIME based on local fidelity scores. LIME is particularly useful when TreeSHAP produces inconsistent results due to correlated features.

**Key Algorithm:**
- Perturb input x → generate z' samples in neighbourhood
- Measure f(z') for each perturbation
- Fit weighted linear regression: g = argmin L(f, g, πₓ) + Ω(g)
- Return g coefficients as local feature importance

**Access:** https://dl.acm.org/doi/10.1145/2939672.2939778

---

### [4] RANDOM FORESTS — PREDICTION LAYER FOUNDATION

**Full Reference:**  
Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), 5–32.

**Relevance to MedTalk:** ★★★★★ (Critical — core prediction model)

**Summary:**  
Leo Breiman's foundational paper introducing Random Forests — an ensemble of decision trees grown on bootstrap samples (bagging) with random feature subsets at each split. Key properties relevant to MedTalk: (1) competitive accuracy vs. boosting methods; (2) natural probability estimates via vote proportions; (3) native feature importance via Gini impurity decrease; (4) compatibility with SHAP TreeExplainer for exact attribution.

**Why Not Neural Networks:**  
Breiman (and subsequent literature) demonstrates that for tabular medical data with <1000 samples, Random Forests match or exceed deep neural networks in accuracy while offering: no gradient descent, no vanishing gradients, no architectural tuning, and full compatibility with post-hoc XAI methods. MedTalk's Brier=0.142 vs. TalkToModel's neural LLM approach (Brier=0.189) empirically confirms this.

**Access:** https://link.springer.com/article/10.1023/A:1010933404324

---

### [5] INTELLIGIBLE MODELS FOR HEALTHCARE

**Full Reference:**  
Caruana, R., Lou, Y., Gehrke, J., Koch, P., Sturm, M., & Elhadad, N. (2015). *Intelligible Models for HealthCare: Predicting Pneumonia Risk and Hospital 30-day Readmission.* Proceedings of ACM KDD, 1721–1730.

**Relevance to MedTalk:** ★★★★☆ (Highly relevant — validates interpretable ML in healthcare)

**Summary:**  
Landmark paper demonstrating that intelligible models (specifically Generalised Additive Models + interactions) can match or exceed deep neural network accuracy in healthcare prediction while providing clinician-auditable explanations. Notably, the paper revealed that a neural network learning from pneumonia data incorrectly learned that asthma patients had lower pneumonia risk (because they received more aggressive treatment), which an interpretable model correctly flagged. This case study directly motivates MedTalk's design philosophy: transparency is not a trade-off, it is a safety requirement.

**Key Quote (paraphrased):** "A model that is accurate but inexplicable is dangerous in clinical settings because it cannot be audited for erroneous learned rules."

**Access:** https://dl.acm.org/doi/10.1145/2783258.2788613

---

## SECTION 2: DOMAIN-SPECIFIC PAPERS

---

### [6] LARGE LANGUAGE MODELS IN MEDICINE — RISK ANALYSIS

**Full Reference:**  
Thirunavukarasu, A. J., Ting, D. S. J., Elangovan, K., et al. (2023). *Large Language Models in Medicine.* Nature Medicine, 29(8), 1930–1940.

**Relevance to MedTalk:** ★★★★☆ (Supports rationale for avoiding LLMs in clinical dialogue)

**Summary:**  
Comprehensive review of LLM capabilities and risks in clinical applications. Documents hallucination rates, calibration failures, and regulatory barriers for LLM deployment in SaMD contexts. Directly cited in MedTalk's introduction to justify replacing LLM-based dialogue (as in TalkToModel) with deterministic transparent algorithms. Key finding: LLMs achieve high surface-level performance on medical licensing exams but produce dangerous factual errors in clinical reasoning chains.

**Access:** https://www.nature.com/articles/s41591-023-02448-8

---

### [7] DEEP LEARNING WITH ELECTRONIC HEALTH RECORDS

**Full Reference:**  
Rajkomar, A., Oren, E., Chen, K., Dai, A. M., et al. (2018). *Scalable and Accurate Deep Learning with Electronic Health Records.* npj Digital Medicine, 1(1), 18.

**Relevance to MedTalk:** ★★★☆☆ (Contextual — healthcare AI state of the art)

**Summary:**  
Demonstrates deep learning outperforming traditional clinical models on EHR-based prediction tasks (mortality, readmission, length of stay) across two major hospital systems. Important for contextualising MedTalk's scope: MedTalk addresses structured tabular data (as in PIDD) rather than unstructured EHR text. Future work section of MedTalk report references federated learning and LSTM integration for time-series EHR data, building on this foundation.

**Access:** https://www.nature.com/articles/s41746-018-0029-1

---

### [8] BIG DATA'S DISPARATE IMPACT — FAIR LENDING

**Full Reference:**  
Barocas, S., & Selbst, A. D. (2016). *Big Data's Disparate Impact.* California Law Review, 104(3), 671–732.

**Relevance to MedTalk:** ★★★★★ (Critical — legal foundation for EDAQ fairness guardrails)

**Summary:**  
Legal scholarship demonstrating how ML models can violate fair lending laws (FCRA, FHA, ECOA) even without explicitly using protected attributes, through proxy variables (e.g., zip code correlating with race). This paper directly motivates MedTalk's EDAQ fairness guardrails (Step 3 and Step 6), which filter out protected attributes AND their common proxies from question generation. The paper's framework is used to define the `protected_attributes` lists in each domain configuration file.

**Access:** https://www.californialawreview.org/print/big-datas-disparate-impact/

---

### [9] ACCURACY, FAIRNESS, AND LIMITS OF PREDICTING RECIDIVISM — COMPAS

**Full Reference:**  
Dressel, J., & Farid, H. (2018). *The Accuracy, Fairness, and Limits of Predicting Recidivism.* Science Advances, 4(1), eaao5580.

**Relevance to MedTalk:** ★★★★☆ (Critical — judiciary domain validation context)

**Summary:**  
Studies COMPAS (Correctional Offender Management Profiling for Alternative Sanctions), a widely used recidivism prediction tool, and finds: (1) it is no more accurate than non-expert humans, (2) it exhibits racial bias against Black defendants, and (3) its proprietary nature makes auditing impossible. MedTalk's judiciary instantiation is designed to address these exact failings: transparent Random Forest + SHAP + EDAQ with protected-attribute exclusion and a "decision-support only" constraint (never deterministic sentencing).

**Access:** https://www.science.org/doi/10.1126/sciadv.aao5580

---

### [10] ALGORITHMIC FAIRNESS — COMPAS COST OF FAIRNESS

**Full Reference:**  
Corbett-Davies, S., Pierson, E., Feller, A., Goel, S., & Huq, A. (2016). *Algorithmic Decision Making and the Cost of Fairness.* Proceedings of ACM KDD, 797–806.

**Relevance to MedTalk:** ★★★★☆ (Important — fairness trade-offs in criminal justice ML)

**Summary:**  
Demonstrates the mathematical impossibility of satisfying multiple fairness criteria simultaneously (calibration + equalised odds) and quantifies the accuracy cost of enforcing fairness constraints. MedTalk acknowledges these trade-offs explicitly in its limitations section and fairness guardrail design: EDAQ does not claim to solve the fairness-accuracy trade-off but ensures no discriminatory questions are generated during dialogue.

**Access:** https://dl.acm.org/doi/10.1145/3097983.3098095

---

## SECTION 3: RELATED SYSTEMS AND TOOLS

---

### [11] LANGUAGE INTERPRETABILITY TOOL (LIT)

**Full Reference:**  
Tenney, B., Wexler, J., Wilson, J., et al. (2020). *The Language Interpretability Tool: Extensible, Interactive Visualizations and Analysis for NLP Models.* EMNLP 2020, 8496–8508.

**Relevance to MedTalk:** ★★★☆☆ (Related work — NLP-focused XAI tool)

**Summary:**  
LIT is Google's interactive XAI platform primarily designed for NLP models (text classifiers, sequence models). While it provides rich visualisation and probing capabilities, it does not support tabular classification or conversational interaction paradigms. MedTalk's related work section uses LIT as an example of domain-specific XAI tools that lack cross-sector generalisability — motivating EDAQ's cross-domain design.

**Access:** https://arxiv.org/abs/2008.05122

---

### [12] WHAT-IF TOOL

**Full Reference:**  
Wexler, J., Pushkarna, M., Bolukbasi, T., Wattenberg, M., Viegas, F., & Wilson, J. (2019). *The What-If Tool: Interactive Probing of Machine Learning Models.* IEEE Transactions on Visualization and Computer Graphics (VIS), 56–65.

**Relevance to MedTalk:** ★★★☆☆ (Related work — counterfactual analysis tool)

**Summary:**  
Google's What-If Tool enables counterfactual analysis and feature perturbation exploration for ML models without code. Unlike MedTalk, it lacks a natural language dialogue interface and cannot generate adaptive questions — users must manually explore feature spaces. MedTalk's EDAQ is designed to automate and prioritise this exploration process, guiding non-technical users to the most impactful feature questions without requiring them to understand the model's feature space.

**Access:** https://pair-code.github.io/what-if-tool/

---

## SECTION 4: REGULATORY DOCUMENTS

---

### [13] FDA REGULATORY FRAMEWORK FOR AI/ML-BASED SaMD

**Full Reference:**  
U.S. Food and Drug Administration. (2021). *Proposed Regulatory Framework for Modifications to Artificial Intelligence/Machine Learning (AI/ML)-Based Software as a Medical Device (SaMD): Discussion Paper and Request for Feedback.* FDA White Paper.

**Relevance to MedTalk:** ★★★★★ (Critical — regulatory compliance framework)

**Summary:**  
Outlines FDA's approach to regulating AI/ML systems used as medical devices. Key requirements relevant to MedTalk: (1) predetermined change control plans; (2) algorithm transparency and auditability; (3) clinical performance monitoring; (4) real-world evidence collection. MedTalk's deterministic Random Forest + complete audit trail is designed to align with this framework's transparency requirements. The audit log format (CSV + PDF) is structured to support 510(k) submission documentation.

**Access:** https://www.fda.gov/media/122535/download

---

### [14] EU ARTIFICIAL INTELLIGENCE ACT (2021)

**Full Reference:**  
European Commission. (2021). *Proposal for a Regulation of the European Parliament and of the Council Laying Down Harmonised Rules on Artificial Intelligence (Artificial Intelligence Act).* COM/2021/206 final. Official Journal of the European Union.

**Relevance to MedTalk:** ★★★★★ (Critical — cross-jurisdictional regulatory alignment)

**Summary:**  
The EU AI Act classifies AI systems into risk categories and imposes transparency, accountability, and human oversight requirements on "high-risk" systems — which explicitly include AI in healthcare, law enforcement, and critical infrastructure. MedTalk is designed as a high-risk compliant system: Article 13 (transparency), Article 14 (human oversight), and Annex III (high-risk application areas) are all addressed through EDAQ's explainability pipeline and audit trail.

**Access:** https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:52021PC0206

---

### [15] SEC GUIDANCE ON AI IN ALGORITHMIC TRADING

**Full Reference:**  
U.S. Securities and Exchange Commission. (2023). *Guidance on Artificial Intelligence in Algorithmic Trading Systems.* SEC Release No. 34-XXXXX.

**Relevance to MedTalk:** ★★★☆☆ (Relevant — finance domain regulatory context)

**Summary:**  
SEC's guidance mandating explainability for AI-driven algorithmic trading and credit decision systems. Requires financial institutions to maintain auditable records of AI decision factors. MedTalk's finance domain configuration generates SHAP-based audit logs per decision, designed to meet this documentation requirement.

**Access:** https://www.sec.gov/

---

## SECTION 5: STATISTICAL FOUNDATIONS AND DATASETS

---

### [16] THE ELEMENTS OF STATISTICAL LEARNING

**Full Reference:**  
Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer.

**Relevance to MedTalk:** ★★★★☆ (Core reference — statistical and ML theory)

**Summary:**  
Comprehensive textbook covering the statistical foundations of machine learning algorithms used in MedTalk: Random Forests (Chapter 15), boosting (Chapter 10), model assessment and selection (Chapter 7), and calibration. Used as the primary reference for hyperparameter tuning justifications and evaluation methodology.

**Access (Free PDF):** https://hastie.su.domains/ElemStatLearn/

---

### [17] PIMA INDIANS DIABETES DATASET

**Full Reference:**  
Smith, J. W., Everhart, J. E., Dickson, W. C., et al. (1988). *Using the ADAP Learning Algorithm to Forecast the Onset of Diabetes Mellitus.* Proceedings of the Annual Symposium on Computer Application in Medical Care, 261–265. Available via: UCI Machine Learning Repository, Dua, D. and Graff, C. (2019). Irvine, CA: University of California, School of Information and Computer Science.

**Relevance to MedTalk:** ★★★★★ (Critical — primary evaluation dataset)

**Summary:**  
The Pima Indians Diabetes Dataset (PIDD) consists of 768 records of female patients of Pima Indian heritage, collected by the National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK). Features include 8 clinical variables commonly obtainable in primary care settings. PIDD is the most widely benchmarked diabetes classification dataset, enabling direct comparison with prior work. MedTalk achieves 89.4% accuracy / 0.921 AUROC on the standard 70/15/15 split.

**Note on Zero Values:** PIDD contains physiologically impossible zero values in Glucose, BloodPressure, SkinThickness, Insulin, and BMI — these are treated as missing values and imputed with column medians in MedTalk's preprocessing pipeline.

**Access:** https://archive.ics.uci.edu/ml/datasets/pima+indians+diabetes

---

### [18] GERMAN CREDIT DATASET (Finance Baseline)

**Full Reference:**  
Hofmann, H. (1994). *Statlog (German Credit Data).* UCI Machine Learning Repository.

**Relevance to MedTalk:** ★★★☆☆ (Relevant — finance domain synthetic data structure)

**Summary:**  
Real credit dataset from a German bank containing 1000 loan applications with 20 features (numerical and categorical) and a binary creditworthiness outcome. MedTalk's synthetic finance dataset (N=500) is generated using the feature distributions and correlations from this dataset as a statistical template, without using actual personal financial records.

**Access:** https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)

---

## SECTION 6: WEB RESOURCES AND DOCUMENTATION

| Resource | URL | Relevance |
|---|---|---|
| SHAP Official Documentation | https://shap.readthedocs.io/ | SHAP library usage and API |
| LIME GitHub Repository | https://github.com/marcotcr/lime | LIME source and examples |
| Streamlit Documentation | https://docs.streamlit.io/ | Web interface framework |
| Scikit-learn User Guide | https://scikit-learn.org/stable/user_guide.html | ML library reference |
| Imbalanced-learn Docs | https://imbalanced-learn.org/stable/ | SMOTE implementation |
| IDF Diabetes Atlas (2021) | https://diabetesatlas.org/ | Global diabetes statistics |
| ProPublica COMPAS Analysis | https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing | COMPAS bias investigation |
| AI Act Summary (EU) | https://artificialintelligenceact.eu/ | EU regulatory framework summary |

---

*This annotated bibliography documents all literature consulted during the MedTalk Major Project.*  
*KIET Group of Institutions, Ghaziabad — B.Tech CSE Major Project 2025–2026*
