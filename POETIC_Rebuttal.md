---
title: POETIC-Rebuttal

---

Rebuttal: POETIC

R1: Reviewer 9M95

We sincerely appreciate the time and effort you dedicated to reviewing our paper. In the following response, we aim to address your main concern and provide further clarification.

> Q1: Range of property values

A1: Thank you for pointing this out. To fully address your request regarding the property settings, we provide the detailed descriptive statistics (Min, Max, Mean, Std) for all six properties used in our experiments in the table below. We will include this table in the Appendix of the revised paper.

| Property           | Symbol                   | Unit                              | Min    | Max    | Mean  | Std  |
|:------------------ |:------------------------:|:---------------------------------:|:------:|:------:|:-----:|:----:|
| **Polarizability** | $\alpha$                 | $\text{Bohr}^3$                   | 6.31   | 196.62 | 75.28 | 8.17 |
| **HOMO-LUMO Gap**  | $\Delta\epsilon$         | eV                                | 1.24   | 13.10  | 6.98  | 1.51 |
| **HOMO Energy**    | $\epsilon_{\text{HOMO}}$ | eV                                | -11.20 | -1.56  | -6.54 | 0.60 |
| **LUMO Energy**    | $\epsilon_{\text{LUMO}}$ | eV                                | -5.16  | 4.17   | 0.44  | 1.22 |
| **Dipole Moment**  | $\mu$                    | Debye (D)                         | 0.00   | 29.56  | 2.70  | 1.53 |
| **Heat Capacity**  | $C_v$                    | $\frac{\text{cal}}{\text{mol K}}$ | 6.26   | 46.78  | 31.45 | 4.06 |

> Q2: Sampling Distribution for Controllable Generation

A2: Thank you for your question. To comprehensively evaluate our model, we designed two distinct testing protocols targeting **controllability** (in-distribution) and **generalizability** (out-of-distribution/unseen), respectively. The sampling distributions for each are set as follows:

1. **For Controllability (In-distribution):**
   To evaluate the model's precision in realistic molecular design scenarios, we sampled **10,000 target property values** from the **empirical marginal distribution** of the dataset.
   
   * **Rationale:** This sampling strategy ensures that the target conditions correspond to physically realizable and chemically meaningful values that lie within the high-density regions of the chemical space.
   * **Evaluation:** We calculate the Mean Absolute Error (MAE) to assess how well the generated molecules align with these realistic targets. Additionally, to ensure that sampling from the high-density regions does not lead to mere memorization, we strictly evaluated the **Novelty** of the generated molecules. In the revised manuscript (Appendix E.1, Table 5), we have added comparisons with baseline, demonstrating that our method generates novel structures at a competitive rate, effectively ruling out memorization.

2. **For Generalizability (Unseen/Out-of-distribution):**
   To assess the model's ability to extrapolate to property ranges not seen during training, we adopt a specific data partitioning protocol. We sort the dataset based on property values and partition it into three segments: the **lower 10%**, the **middle 80%**, and the **upper 10%**.
   
   * **Training:** The model is trained *exclusively* on the **middle 80%** of the data.
   * **Testing:** We sample "desired" target values from the withheld **lower 10%** and **upper 10%** tails. This protocol strictly evaluates the model's capability to generalize to out-of-distribution regions.

> Q3: how does your property predictive model perform?

A3: We apologize for not making the definition of the "Data" baseline sufficiently clear in the main text. The row labeled **"Data"** in Table 1 explicitly reports the performance (MAE) of our property predictive model.

**1. Experimental Setup and Data Splitting:**
To ensure a rigorous and fair evaluation, we followed the **EDM protocol** (Hoogeboom et al., 2022) described in **Section 4.1**. We partitioned the training data (100K samples) into two disjoint halves:

* **first half (50K):** Used exclusively to train the **EGNN property predictor** (the evaluator).
* **second half (50K):** Used exclusively to train our **Generative Model (POETIC)**.
  We apologize for the confusion. The row labeled **"Data"** in Table 1 explicitly reports the test performance (MAE) of our property predictive model.
  To strictly **prevent data leakage** and ensure a rigorous evaluation, we followed the EDM [1] protocol by partitioning the training dataset (100K samples) into two disjoint halves. The **First Half (50K)** was used exclusively to train the EGNN property predictor (evaluator), while the **Second Half (50K)** was used to train our Generative Model (POETIC).
  Consequently, the **"Data"** row reports the Mean Absolute Error (MAE) of the pre-trained EGNN evaluated on the **Second Half**. Since the predictor never saw these samples during its training, this metric effectively quantifies the inherent generalization error of the evaluator on the specific data distribution used for generative modeling.

**Quantitative Performance:**
The specific MAE performance of the predictor (the "Data" baseline) is listed below:

| Metric            | $\alpha$ ($\text{Bohr}^3$) | $\Delta\epsilon$ (meV) | $\epsilon_{\text{HOMO}}$ (meV) | $\epsilon_{\text{LUMO}}$ (meV) | $\mu$ (D) | $C_v$ ($\frac{\text{cal}}{\text{mol K}}$) |
|:----------------- |:--------------------------:|:----------------------:|:------------------------------:|:------------------------------:|:---------:|:-----------------------------------------:|
| **Predictor MAE** | 0.10                       | 64                     | 39                             | 36                             | 0.043     | 0.040                                     |

These results confirm that the predictor provides accurate guidance for the reinforcement learning process and serves as a reliable metric for evaluation.

[1] EDM: Equivariant diffusion for molecule generation in 3d. ICML 2022.

> Q4: why a Mamba model is used, instead a regular transformer model?

A4: Thank you for this insightful question. Initially, we selected the Mamba architecture primarily for its theoretical advantages in handling long sequences, which is critical for our task involving fine-grained 3D coordinate tokenization and retrieval-augmented prefixes. To empirically validate this design choice, we extended our experiments by training a variant of POETIC that replaces the Mamba backbone with a standard 12-layer GPT, while keeping all other hyperparameters and training data identical. The performance comparison is summarized in the table below:

| Backbone Architecture   | $\alpha$ ($\text{Bohr}^3$) | $\Delta\epsilon$ (meV) | $\epsilon_{\text{HOMO}}$ (meV) | $\epsilon_{\text{LUMO}}$ (meV) | $\mu$ (D) | $C_v$ ($\frac{\text{cal}}{\text{mol K}}$) |
|:----------------------- |:--------------------------:|:----------------------:|:------------------------------:|:------------------------------:|:---------:|:-----------------------------------------:|
| POETIC (with GPT)       | 0.35                       | 92                     | 78                             | 58                             | 0.117     | 0.134                                     |
| **POETIC (with Mamba)** | **0.21**                   | **62**                 | **39**                         | **27**                         | **0.080** | **0.077**                                 |

As the results demonstrate, the Mamba-based model significantly outperforms the GPT variant across all six properties. We attribute this superiority to Mamba's efficiency in long-sequence modeling. In our framework, the combination of coordinate-level tokens and detailed RAG prefixes results in extended sequence lengths. Mamba's State Space Model (SSM) architecture captures the complex, long-range dependencies inherent in these 3D geometric sequences more effectively than the standard attention mechanism used in GPT, leading to more precise structural generation and property alignment.

> Q5: how are the six properties picked for QM9 dataset?

Thank you for the question regarding our evaluation metrics. We selected these specific six quantum properties ($\alpha$, $\Delta\epsilon$, $\epsilon_{\text{HOMO}}$, $\epsilon_{\text{LUMO}}$, $\mu$, $C_v$) based on two primary considerations:

- First, we strictly followed the standard evaluation protocol established by the prior works (EDM [1], GeoLDM [2], Geo2Seq [3]). Utilizing this standardized set of properties ensures that our results are directly comparable with the state-of-the-art, providing a fair and rigorous assessment of our method's contribution.

- Second, these properties were chosen because they are quantum chemical descriptors inherently dependent on precise 3D geometry (conformation), rather than just 2D topology [4]. Unlike common drug discovery metrics such as Solubility (LogP) or QED, which are largely determined by molecular connectivity, the selected properties are derived from electron density distributions defined by specific atomic coordinates. Therefore, they serve as strictly more rigorous benchmarks for 3D controllable generation, as they require the model to capture subtle geometric variations to achieve high accuracy.

[1] EDM: Equivariant diffusion for molecule generation in 3d. ICML 2022.
[2] GEOLDM: Geometric Latent Diffusion Models for 3D Molecule Generation. ICML 2023.
[3] Geo2Seq: Geometry Informed Tokenization of Molecules for Language Model Generation. ICML 2025.
[4] Quantum chemistry structures and properties of 134 kilo molecules. Scientific data 2014.

> Q6: Dataset and property in toy experiment

A6: Thank you for the question. The toy experiment presented in Figure 1, we utilized the QM9 dataset, consistent with the setup in our main experiments. Specifically, we focused on the property of Polarizability ($\alpha$). To efficiently verify the feasibility of the proposed framework, we evaluated the models on a subset of 200 target property values for the in-distribution and unseen settings, respectively.

> Q7: Rationale for structural embeddings in retrieval

A7: Thank you for the question. We chose these specific 3D descriptors based on three considerations: 3D geometric necessity, empirical superiority over 2D representations, and computational efficiency.

- **3D geometric necessity**. Our rationale for selecting element frequencies and interatomic distances in Eq. 2 stems from the fundamental requirement of 3D molecule generation, where the target properties are intrinsically governed by 3D conformations rather than just 2D topological connectivity. Unlike 1D SMILES strings or 2D graph fingerprints (e.g., ECFP) which are invariant to conformational changes, our task requires an SE(3)-invariant representation that can explicitly distinguish spatial structures. Consequently, we adopt interatomic distance statistics as the core structural embedding, a choice that is strongly supported by established paradigms in machine learning for quantum chemistry. For instance, Rupp et al. demonstrated that pairwise internuclear distances (formalized as the Coulomb Matrix) serve as minimal sufficient statistics for accurately predicting quantum mechanical properties [1]. Similarly, Hansen et al. introduced the "Bag of Bonds" representation, which utilizes histograms of interatomic distances to capture geometric distributions [2]. Our design in Eq. 2  aligns directly with these physics-informed principles, ensuring that the retrieval process provides precise, property-relevant geometric guidance that topological descriptors cannot offer.
- 
- 
