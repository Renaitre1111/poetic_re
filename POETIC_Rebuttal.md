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
   To evaluate the model's precision in realistic molecular design scenarios, we sampled **10,000 target property values** from the **empirical marginal distribution** of the dataset. This sampling strategy ensures that the target conditions correspond to physically realizable and chemically meaningful values that lie within the high-density regions of the chemical space. We calculate the Mean Absolute Error (MAE) to assess how well the generated molecules align with these realistic targets. Additionally, to ensure that sampling from the high-density regions does not lead to mere memorization, we strictly evaluated the **Novelty** of the generated molecules.

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

| Metric | $\alpha$ ($\text{Bohr}^3$) | $\Delta\epsilon$ (meV) | $\epsilon_{\text{HOMO}}$ (meV) | $\epsilon_{\text{LUMO}}$ (meV) | $\mu$ (D) | $C_v$ ($\frac{\text{cal}}{\text{mol K}}$) |
|:----------------- |:--------------------------:|:----------------------:|:------------------------------:|:------------------------------:|:---------:|:-----------------------------------------:|
| **Predictor MAE** | 0.10 | 64 | 39 | 36 | 0.043 | 0.040 |

These results confirm that the predictor provides accurate guidance for the reinforcement learning process and serves as a reliable metric for evaluation.

[1] EDM: Equivariant diffusion for molecule generation in 3d. ICML 2022.

> Q4: why a Mamba model is used, instead a regular transformer model?

A4: Thank you for this insightful question. Initially, we selected the Mamba architecture primarily for its theoretical advantages in handling long sequences, which is critical for our task involving fine-grained 3D coordinate tokenization and retrieval-augmented prefixes. To empirically validate this design choice, we extended our experiments by training a variant of POETIC that replaces the Mamba backbone with a standard 12-layer GPT, while keeping all other hyperparameters and training data identical. The performance comparison is summarized in the table below:

| Backbone Architecture   | $\alpha$ ($\text{Bohr}^3$) | $\Delta\epsilon$ (meV) | $\epsilon_{\text{HOMO}}$ (meV) | $\epsilon_{\text{LUMO}}$ (meV) | $\mu$ (D) | $C_v$ ($\frac{\text{cal}}{\text{mol K}}$) |
|:----------------------- |:--------------------------:|:----------------------:|:------------------------------:|:------------------------------:|:---------:|:-----------------------------------------:|
| POETIC (with GPT) | 0.35 | 92 | 78 | 58 | 0.117 | 0.134 |
| **POETIC (with Mamba)** | **0.21** | **62** | **39** | **27** | **0.080** | **0.077** |

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

A7: Thank you for the question. We selected these specific 3D descriptors based on two primary considerations: theoretical necessity for 3D geometry and empirical superiority over 2D representations.

- **3D geometric necessity.** Our rationale for selecting element frequencies and interatomic distances in Eq. 2 stems from the fundamental requirement of 3D molecule generation, where the target properties are intrinsically governed by 3D conformations rather than just 2D topological connectivity. Unlike 1D SMILES strings or 2D graph fingerprints (e.g., ECFP) which are invariant to conformational changes, our task requires an SE(3)-invariant representation that can explicitly distinguish spatial structures. Rupp et al.[1] demonstrated that pairwise internuclear distances (formalized as the Coulomb Matrix) serve as minimal sufficient statistics for accurately predicting quantum mechanical properties, so we adopt interatomic distance statistics as the core structural embedding, a choice that is strongly supported by established paradigms in machine learning for quantum chemistry. Similarly, Hansen et al. [2] introduced the "Bag of Bonds" representation, which validates the utility of distance histograms and atomic distributions for capturing geometric and chemical variations. Our design in Eq. 2 aligns directly with these physics-informed principles, ensuring that the retrieval process provides precise, property-relevant geometric guidance that topological descriptors cannot offer.

- **Empirical Verification.** To empirically justify our choice over standard topological descriptors, we conducted an additional comparative experiment where the 3D structural retrieval in Eq. 2 was replaced by a 2D baseline using ECFP4 fingerprints with Tanimoto similarity. As shown in Table below, relying solely on 2D topological similarity consistently degrades performance across all six properties compared to our proposed 3D descriptors. Notably, the error for frontier orbital energies increases significantly, confirming that topological information alone is a suboptimal proxy for the 3D conformational features required for accurate quantum property targeting.

| Metric | $\alpha$ | $\Delta\epsilon$ | $\epsilon_{\text{HOMO}}$ | $\epsilon_{\text{LUMO}}$ | $\mu$ | $C_v$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| POETIC with ECFP4 | 0.23 | 68 | 51 | 49 | 0.110 | 0.110 |
| **POETIC (Ours)** | **0.21** | **62** | **39** | **27** | **0.080** | **0.077** |


[1] Fast and accurate modeling of molecular atomization energies with machine learning. Physical review letters, 2012.

[2] Machine learning predictions of molecular properties: Accurate many-body potentials and nonlocality in chemical space. The journal of physical chemistry letters, 2015.

> Q8: how is the normalized ... calculated?
 
A8: Thank you for the clarification. These statistics are computed by aggregating structural information from the retrieved exemplar set $\mathcal{N}$ (i.e., the top-$K$ molecules obtained from the retrieval stage) to capture the common structural characteristics of the target property:

- **Normalized Element Frequencies:** We sum the counts of each atom type (e.g., C, N, O) across all molecules in the exemplar set $\mathcal{N}$. These counts are then normalized by the total number of atoms in $\mathcal{N}$ to produce a probability distribution representing the average stoichiometric composition (e.g., `H:0.60,C:0.34,N:0.05,O:0.01`).
- **Most Prominent Distance Peaks ($\{[l_r, h_r]\}_r$):** We collect all pairwise interatomic distances from every molecule in $\mathcal{N}$ to construct a collective distance histogram. We then identify the histogram bins with the highest densities (local maxima) and select the intervals $[l_r, h_r]$ corresponding to the top-$k$ peaks (e.g., `[2.81, 2.97]`). These intervals serve as tokens to guide the model toward valid geometric conformations dominant in that property region.

> Q9: QM9 is a simple ... in further revisions.

A9: Thank you for the suggestion. QM9 serves as a standard benchmark for physical properties, but we acknowledge its limitations regarding molecule size. To address this and demonstrate the scalability of POETIC, we conducted two additional experiments on more practical datasets:

- **Alchemy Dataset Evaluation:** We extended our evaluation to the Alchemy dataset [1], which features significantly higher structural complexity than QM9. Specifically, Alchemy molecules contain up to 14 heavy atoms and cover a much broader and more diverse chemical space. The experimental settings were kept consistent with the main experiments reported in the paper. We compared POETIC directly against the strongest baseline, Geo2Seq with Mamba. As shown in the table below, POETIC maintains its superiority even on this more complex manifold, achieving lower MAE across six quantum properties compared to the baseline.

| Method | $\alpha$ | $\Delta\epsilon$ | $\epsilon_{\text{HOMO}}$ | $\epsilon_{\text{LUMO}}$ | $\mu$ | $C_v$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| data | 0.18 | 60 | 39 | 38 | 0.056 | 0.117 |
| Geo2Seq with Mamba | 1.85 | 445 | 603 | 296 | 0.603 | 0.110 |
| **POETIC (Ours)** | **1.32** | **162** | **155** | **66** | **0.167** | **0.077** |

- **ZINC250k with Practical Property.** To address the suggestion regarding "practical properties" like solubility, we further evaluated our framework on the ZINC250k dataset [2], focusing on LogP (Octanol-water partition coefficient). Since ZINC250k provides only 2D topologies without ground-truth 3D structures, we utilized RDKit to generate pseudo-3D conformers for training, and similarly employed RDKit as the oracle for both the reward model and evaluation. We compared POETIC with our strongest baseline, Geo2Seq with Mamba. 

| Metric | LogP |
| :--- | :---: |
| Geo2Seq with Mamba | 13.98 |
| **POETIC (Ours)** | 10.84 |

As shown in the table below, POETIC outperforms the baseline on this task. However, we note that the absolute MAE remains relatively higher compared to our quantum property experiments. We attribute this to two primary factors: 
1. Unlike QM9, which contains rigorous quantum-chemical coordinates (DFT-calculated), the "ground truth" 3D coordinates in ZINC250k are algorithmic approximations generated by RDKit. This introduces inherent noise and lacks the fine-grained geometric precision that our 3D-aware framework is designed to capture.
2. LogP is a topology-dominated property, typically calculated based on 2D fragment contributions rather than sensitive 3D conformational variations. In contrast, our framework is explicitly optimized for 3D molecule generation, excelling in tasks where properties (e.g., HOMO/LUMO) are strictly geometry-dependent. While our method generalizes to drug-like molecules, its primary advantage is best realized in scenarios requiring precise 3D structural control for quantum property targeting.

[1] Alchemy: A Quantum Chemistry Dataset for Benchmarking AI Models. arXiv:1906.09427

[2] ZINC 15 – Ligand Discovery for Everyone. Journal of Chemical Information and Modeling (JCIM), 2015.

> Q10: I think the proposed method ... RL and RAG.

A10: Thank you for your comment. We use the term "unified" to describe the deep algorithmic integration where reinforcement learning (RL) explicitly optimizes the utilization of retrieval-augmented generation (RAG) within a single conditional framework. In our pipeline, the RL policy is not merely appended after retrieval; rather, it is trained to generate molecules conditioned on the specific structural priors provided by the RAG prefix. This means the RL process is actively learning how to interpret and leverage the retrieved context to maximize property rewards. The gradient flow from the reward signal updates the model's attention to the retrieved exemplars, effectively "teaching" the language model to align its generation with the external structural guidance, which goes beyond a simple sequential combination of two independent modules.

Functionally, these components are mutually reinforcing, addressing the inherent trade-off between controllability and generalizability that neither can solve alone. As demonstrated in our ablation study (Table 3), RAG alone provides necessary structural priors for generalization but lacks precision (MAE 0.96), while RL alone ensures tight in-distribution alignment but suffers from overfitting (Unseen MAE 14.89) . By unifying them, POETIC creates a synergistic cycle where retrieval defines the valid chemical space (exploration) and RL enforces precise target adherence (exploitation). This organic integration allows the framework to achieve superior performance on both in-distribution controllability and out-of-distribution generalization, validating the "unified" design rationale .

> Q11: Fig. 4 does not show clear trend

A11: Thank you for pointing this out. We respectfully point out that Figure 4 illustrates a distinct and physically grounded trend regarding molecular geometry. Polarizability ($\alpha$) is intrinsically correlated with molecular volume and spatial extent: larger, more elongated molecules typically exhibit higher polarizability due to greater electron cloud distortion, while compact structures exhibit lower values.
As observed in Fig. 4:
- Low $\alpha$ region (e.g., 63.44 - 65.67): The model generates compact, globular structures (often with fused rings or clustered atoms), minimizing spatial volume.
- High $\alpha$ region (e.g., 86.47 - 91.76): The generated molecules systematically shift towards elongated, chain-like conformations with significantly larger spatial spans.

This progression from "compact" to "extended" perfectly aligns with physical intuition. It confirms that POETIC has successfully learned the complex, non-linear mapping between continuous quantum properties and 3D structural distributions, rather than generating random conformers.

> Q12: Additional qualitative study ... actually generated molecules.

A12: Thank you for the suggestion. To visually demonstrate the controllability and novelty of POETIC, we conducted a qualitative study covering three diverse quantum properties: Polarizability ($\alpha$), HOMO-LUMO Gap ($\Delta\epsilon$), and Dipole Moment ($\mu$). For each property, we randomly sampled two target values from the training set. The table below presents:

| Target Property | Generated Molecule | Predicted Property |
| :--- | :---: | :---: |
| $\alpha$=57.35 | O=C[C@@H]1C[C@@H]1C=O | 57.32 |
| $\alpha$=82.66 | C[C@H]1C[C@H]2C[C@]21[C@H]1CO1 | 82.45 |
| $\epsilon_{\text{HOMO}}$=-7.27 | N#C[C@H]1COCCCO1 | -7.34 |
| $\epsilon_{\text{LUMO}}$=-4.49 | CN(C)c1cc(N)co1 | -4.53 |
| $\mu$=4.31 | CCC[C@@]12CC(=O)N1C2 | 4.32 |
| $\mu$=1.76 | CC[C@@]12C[C@H]3OC[C@@H]1[C@H]32 | 1.76 |

This table lists randomly selected target property values (Desired) alongside the actually generated molecules (converted to SMILES for readability) and their predicted properties. As observed, the generated molecules exhibit properties that align precisely with the target values across different attributes (e.g., $\alpha$, $\mu$, $\epsilon_{\text{HOMO}}$). This offers concrete evidence that POETIC effectively translates specific numerical constraints into valid, property-matched chemical structures.

R2: Reviewer iroE

> Q1: Results for RAG-only and RL-only baselines.

A1: Thank you for your question. We have indeed conducted this ablation study in Section 4.3 (Table 3) of our manuscript. To provide a clear view of the synergy between these two components, we present the comparison of the RAG-only and RL-only variants below.

| Model Variant | Components | In-Distribution (Controllability) | Unseen Property (Generalization) |
| :--- | :---: | :---: | :---: |
| MLE | Baseline | 1.06 | 14.44 |
| MLE + RL | RL-only | 0.34 | 14.89 |
| RAG + MLE | RAG-only | 0.96 | 9.81 |
| POETIC | RAG + RL | 0.21 | 9.24 |

As shown in the table, RAG and RL play distinct yet complementary roles:
1. RL-only (MLE+RL): Reinforcement learning acts as a strong optimization engine, drastically improving in-distribution controllability by enforcing alignment with seen targets. However, without structural guidance, it tends to overfit the training distribution, leading to weaker extrapolation on unseen properties.
2. RAG-only (RAG+MLE): Retrieval augmentation introduces data-driven structural priors that serve as anchors. This significantly boosts generalization by grounding the generation in valid chemical space, even for unseen targets. Yet, without the explicit feedback from RL, it lacks the fine-grained pressure required for precise property matching.
3. Synergy (POETIC): By integrating both, POETIC achieves a robust balance. The retrieval mechanism provides a generalized structural context that prevents RL from overfitting, while RL fine-tunes these retrieved priors to ensure high-precision adherence to specific target values, effectively achieving the best.

This demonstrates that RAG and RL are not merely additive but synergistic: RAG provides the necessary "structural scaffold" for generalization, allowing RL to safely optimize for precision without catastrophic overfitting.

> Q2: In the RAG stage, have you considered ... similarity and diversity?

A2: Thank you for your insightful suggestion. We agree that clustering-based retrieval is an interesting direction. In this work, we adopted the "Prototype-based" retrieval strategy primarily to reduce variance in the conditioning signal and ensure high-fidelity guidance, based on the following considerations:

1. **Ensuring Signal Consistency:** The primary goal of our RAG module is to provide the Language Model with the most chemically "representative" structural motifs for a given property target. To achieve this, our prototype-based approach (Eq. 3) explicitly selects exemplars that align closest to the structural consensus of the candidate pool, thereby creating a stable and consistent reference for the LM to learn the core structure-property mapping. In contrast, while clustering increases retrieval diversity, it often introduces "outliers" or edge cases into the prompt. In the context of conditional generation, these inconsistent examples can act as noisy supervision, potentially confusing the model and degrading the precision of property alignment.
2. **Decoupling Retrieval and Generation Diversity:** We designed the system such that retrieval focuses on generalizability, while diversity is handled by the generative decoder. By feeding the model with high-quality, consistent prompts, we establish a strong baseline for validity and controllability. The generation diversity is then naturally achieved through the stochastic sampling (Top-k sampling) of the Language Model itself, allowing the model to explore variations around the optimal structural template.

Therefore, our choice was to prioritize the quality and consistency of the retrieval signal to maximize generation reliability. However, we acknowledge the value of clustering methods for broader exploration tasks and have added a discussion on this potential extension in the revised manuscript.

> Q3: The diversity metric of generated molecules.

A3: 

> Q4: Using a frozen EGNN property predictor and a fine-tuned version.

A4: Thank you for your question. In our experiment, we strictly employed a frozen EGNN property predictor rather than a fine-tuned version, based on two critical design considerations regarding generalization and optimization stability.

1. First, regarding the data setup, we partitioned the dataset into two disjoint halves. The EGNN was trained on the first half, but importantly, we used the second half (the generator's training set) as the test set to select the best EGNN checkpoint. This ensures that the frozen predictor we used had already achieved optimal predictive performance on the generator's data distribution, rendering further fine-tuning redundant and potentially risky for overfitting.
2. Second, and more importantly, keeping the reward model frozen is essential for stable reinforcement learning. The EGNN acts as a static surrogate for the ground-truth physical simulator. If we were to fine-tune or update the EGNN during the generation process, it would create a non-stationary objective (a "moving target"). This typically leads to reward drift or reward hacking, where the policy $\pi_\theta$ learns to exploit transient fluctuations or loopholes in the updating reward model rather than genuinely optimizing the molecular properties. Therefore, using the frozen, optimally-selected predictor ensures that the optimization process is driven by a consistent and robust standard.

> Q5: Can the model be scaled to larger datasets to validate its scalability and generalization?

A5: Thank you for your question. We agree that verifying performance on larger and more complex datasets is essential to demonstrate the scalability and robustness of our framework. To address this, we extended our evaluation to the Alchemy dataset [1]. Alchemy features significantly higher structural complexity than QM9, containing molecules with up to 14 heavy atoms and covering a much broader and more diverse chemical space. This makes it an ideal benchmark for testing scalability. We maintained the experimental settings consistent with the main paper and compared POETIC directly against the strongest baseline, Geo2Seq with Mamba. As shown in the table below, POETIC maintains its superiority even on this more complex manifold, achieving substantially lower MAE across all six quantum properties compared to the baseline.

| Method | $\alpha$ | $\Delta\epsilon$ | $\epsilon_{\text{HOMO}}$ | $\epsilon_{\text{LUMO}}$ | $\mu$ | $C_v$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| data | 0.18 | 60 | 39 | 38 | 0.056 | 0.117 |
| Geo2Seq with Mamba | 1.85 | 445 | 603 | 296 | 0.603 | 0.110 |
| **POETIC (Ours)** | **1.32** | **162** | **155** | **66** | **0.167** | **0.077** |

These results confirm that POETIC effectively scales to larger datasets and handles increased structural complexity without compromising controllable generation performance.

[1] Alchemy: A Quantum Chemistry Dataset for Benchmarking AI Models. arXiv:1906.09427

> Q6: The structure and format of the generated molecule tokens.

A6: Thank you for your question. Following the Geo2Seq protocol [1], our model generates molecules as sequences of atoms and their corresponding 3D spherical coordinates. Specifically, the generated sequence follows a canonical traversal order. For each atom step, the model autoregressively generates four tokens:

1. Atom Type: The element symbol (e.g., C, N, O).
2. Distance ($d$): The Euclidean distance to the anchor atom.
3. Bond Angle ($\theta$): The angle relative to the preceding two atoms.
4. Torsion Angle ($\phi$): The dihedral angle relative to the preceding three atoms.

All continuous values are quantized into discrete tokens with two-decimal precision. Below is a representative example of a generated sequence from our model:
> H 0.000 0.000° 0.000° C 1.088 1.571° 0.000° N 2.058 2.195° 0.000° C 3.263 1.973° -0.000° C 4.506 2.195° -0.000° H 5.186 2.133° 0.201° H 5.183 2.133° -0.202° H 4.462 2.438° 0.001° C 3.766 1.599° 0.000° F 5.105 1.612° 0.000° C 3.371 1.226° 0.000° H 4.302 1.080° 0.000° C 2.151 0.972° 0.000° C 2.741 0.394° 0.000° H 3.509 0.468° 0.591° H 2.390 0.012° 3.082° H 3.508 0.467° -0.592°

[1] Geo2Seq: Geometry Informed Tokenization of Molecules for Language Model Generation. ICML 2025.

> Q7: How are bond (edge) connections predicted and reconstructed in the generation process?

A7: 




R3: Reviewer Gm3J

> Q1: 

A1:

> Q2: Concerns on Baseline Fairness and Data Splits，

A2: Thank you for your question. We respectfully clarify that our comparisons are strictly "apples-to-apples," and we adhered rigidly to the standard protocols established in prior work.
1. Strict Adherence to the Standard EDM Protocol. To ensure a strictly fair comparison, we adhered rigidly to the standard EDM protocol [1]. Specifically, the QM9 dataset was partitioned into 100K training, 18K validation, and 13K test samples. Crucially, the 100K training set was further divided into two disjoint halves of 50K samples each: one half was used exclusively to train the property predictor (EGNN), while the other was reserved for training the generative models. All baselines, including EDM and GeoLDM, were evaluated using this exact same split and the same pre-trained EGNN oracle to guarantee an apples-to-apples comparison.

2. Verification via Baseline Reproduction. To empirically verify this and address the concern about potential data mismatch, we re-trained and re-evaluated the primary baselines (EDM and GeoLDM) using our exact codebase and data partitions. The comparison between the numbers reported in the original papers and our reproduction is shown below:

| Method | $\alpha$ | $\Delta\epsilon$ | $\epsilon_{\text{HOMO}}$ | $\epsilon_{\text{LUMO}}$ | $\mu$ | $C_v$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| EDM | 2.76 | 655 | 356 | 584 | 1.11 | 1.10 |
| Our Reproduction | 2.79 | 648 | 352 | 591 | 1.13 | 1.09 |
| GeoLDM | 2.37 | 587 | 340 | 522 | 1.11 | 1.03 |
| Our Reproduction | 2.41 | 592 | 338 | 518 | 1.09 | 1.05 |
| POETIC | 0.21 | 62 | 39 | 27 | 0.08 | 0.08 |

As shown in the table, our reproduced results are highly consistent with the reported values. This confirms that the baselines' performance is stable under the standard EDM protocol.

[1] EDM: Equivariant diffusion for molecule generation in 3d. ICML 2022.


> Q3: Using the same EGNN model for both RL optimization may lead to overfitting.

A3: Thank you for your question. To prove that our improvements stem from genuine chemical alignment rather than exploiting specific biases of the EGNN, we conducted an additional ablation experiment.

**Experiment Setup**: We replaced the EGNN reward model in the RL training stage with SchNet [1], a completely different graph neural network architecture. Crucially, we kept the final evaluation metric (the standard pre-trained EGNN) unchanged to ensure a fair comparison with the baselines and the established benchmark protocol.

The results of POETIC w/ SchNet Reward compared to the baselines are presented below:

| Method | $\alpha$ | $\Delta\epsilon$ | $\epsilon_{\text{HOMO}}$ | $\epsilon_{\text{LUMO}}$ | $\mu$ | $C_v$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| EDM | 2.76 | 655 | 356 | 584 | 1.11 | 1.10 |
| GeoLDM | 2.37 | 587 | 340 | 522 | 1.11 | 1.03 |
| NExT-Mol | 1.16 | 297 | 205 | 235 | 0.507 | 0.512 | 
| Geo2Seq with Mamba | 0.46 | 98 | 57 | 71 | 0.164 | 0.275 |
| POETIC (Schnet Reward)| 0.28 | 89 | 54 | 44 | 0.121 | 0.110 |

As shown above, even when the policy is optimized using a reward signal (SchNet) that is completely independent of the evaluator (EGNN), POETIC still significantly outperforms all the baselines. This confirms that our framework effectively captures universal structure-property relationships and is robust to the choice of reward model.

[1] SchNet – A deep learning architecture for molecules and materials. The Journal of chemical physics, 2018.