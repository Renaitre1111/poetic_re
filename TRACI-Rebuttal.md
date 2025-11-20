---
title: TRACI-Rebuttal

---

Rebuttal: TRACI

R4: Reviewer ukrJ(2)

We sincerely appreciate the time and effort you dedicated to reviewing our paper. In the following response, we aim to address your main concern and provide further clarification.

> Q1: State the fundamental differences between TRACI and existing graph clustering methods.

A1: Thank you for pointing this out. The main differences between our work and existing graph clustering methods are summarized as follows:

- **Different Scenario**. We focus on the text-attributed graph clustering problem under class-imbalanced scenarios, which remains under-explored by existing methods.
- **Different Motivation**. Rather than relying solely on pre-extracted embeddings from textual information, our work comprehensively exploits textual features in TAGs. Specifically, we leverage the reasoning capabilities of LLMs to generate augmented views for each text, derive correlation scores reflecting semantic consistency within a group of texts, and infer cluster predictions for boundary nodes.
- **Different Methodology**. To address class imbalance in an unsupervised manner, we leverage LLMs to implement a minority-aware mixup with textual guidance, which assigns larger combining weights to minority nodes based on semantic consistency and further generates group-level embeddings.

> Q2: The semantic diversity and consistency of the LLM generated texts. What specific settings can alleviate the long-tail problem?

A2: Thank you for your valuable comments. We address your concerns point by point below:

- **Semantic Diversity**. We measure the similarity between the embeddings of the original texts and the generated texts as follows:
  
  | $\text{cosine}(\cdot,\cdot)$             | Cora   | CiteSeer | WikiCS | PubMed |
  | ---------------------------------------- | ------ | -------- | ------ | ------ |
  | $\text{cosine}(X,X^{aug})$               | 0.8757 | 0.8843   | 0.7848 | 0.9109 |
  | $\text{cosine}(X^{aug}_{1},X^{aug}_{2})$ | 0.8334 | 0.8482   | 0.9029 | 0.8609 |

- **Prediction Consistency**: We assess the consistency of predictions across two augmented views using NMI:
  
  | Metric | Cora   | CiteSeer | WikiCS | PubMed |
  | ------ | ------ | -------- | ------ | ------ |
  | NMI    | 0.6328 | 0.6650   | 0.8383 | 0.7945 |

- **Ablation Study**. We additionally conduct an ablation study to compare our method with and without fine-tuning:
  
  | Dataset  | Metric | Ours w/o fine-tuning | Ours           |
  | -------- | ------ | -------------------- | -------------- |
  | Cora     | ACC    | $72.95\pm2.27$       | $73.48\pm2.21$ |
  |          | NMI    | $55.08\pm0.16$       | $55.60\pm0.10$ |
  |          | F1     | $66.06\pm3.53$       | $67.03\pm3.79$ |
  | CiteSeer | ACC    | $64.77\pm0.04$       | $67.15\pm0.17$ |
  |          | NMI    | $40.69\pm0.02$       | $41.72\pm0.10$ |
  |          | F1     | $57.59\pm0.04$       | $60.44\pm0.14$ |

- **Text-guided Group Mixup**. We alleviate class imbalance by randomly partitioning the nodes into mixed groups and obtaining minority-aware representations by assigning higher weights to nodes from the minority class based on semantic consistency. Specifically, since the groups are dominated by nodes from the majority class, the contribution scores of minority nodes tend to be lower, resulting in higher correlation weights. Consequently, we re-balance the representation learning through group mixup, guided by textual semantics.

- **Correlation Scores**. The correlation scores are derived from the contribution scores $b$ and confidence scores $c$ produced by the LLM, as shown in Equation (2).

> Q3: What insights can be drawn solely from providing a generalization bound?

A3: Thanks for your comment. Several key insights can be concluded:

1. **Tighter Generalization Bound**. A tighter generalization bound suggests that TRACI is more effective at learning meaningful representations from imbalanced datasets. This means it can generalize well without overfitting to the majority classes.
2. **Re-balanced Group-wise Mixup**. Our work synthesizes group-wise representations guided by semantic coherence, which re-balances the contributions of majority and minority nodes within a mixed group. This re-balancing provably enhances generalization under imbalanced conditions by controlling representation divergence. 
3. **Improved Sample Efficiency**. The generalization error becomes significantly less sensitive to class imbalance, thereby improving sample efficiency for minority classes.

> Q4: How does the method scale to million-node graphs? What measures have been taken to address potential scalability issues?

A4: Thank you for your comment. To improve efficiency and scalability on large graphs, we perform subgraph sampling. Due to space constraints, please refer to **A2 in our response to Reviewer Q5Cw** for further details.

> Q5: Which of the three metrics reflect the effectiveness under class imbalance?

A5: Thank you for your comment. The macro F1 score treats all classes equally, regardless of their size. Therefore, poor performance on minority classes will directly reduce the overall F1 score, even if other metrics such as accuracy or NMI remain high.

> Q6: The temperature parameters in Equations 10 and 14 appear to be important hyperparameters. How do they affect model sensitivity?

A6: Thanks for your feedback. Our framework involves two temperature parameters, that is, $\tau_{1}$ for $\mathcal{L}_{mixup}$ and $\tau_{2}$ for $\mathcal{L}_{rank}$. To assess the model's sensitivity under class-imbalanced settings, we vary $\tau_{1}$ over $\{0.1, 0.3, 0.5, 0.7, 0.9\}$ and $\tau_{2}$ over $\{0.01, 0.03, 0.05, 0.07, 0.09\}$, respectively. The results of on the Cora dataset are presented below:

| $\tau_1$ | 0.1            | 0.3            | 0.5            | 0.7            | 0.9            |
|:--------:| -------------- | -------------- | -------------- | -------------- | -------------- |
| ACC      | $72.90\pm2.19$ | $73.30\pm2.42$ | $73.48\pm2.21$ | $74.61\pm1.05$ | $75.29\pm1.41$ |
| NMI      | $55.33\pm1.42$ | $55.70\pm1.55$ | $55.60\pm0.10$ | $58.69\pm0.37$ | $58.69\pm0.62$ |
| F1       | $66.00\pm3.80$ | $66.24\pm4.04$ | $67.03\pm3.79$ | $65.76\pm0.51$ | $66.00\pm0.28$ |

| $\tau_2$ | 0.01           | 0.03           | 0.05           | 0.07           | 0.09           |
|:--------:| -------------- | -------------- | -------------- | -------------- | -------------- |
| ACC      | $73.48\pm2.21$ | $73.28\pm1.92$ | $73.27\pm1.93$ | $73.22\pm1.88$ | $73.13\pm1.87$ |
| NMI      | $55.60\pm0.10$ | $55.42\pm0.09$ | $55.55\pm0.05$ | $55.49\pm0.05$ | $55.36\pm0.11$ |
| F1       | $67.03\pm3.79$ | $66.50\pm3.37$ | $66.35\pm3.29$ | $66.16\pm3.17$ | $66.07\pm3.16$ |

> Q7: Code organization.

A7: Thank you for your interest in our code. We detail the involvement of the LLM and the fine-tuning modules as follows:

- The function `efficient_gpt_text_ge` in `prompt.py` is responsible for generating augmented textual views using the LLM;
- The function `efficient_gpt_text_score` in `prompt.py` derives scores for each group of texts based on semantic similarity with the LLM;
- The functions `efficient_gpt_text_ind` and `efficient_gpt_text_cls` are used to generate concept representations for each cluster and to predict labels for boundary nodes, respectively;
- The training module is implemented in `lines 75-123` while the fine-tuning module is located in`lines 221-276`, and both in `main.py`.

> Q8: “Trace” in Eq.1.

A8: Thanks for you feedback. The term *"Trace"* in Equation 1 refers to the trace of a matrix, which is defined as the sum of the elements on its main diagonal.

> Q9: Constructed imbalanced datasets.

A9: Thank you for your comment. We construct class-imbalanced datasets with varying imbalance ratios $\rho$ following a long-tailed distribution. Specifically, we first sort the classes in descending order based on the number of nodes. Subsequently, the sample size for the $k$-th class is given by $n_k = \text{min}\{n_{\text{max}}\cdot \rho ^{-\frac{k−1}{K-1}}, n_k^0\}$, where $K$ denotes the total number of classes, and $n_k^0$ is the original number of nodes in the $k$-th class. 

> Q10: Extremely imbalanced scenarios.

A10: Thank you for pointing this out. We have evaluated our method under extremely imbalanced scenarios, with imbalance ratios between the majority and minority classes set to 50 and 100. The corresponding results are presented in **Table 7 in the appendix**. 

> Q11: Different LLMs and NMI missing in Table 2.

A11: Thanks for your comment. 

- **Different LLMs**. We have evaluated the model's performance using various LLMs, including the open-weight DeepSeek-V3 and three ChatGPT variants: GPT-3.5, GPT-4o-mini, and GPT-4.1-mini. The comparative results are presented in **Figure 4** of our manuscript.

- **NMI Comparison**. We apologize for the oversight regarding the missing NMI scores in **Table 2**. The complementary version of **Table 2** is provided below:
  
  | Method    | Cora           | CiteSeer       | WikiCS         | PubMed         |
  | --------- | -------------- | -------------- | -------------- | -------------- |
  | variant 1 | $53.41\pm1.39$ | $41.09\pm2.25$ | $44.38\pm0.73$ | $22.99\pm0.02$ |
  | variant 2 | $54.21\pm2.16$ | $41.54\pm2.89$ | $48.64\pm0.19$ | $19.53\pm2.16$ |
  | variant 3 | $52.07\pm0.63$ | $37.57\pm2.74$ | $48.33\pm1.26$ | $19.09\pm2.12$ |
  | Ours      | $55.60\pm0.10$ | $41.72\pm0.10$ | $49.23\pm1.25$ | $20.90\pm1.61$ |

> Q12: Optimization of losses.
> A12: Thanks for your comment. We incorporate three loss functions in our framework: $\mathcal{L}_{\text{corr}}$, $\mathcal{L}_{\text{mixup}}$, and $\mathcal{L}_{\text{rank}}$. In the training stage, we optimize $\mathcal{L}_{\text{corr}}$ and $\mathcal{L}_{\text{mixup}}$ together. In the fine-tuning stage, all three losses are jointly optimized. This fine-tuning stage introduces limited complexity, as it mainly adapts the GNN with LLM-guided feedback.

R1: Reviewer Q5Cw(3)

Thank you for your thoughtful review and constructive feedback. We sincerely appreciate the opportunity to clarify the raised points and further improve our manuscript. Since the disadvantages and questions overlap, we address them together in the following response.

> Q1: The technical novelty of TRACI lies primarily in integrating and adapting existing techniques (augmentation, mixup, canonical correlation) with LLMs, rather than introducing fundamentally new algorithms or theoretical advancements.

R1: Thanks for your feedback. the novelty of our work lies in the following four key aspects:

- **Under-explored Settings**. Our work focuses on the imbalanced text-attributed graph clustering problem, which has received limited attention, whereas existing studies primarily address standard graph clustering problems.
- **Textual Feature Mining**. Rather than merely extracting textual features, our work deeply explores them from various perspectives to better capture semantic richness.
- **Novel Framework**. To address class imbalance in unsupervised scenarios, we introduce: (1) *canonical correlation alignment* between various augmented views of nodes to ensure semantic alignment; (2) *minority-aware mixup* among samples with respect to the synthesized groups, reinforcing minority-aware group-level representations; (3) *fine-tuning with ranking guidance*, which refines the GNN encoder through ranking-based supervision from pseudo-labels predicted by the LLM.
- **Performance Improvement**. Compared with existing baselines, our work achieves substantial performance gains under various imbalance ratios.

> Q2: Could the authors provide more details on the computational cost and scalability of TRACI, especially regarding the use of large language models (LLMs) for generating augmented views and computing correlation scores? How does the framework perform on large-scale graphs in terms of runtime and resource consumption?

A2: Thank you for your comment. We address your concerns point by point as follows:

- **Computational Cost**. Similar to TAPE [1], the cost is estimated based on the token-level pricing of GPT-4o-mini for both input and output tokens, while the runtime is estimated using its per-minute rate limit. Here, we take the Cora dataset as an example. Specifically, each node involves approximately 265 input tokens and 260 output tokens for generating augmented views, and about 265 input tokens and 65 output tokens for computing correlation scores. Detailed cost estimations for each dataset are provided in **Table 8 in the appendix**.

- **Large-Scale Graphs**. To ensure scalability on large-scale datasets, we perform subgraph sampling on the whole graph to improve efficiency with minimal performance loss. Specifically, we randomly assign $n=126$ nodes as root nodes and then sample their one-hop and two-hop neighbors to construct a subgraph. Specifically, we take Cora and CiteSeer datasets as examples: the number nodes is reduced from 2258 to 947 for Cora, and from 1737 to 716 for CiteSeer. As a result, both runtime and cost are reduced by approximately 60%, while the model's performance remains competitive. The detailed results are shown below:
  
  | Dataset  | Metric | subgraph       | Ours           |
  | -------- | ------ | -------------- | -------------- |
  | Cora     | ACC    | $71.08\pm1.14$ | $73.48\pm2.21$ |
  |          | NMI    | $54.29\pm1.46$ | $55.60\pm0.10$ |
  |          | F1     | $63.15\pm1.54$ | $67.03\pm3.79$ |
  | CiteSeer | ACC    | $65.27\pm0.65$ | $67.15\pm0.17$ |
  |          | NMI    | $39.32\pm0.65$ | $41.72\pm0.10$ |
  |          | F1     | $56.47\pm1.06$ | $60.44\pm0.14$ |

These results demonstrate that subgraph sampling is a promising approach for scaling to large datasets.

[1] Harnessing Explanations: LLM-to-LM Interpreter for Enhanced Text-Attributed Graph Representation Learning. ICLR 2024.

> Q3: Since TRACI relies heavily on LLM-generated augmentations and correlation/confidence scores, how sensitive is the framework to the choice of LLM or prompt design? Have the authors evaluated the impact of different LLMs or prompt variations on clustering performance and semantic consistency?

Q3: Thank you for your valuable feedback. We have evaluated the model's performance using various LLMs, including the open-weight DeepSeek-V3 and three ChatGPT variants: GPT-3.5, GPT-4o-mini, and GPT-4.1-mini. The comparative results are presented as follows:

| Dataset  | Metric | DeepSeek-V3    | GPT-3.5        | GPT-4o-mini    | GPT-4.1-mini   |
| -------- | ------ | -------------- | -------------- | -------------- | -------------- |
| Cora     | ACC    | $69.27\pm0.39$ | $73.16\pm1.05$ | $73.48\pm2.21$ | $70.07\pm4.77$ |
|          | NMI    | $53.81\pm0.39$ | $56.52\pm1.64$ | $55.60\pm0.10$ | $55.22\pm1.83$ |
|          | F1     | $61.41\pm0.43$ | $63.79\pm0.77$ | $67.03\pm3.79$ | $61.11\pm6.13$ |
| CiteSeer | ACC    | $66.74\pm7.13$ | $68.92\pm0.83$ | $67.15\pm0.17$ | $68.14\pm0.03$ |
|          | NMI    | $42.95\pm2.37$ | $43.68\pm0.20$ | $41.72\pm0.10$ | $43.40\pm0.00$ |
|          | F1     | $58.75\pm7.26$ | $61.34\pm0.99$ | $60.44\pm0.14$ | $61.77\pm0.01$ |

> Q4: The experiments focus on benchmark datasets with imbalanced text-attributed graphs. Can the authors clarify how well TRACI generalizes to real-world or industrial datasets that may have different imbalance patterns, noise levels, or textual characteristics? Are there known failure modes or limitations in these scenarios?

A4: Thank you for your comment.

- **E-commerce Dataset**. To demonstrate the generalizability of our method on real-world datasets, we further evaluate it on the e-commerce network, Books-History [1], from Amazon. Particularly, node attributes correspond to book descriptions, and edges indicate that two books were co-purchased or co-viewed. The performance of our method and the baselines is presented below, where our method achieves consistently strong results on this real-world dataset.
  
  | Method    | ACC            | NMI            | F1             |
  | --------- | -------------- | -------------- | -------------- |
  | DMoN      | $37.17\pm0.22$ | $34.49\pm1.17$ | $30.22\pm1.56$ |
  | Dink-Net  | $41.26\pm3.31$ | $35.35\pm1.44$ | $33.28\pm1.44$ |
  | HSAN      | $42.27\pm1.78$ | $36.90\pm0.83$ | $31.81\pm0.99$ |
  | S3GC      | $31.67\pm1.46$ | $24.29\pm0.78$ | $24.61\pm0.50$ |
  | DGCluster | $37.47\pm2.74$ | $38.47\pm0.57$ | $32.47\pm2.57$ |
  | MAGI      | $42.43\pm1.06$ | $40.34\pm0.69$ | $33.66\pm1.42$ |
  | IsoSEL    | $39.57\pm0.55$ | $30.51\pm0.96$ | $22.89\pm1.18$ |
  | Ours      | $44.46\pm5.15$ | $39.56\pm1.77$ | $36.48\pm4.08$ |

[1] A comprehensive study on text-attributed graphs: Benchmarking and rethinking. NIPS 2023.

- **Noise levels**. We evaluate our method under varying noise levels $p$ where we randomly drop a percentage $p$ of edges and add virtual edges between nodes from distinct classes. The results are illustrated as follows:
  
  | Metric | p=0            | p=0.1          | p=0.2          |
  | ------ | -------------- | -------------- | -------------- |
  | ACC    | $73.48\pm2.21$ | $74.41\pm2.32$ | $71.78\pm0.68$ |
  | NMI    | $55.60\pm0.10$ | $56.60\pm0.23$ | $54.94\pm0.59$ |
  | F1     | $67.03\pm3.79$ | $68.19\pm4.04$ | $64.57\pm1.63$ |

R2: Reviewer KWPz(4)

Thank you for your thoughtful review and constructive feedback. We appreciate the opportunity to clarify these points and improve our paper.

> Q1: Comparison with baselines specifically designed for imbalance graph learning methods.

A1: Thank you for your feedback. We clarify the differences between our method and existing imbalanced graph clustering approaches as follows:

- **Different Settings**. Most existing imbalanced graph learning methods are designed for supervised settings, where label information is available during training. In contrast, our work focuses on the unsupervised scenario, where no labels are used, making the task inherently more challenging and generalizable.  

- **Compare with Imbalance Graph Learning Methods**. We compare the representations learned by our method with those of other imbalanced graph learning methods, including GraphSMOTE [1], GraphENS [2], and BAT [3], using our established imbalanced datasets. The comparison results are presented below:
  
  | Dataset  | Metric | GraphSMOTE     | GraphENS       | BAT            | Ours           |
  | -------- | ------ | -------------- | -------------- | -------------- | -------------- |
  | Cora     | ACC    | $83.47\pm1.28$ | $84.00\pm1.30$ | $84.84\pm0.95$ | $84.92\pm0.97$ |
  |          | NMI    | $65.31\pm2.15$ | $65.71\pm3.30$ | $67.31\pm1.39$ | $65.90\pm1.96$ |
  |          | F1     | $77.75\pm1.89$ | $79.56\pm2.50$ | $80.76\pm1.30$ | $80.47\pm1.53$ |
  | CiteSeer | ACC    | $73.31\pm1.03$ | $73.14\pm2.56$ | $75.54\pm0.56$ | $78.91\pm2.44$ |
  |          | NMI    | $46.07\pm1.70$ | $46.11\pm2.85$ | $48.79\pm1.17$ | $52.41\pm5.12$ |
  |          | F1     | $64.86\pm1.40$ | $64.25\pm2.32$ | $66.35\pm1.23$ | $66.88\pm4.37$ |
  | WikiCS   | ACC    | $81.32\pm1.54$ | $80.28\pm0.97$ | $81.56\pm0.54$ | $82.01\pm0.93$ |
  |          | NMI    | $62.83\pm2.21$ | $61.52\pm1.49$ | $63.12\pm0.76$ | $63.35\pm1.36$ |
  |          | F1     | $78.62\pm1.76$ | $77.52\pm0.56$ | $79.07\pm0.53$ | $79.29\pm0.70$ |
  | PubMed   | ACC    | $86.40\pm1.10$ | $84.72\pm0.39$ | $87.14\pm0.37$ | $89.23\pm0.91$ |
  |          | NMI    | $45.02\pm1.92$ | $42.47\pm0.96$ | $45.64\pm0.84$ | $50.23\pm3.00$ |
  |          | F1     | $76.82\pm1.38$ | $75.06\pm0.45$ | $77.14\pm0.57$ | $79.23\pm0.96$ |

[1] Graphsmote: Imbalanced node classification on graphs with graph neural networks. WSDM 2021.

[2] Graphens: Neighbor-aware ego network synthesis for class-imbalanced node classification. ICLR 2021.

[3] Class-Imbalanced Graph Learning without Class Rebalancing. ICML 2024.

> Q2: Comparison with LLM-based methods.

A2: Thank you for your comment. We compare our method with MARK [1], which utilizes LLMs to perform unsupervised clustering on text-attributed graphs (TAGs).

| Dataset  | Metric | MARK           | Ours           |
| -------- | ------ | -------------- | -------------- |
| Cora     | ACC    | $69.60\pm3.58$ | $73.48\pm2.21$ |
|          | NMI    | $54.62\pm1.27$ | $55.60\pm0.10$ |
|          | F1     | $61.80\pm4.60$ | $67.03\pm3.79$ |
| CiteSeer | ACC    | $65.76\pm3.32$ | $67.15\pm0.17$ |
|          | NMI    | $42.75\pm0.56$ | $41.72\pm0.10$ |
|          | F1     | $56.55\pm5.36$ | $60.44\pm0.14$ |
| WikiCS   | ACC    | $59.17\pm2.49$ | $63.33\pm1.55$ |
|          | NMI    | $48.92\pm1.03$ | $49.23\pm1.25$ |
|          | F1     | $51.13\pm3.15$ | $53.93\pm1.60$ |
| PubMed   | ACC    | $52.09\pm0.72$ | $61.42\pm5.30$ |
|          | NMI    | $16.07\pm1.33$ | $20.90\pm1.61$ |
|          | F1     | $44.95\pm1.12$ | $52.45\pm4.98$ |

[1] MARK: Multi-agent Collaboration with Ranking Guidance for Text-attributed Graph Clustering. ACL Findings 2025.

> Q3: State the specific challenges of adapting supervised LLM-based approaches (LA-TAG[55]) to the unsupervised clustering domain.

A3: Thank you for pointing this out. The supervised LLM-based method LA-TAG addresses imbalanced node classification by using LLMs for data augmentation, leveraging label information to augment minority class nodes. However, since labels are not available in our unsupervised setting, this approach cannot be directly applied.

R3: Reviewer Z2b8(4)

We sincerely appreciate the time and effort you dedicated to reviewing our paper, as well as your invaluable feedback. In the following response, We address each point below:

> Q1: The Relevant Work section can refer to [1] for a more comprehensive review, such as LLM-as-aligner: Grenade[2], G2P2[3].

A1: Thank you for pointing this out. We will revise our related work section by including the "LLM-as-Aligner" paradigm for a comprehensive review. Specifically, the revised paragraph is as follows:

```
More recently, LLM-based approaches for text-attributed graphs have emerged and can be broadly categorized into three paradigms: LLM-as-Predictor, LLM-as-Enhancer and LLM-as-Aligner. More importantly, LLM-as-Aligner aims to align the outputs from GNNs and LLMs iteratively or in parallel [1]. This paradigm simultaneously leverage the structural aggregation capabilities of GNNs and the semantic extraction abilities of LLMs. These methods are typically implemented through prediction alignment [2] or embedding alignment [3-6].
```

[1] Large Language Models on Graphs: A Comprehensive Survey. TKDE 2024.

[2] Learning on large-scale text-attributed graphs via variational inference. ICLR 2023.

[3] LLMs as Zero-shot Graph Learners: Alignment of GNN Representations with LLM Token Embeddings. NIPS 2024.

[4] Prompt Tuning on Graph-augmented Low-resource Text Classification. TKDE 2024.

[5] GRENADE: Graph-Centric Language Model for Self-Supervised Representation Learning on Text-Attributed Graphs. EMNLP Findings 2023.

[6] Large Language Model Meets Graph Neural Network in Knowledge Distillation. AAAI 2025.

> Q2: The problem of existing “Long-tailed Graph Learning” is not only ignoring semantic information, but also focusing on supervised scenarios. The authors should emphasize their contributions in unsupervised learning communities.

A2: Thanks a lot for your valuable suggestion. We will emphasize this distinction in the related work section. The revised paragraph is provided below:

```
Furthermore, it's significant to note that most previous studies rely on supervised signals to address class imbalance, whereas our method focuses on the under-explored unsupervised scenario.
```

> Q3: The authors should discuss in detail how the proposed approach alleviates the class imbalance. I seem to see an explanation of this in the design of the prompt in the appendix, which the author should explain directly in the main text section, especially in Section “Long-tailed Group Mixup with Textual Guidance”.

A3: Thank you for your comment. We will provide a more detailed explanation of how the proposed approach alleviates class imbalance in the main text. The revised paragraph is as follows:

```
The contribution scores estimate the semantic coherence of texts within a randomly formed group, where nodes from the majority class dominatate. As a result, nodes from the majority class tend to receive higher contribution scores while those from the minority class receive lower scores. Meanwhile, the certainty of these scores is captured by the confidence scores. By incorporating the score outputs from the LLM, we construct a correlation-based matrix in which minority nodes exhibit higher values. As a result, the weighted mean of group-level representations becomes minority-aware, thereby effectively re-balancing the influence between majority and minority nodes.
```

> Q4: The ablation experiment lacks validation of “Canonical Correlation”. The author should explain clearly why both “Canonical Correlation” and “Group Mixup” should be used simultaneously?

A4: Thank you for your comment. "Canonical Correlation" is employed to align augmented node-level embeddings for enhancing representation learning, while "Group Mixup" is  designed to re-balance the majority and minority nodes to mitigate class imbalance. To better illustrate the contribution of "canonical correlation", we provides an additional model variant *w/o $\mathcal{L}_{corr}$*, in which the canonical correlation loss is removed. The results are presented below:

| $\rho=10$                | Cora           | CiteSeer       | WikiCS         | PubMed         |
| ------------------------ | -------------- | -------------- | -------------- | -------------- |
| w/o $\mathcal{L}_{corr}$ | $72.60\pm2.32$ | $64.04\pm2.31$ | $55.41\pm2.80$ | $50.96\pm2.99$ |
| Ours                     | $73.48\pm2.21$ | $67.15\pm0.17$ | $63.33\pm1.55$ | $61.42\pm5.30$ |

| $\rho=20$                | Cora           | CiteSeer       | WikiCS         | PubMed         |
| ------------------------ | -------------- | -------------- | -------------- | -------------- |
| w/o $\mathcal{L}_{corr}$ | $64.82\pm0.40$ | $60.05\pm7.06$ | $58.56\pm5.18$ | $50.10\pm1.26$ |
| Ours                     | $68.89\pm6.08$ | $67.15\pm6.50$ | $60.57\pm3.37$ | $64.72\pm5.28$ |

> Q5: Could you be more specific about how unbalanced datasets are generated and provide the number of nodes per category at different balance rates?

A5: Thank you for your feedback. We construct class-imbalanced datasets with varying imbalance ratios $\rho$ following a long-tailed distribution [1]. Specifically, we first sort the classes in descending order based on the number of nodes. Subsequently, the sample size for the $k$-th class is given by $n_k = n_{\text{max}}\cdot \rho ^{-\frac{k−1}{K-1}}$, where $K$ denotes the total number of classes. To preserve the topological structure of the original graph, nodes with higher connectivity are preferentially retained during the sampling process. The number of nodes under varying $\rho$ (10, 20, 50, 100) are shown in **Figure 2 in the manuscript**.

[1] Class-Imbalanced Learning on Graphs: A Survey. ACM Computing Surveys 2025.

> Q6: What is the essential difference between the traditional supervised imbalanced classification method and the proposed method? 

A6: Thank you for your comment. The main differences between our work and existing graph clustering methods are summarized as follows:

- **Different Scenario**. We focus on the text-attributed graph clustering problem under class-imbalanced scenarios, which remains under-explored by existing methods.
- **Different Motivation**. Rather than relying solely on pre-extracted embeddings from textual information, our work comprehensively exploits textual features in TAGs. Specifically, we leverage the reasoning capabilities of LLMs to generate augmented views for each text, derive correlation scores reflecting semantic consistency within a group of texts, and infer cluster predictions for boundary nodes.
- **Different Methodology**. To address class imbalance in an unsupervised manner, we leverage LLMs to implement a minority-aware mixup with textual guidance, which generates group-level representations and assigns larger combining weights to minority nodes based on semantic consistency.

> Q7: The warm loss in Eq. 5 consists of two components. What is the motivation for this combination? A clear explanation might be more helpful for readers to understand. 

A7: Thanks for your comment. The warm loss consists of two components, $\mathcal{L}_{corr}$ and $\mathcal{L}_{mixup}$. Specifically, $\mathcal{L}_{corr}$ encourages embedding alignment between two augmented views at the node level, while $\mathcal{L}_{corr}$ mixes nodes within a group to re-balance the influence of majority and minority classes. An additional ablation experiment to illustrate the contribution of $\mathcal{L}_{corr}$ has shown in **A4**. 

> Q8: I don't have a thorough understanding of how contribution scores and confidence scores mitigate imbalance problem. Can the authors provide a clearer explanation?

A8: Thank you for your feedback. We randomly partition the nodes into distinct groups, in which nodes from the majority class dominate. Since nodes within the same class tend to exhibit semantic consistency, nodes from the minority class typically receive lower scores due to being less representative, while those from the majority class receive higher contribution scores. The confidence scores reflect the reliability of the LLM's outputs. Based on these two scores, we derive a weight matrix $S$ (Equation 2), where higher weights are assigned to minority-class nodes and lower weights to majority-class nodes. This ensures that the weighted mean of group-level representations becomes minority-aware, effectively re-balancing the influence of majority and minority nodes.

What specific objective is the contribution score intended to support? Could you provide empirical statistical results of group scoring to clarify the impact of LLM-generated scores? Please further clarify this point.

Thank you for your comment. The contribution scores are designed to identify samples from the minority class from a semantic perspective. In general, contribution scores for minority samples are significantly lower than those for majority samples. We demonstrate the significance of the contribution scores from two perspectives:

-**Mann-Whitney U test**. For the PubMed dataset with $\rho=10$, the mean ratio between majority and minority in synthesized groups is 10.17, aligning with the original class distribution. We further conduct a Mann-Whitney U test to examine that whether the contribution scores of the minority samples are significantly lower than those of the majority. Specifically, the p-value of the Mann-Whitney U test is $1.90 \times 10^{-12}$, indicating a statistically significant difference. These results confirm that minority samples tend to receive lower contribution scores. Consequently, the correlation weights are set proportional to $(1 - \text{contribution score})$, meaning that minority samples statistically receive higher weights.

-**Ablation study** The significance of this module is further validated through an ablation study. We compare our method, which leverages LLM-generated contribution scores, with a variant that uses uniform correlation weights. The results demonstrate that incorporating the LLM-generated scores leads to a significant performance improvement.
    | $\rho=10$         | Cora           | CiteSeer       | WikiCS         | PubMed         |
    | ----------------- | -------------- | -------------- | -------------- | -------------- |
    | w/o LLM-generated | $70.19\pm2.60$ | $64.31\pm6.59$ | $61.33\pm0.34$ | $59.12\pm6.12$ |
    | Ours              | $73.48\pm2.21$ | $67.15\pm0.17$ | $63.33\pm1.55$ | $61.42\pm5.30$ |

Many thanks for your valuable efforts in helping improve our manuscript.

Thank you for your efforts in improving our manuscript. If there are any additional concerns, please do not hesitate to let us know. We look forward to further discussion with you.

Thank you for your thoughtful review and constructive feedback. We appreciate the opportunity to clarify these points and improve our paper.

A1: Thanks for your comment. The primary methodological distinction lies in the incorporation of semantic guidance of TAGs, while other differences have been previously addressed. The methodological differences are detailed as follows:

- **Semantic Augmentation**. We employ an augmentation strategy that enhances text representations using LLMs, without perturbing the original semantic content. 
- **Text-guided Mixup**. Under unsupervised and imbalanced scenarios, we propose a mixup strategy that identifies minority nodes from other samples based on semantic information. Specifically, minority nodes tend to have statistically lower contribution scores compared to majority nodes. This phenomenon has been validated both statistically and experimentally, as detailed in our additional rebuttal to Reviewer Z2b8.
- **LLM-generated feedback**. For nodes that are difficult for the GNN to classify, we leverage the reasoning capabilities of LLMs to generate reliable feedback. This feedback is then used as a supervision signal to fine-tune the GNN. 

Through these modules, our method enhances text-attributed graph clustering from a semantic perspective, particularly under class-imbalanced scenarios.

A2: Thank you for your comment. We address your concerns through the following two points:

-** Compare with LLM-based methods**. We have compared with a LLM-based method MARK [1],  which utilizes LLMs to perform unsupervised clustering on text-attributed graphs (TAGs). And the results are shown in the following:

| Dataset  | Metric | MARK           | Ours           |
| -------- | ------ | -------------- | -------------- |
| Cora     | ACC    | $69.60\pm3.58$ | $73.48\pm2.21$ |
|          | NMI    | $54.62\pm1.27$ | $55.60\pm0.10$ |
|          | F1     | $61.80\pm4.60$ | $67.03\pm3.79$ |
| CiteSeer | ACC    | $65.76\pm3.32$ | $67.15\pm0.17$ |
|          | NMI    | $42.75\pm0.56$ | $41.72\pm0.10$ |
|          | F1     | $56.55\pm5.36$ | $60.44\pm0.14$ |
| WikiCS   | ACC    | $59.17\pm2.49$ | $63.33\pm1.55$ |
|          | NMI    | $48.92\pm1.03$ | $49.23\pm1.25$ |
|          | F1     | $51.13\pm3.15$ | $53.93\pm1.60$ |
| PubMed   | ACC    | $52.09\pm0.72$ | $61.42\pm5.30$ |
|          | NMI    | $16.07\pm1.33$ | $20.90\pm1.61$ |
|          | F1     | $44.95\pm1.12$ | $52.45\pm4.98$ |

It's obvious that our method shows competitive performance compared to the LLM-based graph clustering method MARK.

-**Variable Explanation**.Apologies for the oversight. $X^{aug}_1$ and $X^{aug}_2$ denote the node embeddings extracted from two augmented views. and $cosine(X^{aug}_1, X^{aug}_2)$ measures the semantic similarity between the two augmented views, while $cosine(X, X^{aug})$ represents the mean cosine similarity between the original text embedding and the two augmented views.

[1] MARK: Multi-agent Collaboration with Ranking Guidance for Text-attributed Graph Clustering. ACL Findings 2025.

A3: Thank you for your insightful comment. In supervised settings, instance-level contrastive learning has indeed proven to be a powerful approach for representation learning. Our work extends beyond the classic contrastive loss by introducing a mixup-inspired strategy that does not require additional label information, thus preserving the fully unsupervised nature of the task. From the theoretical perspective, we establish that our proposed mixup loss, (\mathcal{L}{\text{mixup}}), which exhibits a provably tighter generalization bound compared to the standard contrastive loss). This guarantee suggests that (\mathcal{L}{\text{mixup}}) can lead to improved out-of-sample performance in theory. By operating at the group level, our mixup contrastive objective effectively mitigates the challenges posed by class imbalance in the unsupervised setting. Empirically, we further validate the improvement brought by (\mathcal{L}{\text{mixup}}) over the traditional instance-level contrastive loss via ablation studies. As summarized in the following table, directly replacing (\mathcal{L}{\text{mixup}}) with a standard contrastive objective leads to a consistent decrease in performance across various datasets:

| $\rho=10$                 | Cora           | CiteSeer       | WikiCS         | PubMed         |
| ------------------------- | -------------- | -------------- | -------------- | -------------- |
| w/o $\mathcal{L}_{mixup}$ | $72.96\pm2.15$ | $63.90\pm7.25$ | $50.40\pm3.23$ | $60.76\pm4.82$ |
| Ours                      | $73.48\pm2.21$ | $67.15\pm0.17$ | $63.33\pm1.55$ | $61.42\pm5.30$ |

These results confirm the effectiveness and the practical utility of our proposed mixup-based approach in addressing class imbalance under unsupervised learning scenarios.

A4: Thank you for your comment. We have validated the effectiveness of our method on a large-scale dataset, Reddit [1]. The imbalanced Reddit dataset consists of 18,388 nodes, where each node represents a user, the associated text corresponds to the user's published posts, and an edge indicates that two users have replied to each other. We evaluated our method on this social network dataset, and the results are presented below:

| Method | ACC            | F1             |
| ------ | -------------- | -------------- |
| DMoN   | $83.58\pm3.29$ | $49.36\pm1.85$ |
| Ours   | $85.44\pm0.01$ | $53.43\pm0.01$ |

More importantly, the computational cost and running time have been estimated on the Reddit dataset as well as the four datasets used in our study. The results are presented as follows:

| Dataset   | Cora  | CiteSeer | WikiCS | PubMed | Reddit |
| --------- | ----- | -------- | ------ | ------ | ------ |
| Cost(\$)  | 2.32  | 1.84     | 16.56  | 14.81  | 15.52  |
| Time(min) | 15.01 | 12.05    | 129.05 | 90.86  | 84.17  |

Although the computational cost and running time increase with graph size, they remain estimable and manageable even on relatively large-scale datasets.

[1] Can gnn be good adapter for llms? WWW 2024.

A5: We appreciate the reviewer’s comment and apologize for the misleading. Here, we define classes with top 50% samples as head classes and the bottom 50% samples as tail classes. And we compare performance on the Cora dataset with $\rho=10$, and class average accuracy (CAA) is employed for evaluation. The results are shown in the following table:

| CAA  | Head Classes    | Tail Classes    | All Classes    |
| ---- | --------------- | --------------- | -------------- |
| DMoN | $59.00\pm14.31$ | $56.86\pm15.45$ | $57.78\pm3.28$ |
| Ours | $71.58\pm6.33$  | $66.30\pm4.75$  | $68.56\pm0.01$ |

It's obvious that our method have significant improvement both on head classes and tail classes compared to the baseline method.

A6: Thank you for your comment.

A7: Thank you for you feedback.

A8: Thank you for your comment. We will provide detailed explanations of all abbreviations in the revised manuscript.

A9: I apologize for the oversight. Head classes are defined as those with the top 50% of samples, while tail classes refer to those with the bottom 50% of samples. Related performance improvements for both head and tail classes are reported in A5.

A10: Thank you for your positive feedback.

A11: Thank you for your comment. The performance of LLMs may vary across different datasets and evaluation metrics. Considering the trade-off between performance and computational cost, we select GPT-4o-mini as the baseline LLM and evaluate the impact of LLM selection on the overall performance.

A12: Thank you for your feedback. We incorporate three loss functions: $\mathcal{L}_{\text{corr}}$, $\mathcal{L}_{\text{mixup}}$, and $\mathcal{L}_{\text{rank}}$, into the training process. Specifically, $\mathcal{L}_{\text{rank}}$ is designed to enhance the model's ability to distinguish decision boundaries and hard-to-identify samples based on LLM-generated feedback. Therefore, it's natural to first combine the the initial two loss functions during the warm-up phase, which allows us to identify such challenging samples. Afterward, we query the LLM using the hard samples identified in the warm-up stage, and subsequently enhance the node embeddings with LLM guidance. The three loss functions involve only two temperature hyperparameters, $\tau_1$ and $\tau_2$, thereby minimizing the need for extensive hyperparameter tuning. Their sensitivity has already been evaluated in A6 before.

Many thanks for your efforts and constructive feedback. We have addressed your concerns point by point below:

A1: Thank you for your comment. We evaluate the learned representations of our method on node classification task to enable a direct comparison with the imbalance classification baselines GraphSmote, GraphENS, and BAT. Specifically, we directly apply an additional MLP classifier to the learned representations for this comparison. To ensure fairness, all methods are trained on the same dataset with a unified splitting ratio of 80/10/10 for training, validation, and testing, respectively.

A2: Thank you for your feedback. Since MARK is an LLM-based graph clustering method, we compare it with our approach on the established datasets specifically for the graph clustering task. The results of this comparison align with the findings reported in our paper. In contrast, comparisons with imbalanced graph learning methods are performed on node classification tasks, which explains the observed discrepancy.

A3: Thank you for your comment. We provide several reasons to explain the phenomenon you pointed out. 

- First, the high complexity of graph structures and the presence of potential connectivity noise often lead to fluctuations in performance when comparing results across different datasets. 
- Similar variations and challenges have been observed in prior studies [1], where performance on datasets like Cora and CiteSeer is not always optimal, while results on other datasets remain competitive. This underscores the inherent difficulty of consistently benchmarking graph-based methods. 
- Despite these challenges, our method consistently achieves the best performance on diverse datasets such as CiteSeer, WikiCS, and PubMed, which strongly validates the effectiveness and robustness of the proposed approach.

[1] S3GC: Scalable Self-Supervised Graph Clustering. NIPS 2022.

We hope our rebuttal has addressed your concerns. Thank you once again for your constructive feedback and your efforts in improving our manuscript!

Thank you for your response. All my concerns have been addressed and I will raise my score to 5.

I sincerely appreciate your detailed response. All of my previous concerns have been satisfactorily addressed and
the additional experiments have substantially improved the quality of the work. In light of these improvements, I am pleased to raise my evaluation score to 5.

Dear Area Chair,

Thank you for your time and effort in managing the review process. I am writing to provide a brief follow-up regarding our submission.

We are very grateful for the productive discussions thus far. In response to the reviewers' valuable feedback, we have provided further clarifications and conducted additional extensive experiments.

As the discussion period concludes today, we would greatly appreciate it if you could kindly remind the reviewers to complete the final steps of the review process. We noticed that **Reviewer** has not yet responded to our updated experiments, and **Reviewer** has not participated in the discussion. We wish to ensure that all concerns have been addressed and to clarify any remaining issues if necessary.

Thank you once again for your efforts in facilitating the review process.

Best regards,

The Authors

We sincerely thank the reviewer for raising the score. We greatly appreciate your recognition of our work and the constructive feedback that helped us further improve the quality of the paper. All additional experiments during discussion will be shown in the revised manuscript.