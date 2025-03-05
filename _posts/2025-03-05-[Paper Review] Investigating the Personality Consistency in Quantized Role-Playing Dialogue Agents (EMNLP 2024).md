---
title: "[Paper Review] Investigating the Personality Consistency in Quantized Role-Playing Dialogue Agents (EMNLP 2024)"
date: 2025-03-05 14:00:00 +0900
categories:  
  - Paper Review
tags:  
  - NLP
  - EMNLP 2024
  - Persona-based Dialogue
---

**Abstract** This study explores the personality trait consistency of quantized large language models (LLMs) and evaluates the stability of assigned personalities across multiple interactions in role-playing scenarios. A non-parametric method called **Think2** is proposed to address personality inconsistencies, demonstrating its effectiveness in maintaining the consistency of **Quantized Role-Playing Dialogue Agents (QRPDA).**  

---

## **1. Introduction**  

- **Role-Playing Dialogue Agents (RPDA)** are large language models (LLMs) designed with predefined personas.  
- These personas can represent various groups, such as **teachers, famous characters, or historical figures.**  
- Analyzing RPDA behavior through a role-playing perspective helps avoid the **pitfalls of anthropomorphism** while providing a conceptual framework to study LLM behavior.  
- RPDA has gained attention in **both academia and industry**, being applied in areas such as **emotional companionship, interactive gaming, and personalized assistants.**  
- Understanding the **consistency of personality traits** in RPDA is crucial for ensuring predictable and reliable user interactions.  
- Due to **privacy concerns**, locally deployed RPDA models are gaining popularity, as they **minimize data transmission.**  
- **Resource constraints necessitate optimization techniques** (e.g., quantization).  
- While multiple studies have explored **LLM personality modeling**, no prior research has examined **the impact of quantization on RPDA behavior.**  
- This study investigates the **personality consistency of Quantized RPDA (QRPDA)**, addressing the following research questions (**RQs**):  

  - **RQ1:** How does quantization affect the personality consistency of QRPDAs?  
  - **RQ2:** What strategies can enhance personality consistency in QRPDAs?  
  - **RQ3:** What is the optimal model size, type, and quantization combination for locally deployed QRPDAs?  

- A series of experiments with different LLMs and quantization levels are designed to address these questions.  
- The findings reveal that **quantization reduces personality consistency**, posing challenges in maintaining assigned traits throughout a conversation.  
- To mitigate **personality drift**, we propose a **non-parametric approach called Think2**, which significantly improves **personality consistency** in quantized dialogue agents.  

---

## **2. Related Work**  

- **Personality Trait Measurement:**  
  - The **Big Five Personality Model (Fiske, 1949)** is a widely used framework for evaluating personality traits.  
  - The five components include **Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism (OCEAN).**  
  - Various assessment tools exist, such as the **Big Five Inventory (BFI)** (Fossati et al., 2011), which:  
    - Uses a **self-report questionnaire** with **44 items** on a **5-point Likert scale.**  

- **Psychological Evaluation of LLMs:**  
  - LLM personality is assessed using **self-report surveys** (Frisch & Giulianelli, 2024).  
  - Alternative approaches include **multiple-choice personality assessments** (Jiang et al., 2023b) and **interview-based evaluations** (Wang et al., 2023a).  
  - The **PsychoBench** framework provides a comprehensive evaluation of LLM psychological traits (Huang et al., 2024).  

- **Personality Classification Evaluation:**  
  - When transitioning from **trait allocation** to **character assignment**, more detailed assessments are needed, including **linguistic analysis, lexical consistency, and dialogue accuracy** (Wang et al., 2023b, 2024).  

- **Personality Evaluation of RPDA:**  
  - Previous studies have evaluated RPDA under **default settings** (Pellert et al., 2023; Huang et al., 2024) or in **RPDA-specific scenarios.**  
  - Persona assignments are primarily conducted via **prompting (Wang et al., 2023b; Jiang et al., 2024; Wang et al., 2023a)** or **in-context learning (Mao et al., 2024).**  
  - Some studies have explored **parametric approaches** for inducing specific personality types (Mao et al., 2024).  

- **Focus of LLM Personality Studies:**  
  - Research has primarily focused on **commercial LLMs and large open-source models** (Petrov et al., 2024; Jiang et al., 2024).  
  - Studies on **smaller open-source models** remain limited (La Cava et al., 2024).  

- **Studies on LLM Interaction Behaviors:**  
  - **Frisch et al.** explored LLM behavior through **collaborative storytelling**, but the study was limited to only **two personas** in a **single interaction** (Frisch & Giulianelli, 2024).  
  - **Noh et al.** examined interactions within **game agents** but did not focus on **general interaction behaviors** (Noh & Chang, 2024).  
  - **Ma et al.** emphasized inconsistencies in **assigned personalities** during interactions, highlighting the need for a **comprehensive study on personality consistency in QRPDA** (Ma et al., 2023).  

---

## **3. Methodology**  

<img width="813" alt="image" src="https://github.com/user-attachments/assets/93206209-d904-4c1d-bfdf-77ef888bda64" />

- Designed a series of experiments to examine the impact of **quantization** on QRPDA deployment.  
- Evaluated **personality consistency** between **quantized models** and **full-precision (FP16) models.**  
- Observed **personality retention and drift** in quantized models.  

### **3.1 Quantized On-Device LLMs**  

- Selected **four quantized on-device LLMs** for evaluation:  
  - **LLaMA3 8B Instruct**  
  - **Mistral 7B Instruct v0.3**  
  - **Gemma 7B Instruct v1.1**  
  - **Gemma2 9B Instruct**  
- Focused on **7B-scale models** due to **memory and computational constraints.**  
- Evaluated different quantization levels: **FP16, Q8_0, and Q4_0.**  
- Q8_0 and Q4_0 **reduced memory requirements by 50% and 75%**, respectively.  

### **3.2 RPDA Construction**  

<img width="322" alt="image" src="https://github.com/user-attachments/assets/a693b641-5c0f-4986-8afa-97e7aaebc29b" />

- Assigned personality traits to LLMs using **system prompts.**  
- Used the **Big Five Personality Model (OCEAN)** with **32 binary personality combinations.**  
- Represented **initial personalities** with five **binary indices.**  
- Tracked **personality changes** by comparing **personality pairs.**  

### **3.3 Multi-Turn Interactions**  

<img width="319" alt="image" src="https://github.com/user-attachments/assets/ac1fcacc-463a-47e6-a1c1-e3d0a5ad4391" />

- RPDA pairs simulated **natural multi-turn interactions.**  
- Exchanged **personal stories** at each turn for continuity.  
- Used **BFI self-assessments** to track **personality changes and consistency.**  

### **3.4 Narrative Linguistic Features**  

- Collected **OCEAN scores** and **narratives** post-interaction for **linguistic analysis.**  
- Used **LIWC and embedding (EMBD) methods** for implicit personality assessment.  
- EMBD was utilized to **overcome LIWC limitations.**  

### **3.5 Think2: Personality Reinforcement**  

- Standard RPDA models **only relied on initial personality traits.**  
- Personality traits **drifted** throughout interactions.  
- **Think2 Approach:** Encouraged RPDA to **reflect on personality before generating responses.**  
- Strengthened **personality consistency** across multi-turn interactions.  

---

# **4. Experimental Results**

- **Experiment Framework:** LLMs were deployed using the **Ollama framework**.  
- **Selected Models:**  
  - **LLaMA3 8B Instruct**  
  - **Mistral 7B Instruct v0.3**  
  - **Gemma 7B Instruct v1.1**  
  - **Gemma2 9B Instruct**  
- **Evaluation Method:**  
  - Three quantization levels: **FP16, Q8_0, Q4_0**  
  - **16 pairs of models** with opposing personality traits  
  - **20-turn interactions** per pair, with **15 repetitions** for each experiment  

## **4.1 OCEAN Score Visualization**  

<img width="653" alt="image" src="https://github.com/user-attachments/assets/c478ca76-bef3-4141-981b-77029814f801" />

- **Radar plots** were generated to analyze **OCEAN scores**.  
- Comparison of initial and post-**20-turn interaction** OCEAN scores:  
  - **Baseline models** exhibited **convergence** in personality traits over interactions.  
  - **Think2 demonstrated better personality consistency**, validating its effectiveness.  

## **4.2 Regression Analysis of Linguistic Features**  

<img width="653" alt="image" src="https://github.com/user-attachments/assets/22375109-8d1e-4a1e-aea7-ca7dbdfc1b61" />

- Comparative analysis was conducted using the **Gemma2 9B Instruct model**.  
- **LIWC and EMBD features** were utilized.  
- The **baseline approach** showed **decreased cross-validation accuracy**.  
- The **Think2 method maintained cross-validation accuracy**, ensuring personality consistency.  

## **4.3 Correlation Analysis**  

<img width="653" alt="image" src="https://github.com/user-attachments/assets/9f3c08c0-b8e6-4082-8624-0a84633b102d" />

- **Pearson correlation analysis results were presented**.  
- Correlation between **OCEAN scores and EMBD linguistic features** was computed.  
- **Baseline models** exhibited **a decrease in correlation** over interactions.  
- **Think2 effectively mitigated correlation loss**, reinforcing personality consistency.  

## **4.4 Discussion**  

- The findings indicate that **quantization reduces personality consistency in LLMs**.  
- Higher quantization levels **lead to decreased stability in personality traits**.  
- **Q8_0 is identified as a suitable option**, balancing **efficiency and personality consistency**.  
- **Quantization strategies should be adjusted based on specific models**.  

---

# **5. Conclusions**  

- This study experimentally demonstrated that **higher quantization levels reduce personality consistency** in RPDA built on quantized LLMs.  
- To effectively mitigate this issue, we proposed a **non-parametric method called Think2**.  
- **Think2 successfully maintains personality stability across interactions**.  
- **Gemma2(Q4_0) and LLaMA3(Q8_0) were identified as the optimal choices** for preserving personality traits.  
- The comprehensive analytical framework highlights **Think2’s potential to enhance QRPDA reliability in resource-constrained environments**.  

---

# **6. Limitations**  

- **Methodological Constraints:**  
  - Personality evaluation was limited to the **Big Five Inventory (BFI)**.  
  - The study focused on **specific LLMs and quantization levels**.  

- **Unexplored Areas:**  
  - Further research on **additional personality models** (e.g., HEXACO, Myers-Briggs).  
  - Investigation of **a broader range of LLMs**, including **smaller models and sub-billion-parameter models**.  
  - Exploration of **alternative quantization techniques** beyond those addressed in this study.  

- **Multilingual Research:**  
  - The study was limited to **English**; extending research to other languages is necessary.  

- **Need for Smaller LLMs:**  
  - The necessity of exploring **smaller models** such as **Phi-3, Qwen2, OpenELM**.  
  - Potential applications in **resource-constrained environments**.  

- **Investigation of Multimodal LLMs:**  
  - **Integration of text, image, and audio** input capabilities.  
  - Enhancements in **user interaction understanding and response generation**.  

- **Quantization Methods:**  
  - This study was **limited to GGUF quantization**; further research on **AWQ, GPTQ, and other methods** is needed.  

- **Diverse Interaction Scenarios:**  
  - Incorporation of **varied user demographics** and **different modes of interaction**.  
  - Reinforcement of **the robustness of the research findings**.  

- **Future Research Directions:**  
  - Expanding research **toward generalization and broader applicability**.  
  - Improving **user experience and ensuring responsible AI development**.  