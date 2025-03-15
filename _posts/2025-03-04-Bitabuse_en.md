---
title: "[Paper Review] BitAbuse: A Dataset of Visually Perturbed Texts for Defending Phishing Attacks (NAACL 2025 Findings)"
date: 2025-03-04 00:00:00 +0900
categories:
  - Paper Review
tags:
  - NLP
  - Security
  - NAACL 2025 Findings
  - Dataset Construction
---

**Summary:** This study proposes the **BitAbuse** dataset, which comprises real phishing cases containing visually perturbed (VP) texts. The dataset aims to enhance the performance of language models and support adversarial attack defense research.

---

# 1 Introduction

<img alt="image" src="https://github.com/user-attachments/assets/34f7d1c2-6cce-488a-a8ca-1b7e19f25584" />

- **Social engineering attacks** exploit victims' psychological vulnerabilities to extract confidential information.  
  - Attack types: phishing, spam, pretexting, baiting, tailgating, etc.
- **Phishing attacks** target victims through text-based communication, such as emails and SMS.
  - **Visually perturbed (VP) texts** are used to bypass security systems.
  - Example of VP text: `"Bitcoin"` ➔ `"ßitcöın"`
- **VP text-based phishing attacks can be mitigated by restoring the original text.**
  - Existing research primarily focuses on restoration techniques.
  - However, **limitations exist** in models like **Viper**, which alter unrealistic fixed components.
- **LEGIT research** generated a VP text dataset while considering readability.
  - However, research on real-world VP texts remains insufficient.
- **Defense systems based on synthetic datasets may pose risks due to real-world discrepancies.**
  - **Solution:** Combining real VP texts with synthetic texts.
- **Proposed dataset: BitAbuse**
  - Constructed from **262,258 phishing-related emails**.
  - Contains **26,591 VP sentences** and **298,989 non-VP English sentences**.
  - Introduces three datasets: **BitCore, BitViper, and BitAbuse**.
  - A pilot study was conducted to analyze dataset characteristics.
- **BitAbuse phishing attack dataset is publicly available.**

---

# 2 Related Work

- **Challenges in VP text research:**
  - Difficulties in obtaining sufficient data due to its presence in spam emails.
  - Lack of datasets reflecting real-world phishing attack scenarios.
  - Existing datasets are only valid under specific conditions (e.g., internationalized domain names).
  - Traditional research integrates datasets for **VP text restoration techniques**.

- **Two notable VP text dataset integration studies:**
  - **TextBugger**:
    - Generates VP texts using predefined homograph pairs and transformation methods.
    - Selects characters in text and replaces them with VP characters to degrade LMs' performance.
    - Useful for exposing vulnerabilities in security-sensitive tasks like sentiment analysis and malicious content detection.
  
  - **Viper**:
    - Searches for homographs and generates VP texts based on embedding techniques.
    - Replaces text characters with VP characters and induces visual interference based on replacement probabilities.

- **Traditional VP text restoration methods:**
  - Utilize **SimChar DB, OCR, spell checker, and LMs**.
  - **SimChar DB** automatically collects homographs from Unicode character sets, detects VP characters, and restores text using predefined restoration tables.
  - **OCR-based methods** detect phishing attacks by inserting VP characters into IDNs.
  - **Spell checkers** restore malicious texts distributed across social networks.
  - **Combining two LMs (BERT & GPT)** for restoration strategies is also considered.

- **Common limitations in previous research:**
  - **Existing phishing attack restoration datasets lack real VP texts.**
  - This leads to **overestimation or underestimation** of restoration performance in real-world scenarios, potentially resulting in **unstable pre-trained LM models**.
  
- **This study collects VP texts used on bitcoinabuse[.]com to construct a new dataset for phishing attack research.**

---

# 3 BitAbuse

<img alt="image" src="https://github.com/user-attachments/assets/8d9949a4-74ef-4279-a94f-ab6b332a5f2d" />

- **Email data collection** from bitcoinabuse[.]com to obtain VP texts used in phishing attacks.
- **Bitcoin Abuse website** serves as a platform where users worldwide report Bitcoin-related scams.
- The platform allows users to upload anonymized phishing emails, enabling safe data collection.
- **A total of 262,258 phishing-related emails were collected** from May 16, 2017, to January 15, 2022.
- The goal was to construct an **English VP text dataset**, but **non-English texts** were also included, requiring **irrelevant email filtering**.
- **A BERT model was trained to classify English texts**, using **16,598 manually labeled samples**.
- **Final dataset**: **178,054 English emails** retained for further processing.
- **Sentences were split into a maximum length of 512 tokens**, resulting in **326,732 sentences**.
- **Regular expressions were used** to remove unnecessary character sequences.
- **Manual annotation of VP texts was conducted** on 326,732 sentences.
- **To reduce inefficiencies**, **VP words were converted to non-VP word labels** for automation.
- **1,152 irrelevant sentences were removed**.
- **Final datasets:** **BitCore, BitViper, and BitAbuse**.

---

# 4 Experimental Settings

- **Experimental databases & methodologies**  
  - Methods used: **SimChar DB (Suzuki et al., 2019), OCR (Sawabe et al., 2019), Spell Checker (Imam et al., 2022), Character BERT (El Boukkouri et al., 2020), GPT-4o mini (OpenAI, 2023)**
  - Evaluation metrics: **Word Level Accuracy, Word Level Jaccard, BLEU**

- **Restoration performance evaluation**  
  - Five different restoration methods were evaluated:
    1. **SimChar DB-based**: Identifies alphabet homographs and restores them.
    2. **OCR-based**: Applies OCR to each character and selects the most probable character.
    3. **Spell Checker-based**: Splits sentences into words and uses **Levenshtein Distance** for restoration.
    4. **Character BERT**: Processes token sequences at the character level for context-based restoration.
    5. **GPT-4o mini**: Evaluates performance using the latest **large language model (LLM)**.

- **Character BERT settings**
  - **Learning rate:** 5×10⁻⁵, **Batch size:** 32, **Epochs:** 10
  - **Optimizer:** AdamW (β₁ = 0.9, β₂ = 0.999, weight_decay = 0)
  - **Input & output:** Character-level token sequences.

- **GPT-4o mini model access**
  - Conducted via OpenAI’s inference API with a **carefully designed prompt**.

- **Evaluation metrics**
  - **Word Level Accuracy:** Measures if the restored word matches the target word at each position.
  - **Word Level Jaccard:** Computes the ratio of the intersection and union of word sets in predicted vs. labeled sentences.
  - **BLEU score:** Calculates **n-gram precision** between predicted and labeled sentences.

- **Data split**
  - **BitAbuse dataset split into 60% training, 20% validation, and 20% testing.**
  - **Each method's performance was evaluated on the test set, averaging results from 10 random train-test splits.**

---

# 5 Experimental Results

<img alt="image" src="https://github.com/user-attachments/assets/cb0b2514-d833-409c-b8c2-76e8a83c486e" />

- **Exploratory Data Analysis**: Analyzed VP words, VP characters, and their ratios to aid the development of phishing attack defense methodologies.

<img alt="image" src="https://github.com/user-attachments/assets/8e2d852e-0728-4298-94c1-8aa87823a4b2" />

- **VP Sentence Histogram**: Provided histograms illustrating the occurrence rates of VP characters based on sentence length across BitCore, BitViper, and BitAbuse datasets.

<img alt="image" src="https://github.com/user-attachments/assets/1199157e-05b0-4210-8cec-071e34a24ddb" />

- **VP Character-Word Association Graph**: Visualized the clustering of VP characters and words using the Yifan Hu algorithm, showing vowels as central elements in key clusters.

<img alt="image" src="https://github.com/user-attachments/assets/23027d72-a124-4acd-a040-deaae9ad2d4e" />

- **Restoration Performance Comparison**: Evaluated the restoration performance of SimChar DB, OCR, Spell Checker, Character BERT, and GPT-4o mini-based methods, with Character BERT outperforming the others.

<img alt="image" src="https://github.com/user-attachments/assets/1482461b-4a70-4c3b-8b22-17c1a7dfd505" />

- **VP Word Restoration Errors**: Character BERT-based methods exhibited increased failures when restoring continuous VP characters.

<img alt="image" src="https://github.com/user-attachments/assets/88878fbb-adc6-4711-8558-eabec717272b" />

- **Word-Level Accuracy Evaluation**: Performance of Character BERT-based methods was evaluated across three datasets based on VP character ratios, with BitCore achieving the strongest performance.

<img alt="image" src="https://github.com/user-attachments/assets/2f59752b-b8b9-4cac-88bb-6cccc9f8bc61" />

- **Performance Variations by Training Volume**: Observed performance degradation in BitViper and BitAbuse datasets when trained with only 1% or 5% of VP sentences.

- **Model Generalization Capability**: Despite a low training data ratio, the model successfully restored text, demonstrating practical advantages through rapid training.

- **Additional Experimental Results Summary**: Concluded that Character BERT-based methods demonstrated relatively superior performance, and a sufficient number of VP sentences is essential for building stable models.

---

# 6 Discussion

- **Comparison of restoration methods based on VP character ratio:**
  - Character BERT-based methods **effectively improved performance as VP character ratios increased**.
  - Spell Checker-based methods **suffered a steep decline in performance** due to their inability to utilize contextual information.
  - GPT-4o mini-based methods **exhibited degraded performance** as they failed to maintain input-output sequence order and indexing.

- **Jaccard and BLEU Performance Evaluations:**
  - **Jaccard scores** closely correlated with **Word Level Accuracy**.
  - **BLEU scores** were more sensitive to contextual accuracy, allowing simpler methods that preserve structure and meaning to achieve higher scores.

- **Comparison of five restoration methods across three datasets:**
  - **Character BERT-based methods** clearly outperformed all other approaches.
  - **SimChar DB-based methods** were limited to restoring a single VP character to only one non-VP character.

- **VP Character Restoration Capabilities of Each Method:**
  - Character BERT-based methods **learned context directly and performed fast restoration**.
  - GPT-4o mini exhibited **poor performance when multiple VP characters were present**.
  - Safety features caused **error responses when handling unethical content**.

- **Character BERT-based method achieving nearly 100% accuracy on BitCore dataset:**
  - Suggests potential applications in **digital forensics and secure messaging systems**.
  - The model could help address **critical challenges in secure communication**.

---

# 7 Conclusion

- **Developed three VP text datasets**: BitCore, BitViper, and BitAbuse.
- **BitCore and BitViper exhibit significantly different characteristics**, with LMs-based restoration methods demonstrating strong robustness and potential across all datasets.
- **BitAbuse is available for download**, pre-trained with **325,580 VP sentences**.
- **Future research should explore hybrid approaches** combining OCR and Character BERT.
- **Internalizing data within LMs** could mitigate **the excessive data consumption problem**.
- **Keyword-based tendencies** may facilitate the development of lightweight yet **accurate LMs for real-world phishing attacks**.
- **Zero-shot performance validation of the BitAbuse model** is also necessary.

---

# 8 Limitations

- **The VP text restoration experiments in this study did not include additional restoration methods** to avoid exceeding the research scope.
- **No direct comparison** was made between Character BERT-based methods and **other LM-based restoration techniques**, making it difficult to assess Character BERT's superiority.
- **BitAbuse dataset contains only Bitcoin scam data**, limiting its ability to reflect **diverse phishing attack scenarios**.
- **Phishing attacks are likely to evolve over time**, increasing in variety and complexity; failing to account for this diversity may reduce generalization capability.
- **There is a risk that non-experts could misuse the dataset to learn and execute phishing attacks.**
- **Misuse of the BitAbuse dataset may lead to more sophisticated phishing attacks, increasing victim exposure.**
- **Although the dataset and model are publicly available, there is no clear regulation against their technological misuse.**

---

# 9 Ethics Statement

- **The dataset created in this study is intended for phishing attack defense research.**
- However, **non-experts may use the dataset to learn and execute phishing attacks.**
- **It could be exploited for developing malicious tools** like **WormGPT** from the dark web or **PoisonGPT** released by Mithril Security.
- This may **lead to more sophisticated phishing attacks and increased victims**.
- **Legal responsibility cannot be imposed for damages caused by misuse of the dataset**.
- Many countries **lack clear regulations against dataset misuse**, requiring **careful consideration and monitoring**.
- **Although the dataset and model used in this study are publicly available, they should not be used for non-research purposes.**

---

# Reader Comments

- **This paper was written by the author.**
- **The primary focus is on introducing the dataset**, and methodologies for solving the task are limited to **basic applications of language models**.
- **Future research could incorporate multimodal approaches** by utilizing character shape information to enhance performance.
- However, **even without multimodal approaches, LLMs can achieve sufficiently high performance**, assuming **cost is not a constraint**.
