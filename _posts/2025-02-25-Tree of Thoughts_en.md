---
title: "[Paper Review] Tree of Thoughts: Deliberate Problem Solving with Large Language Models (NeurIPS 2023)"
date: 2025-02-25 00:00:00 +0900
categories:
  - LLM
tags:
  - Reasoning
  - LLM
  - NLP
  - ToT
  - NeurIPS 2023
---

Summary: Expanding the Chain-of-Thought (CoT) approach, this paper proposes the Tree of Thoughts (ToT) framework to enable large language models (LLMs) to perform systematic problem-solving.

---

# 1. Introduction

<img alt="image" src="https://github.com/user-attachments/assets/acb989e7-b1c0-4d34-b609-58de8d1f05db" />

- Existing large language models (LLMs) can perform general problem-solving but remain **limited by their left-to-right, token-level sequential decision-making process**. This leads to challenges in tasks requiring **exploration, strategic foresight (lookahead), and the consideration of early decisions' importance**.

- To address these limitations, this paper introduces the **Tree of Thoughts (ToT)** framework. ToT generalizes the existing **Chain of Thought (CoT)** approach by allowing LLMs to explore multiple **intermediate thoughts**, consider different solution paths, and self-evaluate to make optimal decisions.

- Through this framework, LLMs gain the following capabilities:

  1. **Exploring multiple reasoning paths** to find the best solution.
  2. **Performing self-evaluation** to determine the most promising choices.
  3. **Backtracking or predicting future outcomes** (global decision-making) when necessary to refine decisions.

- Experimental results demonstrate that ToT significantly outperforms CoT-based methods in solving complex problems such as **mathematical puzzles (Game of 24), creative writing, and mini crosswords**.

  - For example, in the **Game of 24**, GPT-4 with CoT prompting achieves only a **4% success rate**, whereas the ToT approach reaches an **impressive 74% success rate**.

- ToT enables LLMs to go beyond simple pattern recognition by incorporating **systematic exploration and planning**, contributing to the development of **more powerful AI systems with advanced problem-solving capabilities**.

---

# 2. Background

- Existing large language models (LLMs) can perform various problem-solving tasks but remain **limited by their token-level, left-to-right sequential generation**, lacking capabilities such as **systematic exploration, strategic foresight, and backtracking**.

- Cognitive science research suggests that humans combine **fast, automatic intuitive thinking (System 1)** with **slow, deliberate logical reasoning (System 2)** when solving problems.

  - Current LLMs primarily operate in a **System 1-like manner**, but incorporating **System 2 capabilities such as planning, searching, and evaluation** could significantly enhance their problem-solving abilities.

- Existing LLM-based problem-solving approaches include:

  1. **Input-Output (IO) Prompting**
     - Directly generates an output from a given input, without explicitly revealing the reasoning process.
  2. **Chain of Thought (CoT) Prompting**
     - Breaks the problem-solving process into multiple intermediate **thoughts**, enhancing logical reasoning.
  3. **Self-Consistency with CoT (CoT-SC)**
     - Generates multiple CoT reasoning paths for the same problem and selects the most frequently occurring answer to improve reliability.

- However, these existing methods have limitations:
  - **CoT follows a single reasoning path**, without exploring multiple possible solutions.
  - **CoT-SC only applies majority voting to the final answer**, lacking exploration and optimization within individual reasoning steps.
  - **A systematic search mechanism for planning, backtracking, and evaluation is necessary**.

---

# 3. Tree of Thoughts: Deliberate Problem Solving with LMs

- **Tree of Thoughts (ToT)** is a novel problem-solving framework designed to overcome the limitations of existing LLMs.
  - The **CoT approach explores only a single reasoning path**, without considering multiple alternatives when generating intermediate thoughts.
  - **ToT, on the other hand, enables the exploration of multiple reasoning paths**, allowing the model to evaluate and select the optimal route through **deliberate problem-solving**.

## 3.1 Core Ideas of ToT

- Frames problem-solving as a **search process**, enabling LLMs to experiment with multiple solution approaches.
- Inspired by AI and cognitive science, it assumes that **"problem-solving involves exploring different paths within a tree-like structure."**
- **ToT represents each thought as a node in a tree**, exploring different paths to discover the most optimal solution.

## 3.2 Key Components of ToT

1. **Decomposition of Thought Steps**

   - While CoT generates reasoning as a single continuous text, ToT **explicitly decomposes thought steps** based on task characteristics, increasing the potential for exploration.

2. **Thought Generation (G)**

   - Allows LLMs to generate multiple alternative thoughts at each state.
   - Uses two methods:
     - **Independent Sampling:** Generates multiple independent thoughts (CoT-based).
     - **Sequential Proposing:** Iteratively proposes new thoughts based on previous ones.

3. **State Evaluation (V)**

   - Enables LLMs to **select the most promising reasoning path** by performing self-evaluation.
   - Evaluation methods:
     - **Individual Evaluation:** Assesses each thought independently (e.g., assigning a score).
     - **Comparative Voting:** Selects the optimal thought from multiple candidates.

4. **Search Algorithm**

   - Uses **Breadth-First Search (BFS) and Depth-First Search (DFS)** to explore the tree structure.
   - Adapts the search strategy based on problem characteristics:
     - **BFS (Breadth-First Search):** Explores multiple paths simultaneously to find the optimal route.
     - **DFS (Depth-First Search):** Explores a single promising path deeply while pruning inefficient routes as needed.

<img alt="image" src="https://github.com/user-attachments/assets/0432cc84-f329-44e4-9a72-478a3795140d" />

## 3.3 Advantages of ToT

- **Generality**

  - Encompasses existing approaches such as **IO Prompting, CoT, CoT-SC, and Self-Refinement**.

- **Modularity**

  - Thought decomposition, generation, evaluation, and search algorithms can be independently combined and applied.

- **Adaptability**

  - Exploration and evaluation strategies can be adjusted for different problem types.

- **Convenience**

  - Can be implemented using **pretrained LLMs without additional training**, leveraging only prompts.

- ToT is a **framework designed to enhance LLMs' problem-solving capabilities**, and subsequent experiments demonstrate its superiority over existing methods in various tasks.

---

# 4. Experiments

<img alt="image" src="https://github.com/user-attachments/assets/8d883e2a-c550-4e9b-bb44-71adb7eb671f" />

- The study conducts experiments on three complex tasks to demonstrate that **Tree of Thoughts (ToT)** outperforms existing methods in problem-solving.
- The tested problems are:
  1. **Game of 24** (Mathematical Reasoning)
  2. **Creative Writing** (Coherent Text Generation)
  3. **Mini Crosswords** (Word Puzzle Solving)

---

## 4.1 Game of 24 (Mathematical Problem Solving)

<img alt="image" src="https://github.com/user-attachments/assets/2809ef6c-5ec2-4dda-b5ff-b1614b87c61d" />

- **Problem Description**

  - Given four numbers (e.g., 4, 9, 10, 13), the task is to **use arithmetic operations to form an equation that results in 24**.

- **Experimental Setup**

  - A set of 100 challenging problems was selected for testing.
  - Comparison with existing methods:
    - **IO Prompting**: Directly generates an answer without reasoning steps.
    - **CoT Prompting**: Solves the problem using intermediate reasoning steps.
    - **CoT-SC (Self-Consistency)**: Generates multiple CoT responses and selects the most frequent answer.
    - **IO + Refine**: Iteratively improves incorrect answers through refinement.

- **ToT Implementation**

  - Reformulates the problem as a **tree search task**.
  - At each step, explores multiple arithmetic operations and **selects the most promising path**.

- **Results Comparison**

  | Method             | Success Rate |
  | ------------------ | ------------ |
  | IO Prompting       | 7.3%         |
  | CoT Prompting      | 4.0%         |
  | CoT-SC (k=100)     | 9.0%         |
  | ToT (b=1)          | 45%          |
  | ToT (b=5)          | 74%          |
  | IO + Refine (k=10) | 27%          |
  | IO (best of 100)   | 33%          |
  | CoT (best of 100)  | 49%          |

  - **ToT (b=5) achieves a 74% success rate**, significantly outperforming all existing approaches.

---

## 4.2 Creative Writing (Coherent Text Generation)

<img alt="image" src="https://github.com/user-attachments/assets/04b65615-7240-4313-9631-7e9c4fef06cf" />

- **Problem Description**

  - Given **four specific sentences**, the task is to write a **coherent four-paragraph story** incorporating them.

- **Experimental Setup**

  - 100 randomly selected sentence sets were tested.
  - Evaluation criteria:
    - GPT-4-based automatic evaluation (scores from 1 to 10).
    - Blind comparison with human evaluators.

- **ToT Implementation**

  - ToT first **plans the overall structure of the story before writing**.
  - **Two-stage tree search approach**:
    1. Generates multiple story outlines (5) and selects the best one through **voting**.
    2. Writes multiple full-text versions (5) based on the selected outline and chooses the best one through **voting**.

- **Results Comparison**

  | Method        | GPT-4 Score (1-10) | Human Preference Comparison |
  | ------------- | ------------------ | --------------------------- |
  | IO Prompting  | 6.19               | -                           |
  | CoT Prompting | 6.93               | 21% preference              |
  | ToT           | 7.56               | 41% preference              |

  - **ToT achieves the highest score (7.56) and is the most preferred method in human evaluation.**

---

## 4.3 Mini Crosswords (Word Puzzle Solving)

<img alt="image" src="https://github.com/user-attachments/assets/cad40a56-01c0-4e05-b18d-3f51f5e93e19" />

- **Problem Description**

  - A 5×5 crossword puzzle where the model must **fill in words that match given clues**.

- **Experimental Setup**

  - Measures accuracy in completing the puzzle based on the provided hints.

- **ToT Implementation**

  - At each step, **generates multiple candidate words and selects the most appropriate one**.
  - Unlike existing methods (CoT), it evaluates various possibilities before finalizing an answer.

- **Results Comparison**

  - ToT demonstrates **higher accuracy and better word arrangement consistency compared to GPT-4's standard approach**.

---

## 4.4 Conclusion

- ToT significantly outperforms existing methods by **applying systematic search and evaluation strategies**.
- It excels particularly in **mathematical reasoning (Game of 24) and creative problem-solving (Creative Writing, Mini Crosswords)**, proving its superiority over **CoT-based approaches**.
- Unlike conventional LLM approaches (IO, CoT, CoT-SC), **ToT explores multiple solution paths using tree search to identify the optimal answer**.

---

# 5. Related Work

- This study extends existing **language model-based problem-solving techniques** by proposing the **Tree of Thoughts (ToT) framework**, drawing influence from multiple research domains.

## 5.1 Chain of Thought (CoT) Prompting

- **CoT (Chain of Thought) prompting** enables LLMs to solve complex problems step by step.
- Instead of generating final answers directly, LLMs **explicitly express intermediate reasoning steps (thoughts)** to enhance logical inference.
- However, CoT is **limited to a single reasoning path**, lacking the ability to explore multiple possibilities or evaluate alternatives.
- ToT overcomes this limitation by introducing **a mechanism to explore and assess multiple thought paths**.

## 5.2 Self-Consistency in CoT

- **CoT-SC (Self-Consistency)** generates multiple CoT responses for the same problem and **selects the most frequently occurring answer** to improve reliability.
- While majority voting enhances robustness, **it lacks the ability to explore alternative reasoning paths within individual thought steps**.
- ToT extends CoT-SC’s multi-sampling concept by incorporating **search and evaluation mechanisms, rather than relying solely on frequency-based selection**.

## 5.3 Tool Use by Language Models

- Recent research suggests that LLMs can **enhance problem-solving by utilizing external tools** (e.g., calculators, APIs).
- For example, **Toolformer** explores how LLMs autonomously call external tools for problem-solving.
- Similarly, ToT equips LLMs with **intrinsic evaluation and search capabilities**, enabling them to engage in **deep reasoning rather than merely generating direct answers**.

## 5.4 Search Algorithms and AI Planning

- **Search and planning algorithms** are fundamental to traditional AI problem-solving.
- Techniques such as **A\* search and Monte Carlo Tree Search (MCTS)** are widely used in games and optimization problems.
- ToT integrates such search techniques into LLMs, enabling them to **evaluate multiple solution paths and select the most optimal one**.

## 5.5 Human Problem-Solving Theories

- Psychological studies suggest that **humans solve problems through tree-structured exploration**.
- For example, **Newell & Simon (1950s) introduced the concept of state-space search** to describe human problem-solving strategies.
- ToT reflects these cognitive science principles by **allowing LLMs to systematically explore and evaluate different reasoning paths**.

## 5.6 Conclusion

- ToT **addresses the limitations of CoT and other LLM-based problem-solving methods** by integrating **traditional search algorithms and cognitive science-based problem-solving theories**.
- While prior research focused on **single-path reasoning**, ToT **introduces a tree-structured approach that explores multiple reasoning paths to determine the optimal solution**.

---

# 6. Discussion

- This paper explores how the **Tree of Thoughts (ToT) framework** enhances **LLMs' problem-solving abilities**.
- This section discusses the strengths, limitations, and future research directions of ToT.

---

## 6.1 Key Advantages of ToT

1. **Enhanced Problem-Solving Capability**

   - Unlike traditional methods that **focus on direct answer generation**, ToT **performs search-based problem-solving**, achieving **higher success rates** than CoT and CoT-SC.
   - By **exploring multiple reasoning paths and selecting the best one through evaluation**, ToT generates **more reliable and accurate answers**.

2. **Modularity and Scalability**

   - ToT can be **applied to various problem types** and **independently integrates thought decomposition, generation, evaluation, and search algorithms**.
   - It is easily **compatible with existing LLM-based approaches**, making it a **flexible problem-solving framework**.

3. **Transparent Reasoning Process**

   - Unlike conventional LLMs, which operate as a "black box," ToT **explicitly outlines the reasoning process at each step**.
   - This transparency **improves trust in AI systems and facilitates interpretable AI development**.

---

## 6.2 Limitations and Challenges of ToT

1. **Increased Computational Cost**

   - Since ToT **explores multiple reasoning paths**, it **requires significantly more computation than CoT**.
   - Future research should focus on designing **optimal search strategies that balance performance and efficiency**.

2. **Optimization of Search Algorithms**

   - Current implementations rely on **basic search techniques like BFS and DFS**, but **more sophisticated methods (e.g., A\*, MCTS) could further enhance performance**.
   - Research is needed to **automatically select the best search strategy for different problem types**.

3. **Reliability of Thought Evaluation**

   - ToT relies on LLMs for self-evaluation, but **the evaluation process is not always accurate**, leading to **potential misjudgments**.
   - This limitation could be addressed by **introducing more refined evaluation techniques or incorporating human feedback mechanisms**.

---

## 6.3 Future Research Directions

1. **Search Optimization**

   - Integrating ToT with **advanced search algorithms** (e.g., A\*, MCTS) could **improve efficiency and problem-solving capabilities**.

2. **Self-Supervised Learning for ToT**

   - Future ToT models could **continuously refine their search strategies through self-supervised learning**, improving their ability to solve complex problems over time.

3. **Expansion to Multi-Modal Problem Solving**

   - While ToT currently focuses on language-based reasoning, it could be extended to **multi-modal applications** such as **image understanding, mathematical reasoning, and code generation**.
   - For example, **ToT could be applied to solving visual puzzles or programming challenges**.

4. **Human-AI Collaboration in ToT**

   - ToT could become even more powerful when combined with **human-in-the-loop systems**, allowing users to **interactively refine thought paths or provide feedback**.
   - Research could focus on **designing interfaces where humans can guide ToT’s reasoning process** for improved decision-making.

---

## 6.4 Conclusion

- ToT **overcomes the limitations of traditional LLMs’ sequential reasoning** by introducing **structured search and evaluation mechanisms**.
- With **higher problem-solving performance, modularity, scalability, and improved interpretability**, ToT represents a **significant advancement in AI reasoning**.
- Future research should focus on **optimizing search and evaluation mechanisms** to further enhance its effectiveness.

---

# Reader Feedback

- As mentioned in the limitations section, **optimizing search strategies based on problem type** is crucial. In some **simpler problems, ToT may not be necessary**, and CoT or even earlier-stage reasoning might suffice. **Determining when ToT is needed** could help **reduce unnecessary inference costs** and improve efficiency.

- This **meta-search approach** could be implemented using **LLMs or smaller models (SLMs)** to **select the best problem-solving strategy (CoT, CoT-SC, ToT, etc.) before execution**. This would ensure **cost-effective and optimized reasoning** based on the complexity of the task.
