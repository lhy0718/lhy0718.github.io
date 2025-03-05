---
title: "[Paper Review] Evaluating Intention Detection Capability of Large Language Models in Persuasive Dialogues (ACL 2024)"
date: 2025-03-05 00:00:00 +0900
categories:
  - Paper Review
tags:
  - NLP
  - ACL 2024
  - Persona-based Dialogue

---

Abstract: The study explores intention detection in persuasive multi-turn dialogues using large language models, highlighting the importance of conversational history and incorporating face acts to analyze intentions effectively.

---

# 1 Introduction

- Identifying speaker intentions is essential for smooth conversations.
- Example: Alice asks Bob for a charity donation; Bob's evasive response indicates hesitance without outright refusal.
- Speaker intentions can be conveyed indirectly and vary by context.
- People instinctively estimate these intentions, which is vital for natural communication.
- Recent advancements in large language models (LLMs) like ChatGPT and GPT-4 facilitate human-like dialogue.
- Ongoing research focuses on developing dialogue systems incorporating LLMs.
- LLMs are applied in real-world scenarios and may effectively detect speaker intentions.
- Existing datasets like GLUE evaluate LLMs' understanding of natural language but lack focus on intention detection.
- This study introduces a new dataset for measuring LLMs' ability to identify intentions in persuasive conversations.
- The dataset features multiple-choice questions that consider the context of past utterances.
- Persuasive conversations often require careful consideration of other parties' feelings and perspectives.
- The dataset utilizes the concept of "face" (Goffman, 1967) to assess intentions linked to social relationships.
- By grouping intentions by face, clarity in analysis and insights improves.
- The research includes an assessment of LLMs' intention detection capabilities and identifies particularly challenging types of intentions.
- Contributions:
  - Developed a dataset for evaluating intention detection from persuasive dialogues.
  - Evaluated state-of-the-art LLMs like GPT-4 and ChatGPT on their ability to detect utterance intentions, highlighting their mistakes and difficulties.

---

# 2 Background

- Explanation of face and face acts.
- Overview of existing dialogue data used in the research.
- Discussion of previous studies on:
  - Dialogue comprehension.
  - Intention detection.

---

# 2.1 Face and Face Acts

- **Definition of Face**: 
  - Face is a primary human need related to social relationships.
  - Introduced by Erving Goffman in 1967.

- **Politeness Theory**: 
  - Developed by Brown and Levinson to analyze verbal behaviors affecting face.
  - Systematizes face-related behaviors as politeness strategies.

- **Types of Face**:
  - **Positive Face**: 
    - The desire to be recognized, admired, and liked by others.
  - **Negative Face**: 
    - The desire to maintain one's freedom and autonomy.

- **Face Acts**: 
  - Utterances that affect the face of oneself or others.
  - **Face Threatening Act (FTA)**: 
    - Acts that threaten face.
  - **Face Saving Act (FSA)**: 
    - Acts that aim to preserve face, e.g., praising or alleviating burdens.

- **Politeness Strategy**: 
  - People typically avoid threatening faces to manage relationships.
  - If they must threaten face, they use strategies to minimize the threat (e.g., implying needs, apologizing).

- **Application in Dialogue**: 
  - Dutt et al. (2020) applied face acts to analyze persuasive dialogues.
  - Face acts are crucial for successful persuasion.

- **Machine Learning Model**: 
  - Developed to track conversation dynamics using face acts and history.
  - Face acts categorized based on three criteria:
    - Direction: Speaker or hearer (s/h).
    - Type: Positive or negative face (pos/neg).
    - Action: Saved or attacked face (+/-).

- **Example of Persuasive Situation**: 

<img width="397" alt="image" src="https://github.com/user-attachments/assets/047dc76e-41cc-4503-8092-a27293678165" />

<img width="397" alt="image" src="https://github.com/user-attachments/assets/ef0fc77a-9d5c-444a-b7bf-cb87db6d5bd0" />

  - **Persuader (ER)**: Person attempting to change the mind of the other.
  - **Persuadee (EE)**: Person whose mind is being changed.
  - Requesting something from EE is a face act categorized as hneg- (threatens EE's freedom).
  - Validating an argument supports positive face, categorized as spos+.

---

# 2.2 Dataset Annotated with Face Acts

- The dataset is created by Dutt et al. (2020) and focuses on dialogues related to persuading donations to Save the Children (STC).
- There are two main participants in the dialogues:
  - **Persuader (ER)**: The individual seeking to convince another to donate.
  - **Persuadee (EE)**: The individual being persuaded.
- The dataset includes various utterances, with some categorized as "other," which includes greetings, fillers, and unrelated remarks.
- The dialogues were sourced from Wang et al. (2019), and only one face act label is assigned to each utterance.
- It is acknowledged that some utterances could have multiple face acts, but these instances make up only 2% of the dataset.
- To simplify annotation, only one face act is randomly selected from the possible options and used as the gold label.
- Example dialogue from the dataset illustrates the interaction:
  - ER asks about interest in donating.
  - EE inquires about the charity.
  - ER provides information about STC.
  - EE reveals limited prior knowledge about the charity.
- Some face act labels used in the example include:
  - "hneg-" for negative acknowledgment.
  - "hpos+" for positive acknowledgment.
  - "spos+" for supportive remarks.

---

# 2.3 Intention Detection

- **Importance in Dialogue Systems**:
  - Essential for task-oriented dialogue systems to understand user objectives.
  - Systems must classify utterances to determine if they fall within their operational domain.

- **Intention Detection Tasks**:
  - Typically involve classifying utterances into predefined intention labels.
  - Examples of specific domains include travel and banking, while some datasets encompass multiple domains.

- **Representative Datasets**:
  - SNIPS is highlighted as a significant dataset in the field.
  - Modern language models (LLMs) like GPT-2 show high performance in intention detection tasks.

- **Model Limitations**:
  - Many existing studies focus on intent prediction solely from individual utterances, without incorporating conversational context.

- **Contextual Studies**:
  - A few studies consider contextual information for intention detection.
  - Cui et al. (2020) developed a dataset to evaluate dialogue understanding by focusing on next utterance predictions within conversational settings.

- **Persuasive Conversations**:
  - Dutt et al. (2020) introduced a model that incorporates conversational context to predict intentions in persuasive dialogues.
  - They focused on face acts as intention labels, showing the model's capability in intention detection but did not use LLMs.

- **Unexplored Areas**:
  - There is a lack of research on how well LLMs can understand intentions in multi-turn persuasive dialogues.

---

# 3 Data

- Previous studies on intention detection often did not utilize multi-turn dialogue data.
- A persuasive dialogue dataset by Dutt et al. (2020) suggests predicting face acts from utterances but face acts are abstract intentions that are not intuitive for humans.
- Face acts are likely insufficiently learned by LLMs due to their infrequency in pretraining data.
- To effectively evaluate LLMs' intention detection capability, it's necessary to modify the approach for zero-shot or few-shot scenarios.
- The study transforms face acts into understandable intention descriptions in natural language.
- Each dataset entry consists of conversational history and four intention descriptions for the last utterance.
- The task output involves selecting one description from four options, akin to a reading comprehension format.
- This format is inspired by past dialogue reasoning studies and is commonly used to assess LLMs’ reasoning abilities.
- The persuasive dialogue dataset was partitioned into training, development, and test subsets in an 8:1:1 ratio, focusing solely on the test subset for evaluation.
- The section details the development process of the evaluation dataset, including:
  - Definition of intention descriptions annotated into utterances.
  - Annotation process through crowdsourcing for each utterance.
  - Selection of distractors to create the four option choices.

---

# 3.1 Preparation of Intention Description

- Dutt et al. (2020) provided various intention descriptions in persuasive contexts alongside associated face acts.
- Adaptation and expansion of these descriptions were conducted.
- The descriptions were annotated to align with specific utterances.
- New descriptions were created to cover all utterances in the development data.
- Broader intention descriptions were refined into more specific ones.
- A total of 42 intention descriptions were curated and presented in Table 2.

---

# 3.2 Intention Annotation

- Selected 30 dialogues from the persuasion dialogue dataset for test data.
- Annotated intention descriptions to utterances, focusing on face act labels by Dutt et al. (2020) for their emotional impact.
- Used crowdworkers from the US via Amazon Mechanical Turk (AMT) for the annotation process.
- Provided fair compensation, with an average hourly wage of $12 for workers.
- Conducted three rounds of pilot tests to refine instructions and select high-quality annotators.
- Finalized instructions for annotation available in Appendix A.
- Workers read entire conversations and assigned intention descriptions from a provided set of candidate descriptions.
  - Descriptions were categorized under the same face act as the utterance.
  - Example: For an EE's utterance with face act hpos-, possible intentions included doubt or lack of interest.
- Each utterance was annotated by three workers, resulting in three intention descriptions per utterance.
- Majority vote determined final annotation; gold labels were assigned if there was agreement from at least two workers.
- A total of 691 utterances were annotated, with 620 receiving agreement from at least two annotators.
- Developed an intention classification problem for the 620 agreed-upon utterances.
- Measured annotator agreement using Krippendorff’s alpha, yielding a value of 0.406, indicating moderate agreement.
- Additional details on annotator agreement are found in Appendix B.

---

# 3.3 Question Creation

- **Data Collection**: 620 utterances were initially annotated with intention descriptions.
- **Utterance Concatenation**: 
  - Consecutive utterances sharing the same intention descriptions were combined.
  - This step is crucial as some intentions become clear only after listening to subsequent utterances.
- **Outcome**: The concatenation process resulted in 549 utterances annotated with intention descriptions.
- **Question Formation**:
  - Multi-choice questions were created from these 549 utterances.
  - For each utterance, three distractors were randomly selected from a predefined description pool.
- **Appendices**:
  - Appendix C provides additional details on the utterance concatenation process.
  - Appendix D outlines the rules for the distractor selection process.
- **Data Statistics**: Table 3 includes specific statistics about the data used.

---

# 4 Experiment

- The goal is to evaluate how effectively Language Models (LLMs) detect intentions in persuasive dialogues.
- Various sizes of LLMs were tested to observe the impact of model size on intention detection ability.
  - Models used include:
    - **GPT-4** and **ChatGPT** from OpenAI
    - **Llama 2-Chat** (Meta)
    - **Vicuna** (LMSYS)
- Prompts provided to the LLMs contained:
  - Information for detecting intentions (conversational context, task explanation, conversational script, and a four-option question).
  - Designed in a zero-shot Chain-of-Thought style, dividing the answering process into two phases:
    - Reason explanation: LLMs state whether intentions are explicit or implied.
    - Option selection: LLMs choose the best option based on reasoning.
- Memory constraints limited history length to the past ten utterances for Llama 2-Chat and Vicuna models.
- Human performance benchmarking was conducted using workers from AMT (Amazon Mechanical Turk):
  - Workers selected intention descriptions for the last utterance from four options, ensuring no prior knowledge of gold intention descriptions.
  - Majority vote among three workers was used to determine the final answer.
- Key statistics:
  - 549 questions across 30 dialogues
  - Average of 18.3 questions and 30.8 turns per dialogue
  - Average words per utterance: 11.99, per description: 10.61
- Model performance results:
  - The smallest model achieved over 50% accuracy; GPT-4 exceeded 90%.
  - Larger models consistently showed improved accuracy.
  - Notably, LLMs struggled with identifying intentions categorized as hpos-.
    - GPT-4 detected hpos- intentions correctly in only 1 out of 7 cases.
- The section plans to address the issues faced by smaller LLMs and analyze the utterances where LLMs, particularly GPT-4, encountered challenges in intention detection.

---

# 4.1 Behavior of Smaller LLMs

<img width="753" alt="image" src="https://github.com/user-attachments/assets/f480365c-26ab-46c7-b3b7-44418e958ed8" />

- **Comparison with GPT-4:**
  - GPT-4 answered over 90% of questions correctly.
  - Smaller models, including ChatGPT and Llama 2-Chat-70B, faced difficulties in inference.

- **Problem Types:**

  - **Intention-related Problems:**

    <img width="814" alt="image" src="https://github.com/user-attachments/assets/1accf2da-7b3a-416f-ab9f-6108c4882553" />

    - Flawed interpretation of intention leads to incorrect answers.
    - Smaller models occasionally overinterpret intentions:
      - Example: GPT-4 accurately inferred intentions regarding donations, while smaller models incorrectly concluded that the speaker had no intention to donate based on overextensions of the conversation.


  - **Non-intention-related Problems:**
 
    <img width="814" alt="image" src="https://github.com/user-attachments/assets/f8a39d97-1ed8-44ee-b058-d47e86b733ad" />

    - Issues like generation loops and misinterpreting utterances unrelated to the main task.
    - Complexity of prompts poses comprehension challenges for smaller models.
    - Logical inconsistencies in responses were prevalent:
      - Examples show Llama 2-Chat-70B frequently selected the last answer option without proper evaluation.
      - While option D had a 25.7% correctness overall, Llama 2-Chat-70B chose it 31.9% of the time, indicating a tendency to select the last option.
    - Inconsistencies and poor option selection significantly reduced the performance of smaller models.

---

# 4.2 About hpos-

- **Weakness of LLMs**: 
  - LLMs struggle particularly with interpreting "hpos" utterances where:
    - ER condemns EE’s hesitation to donate.
    - EE expresses doubts about ER’s credibility.

- **GPT-4's Mistakes**: 
  - Instances of misunderstanding intentions behind EE's utterances are largely attributed to flawed questions highlighted in the limitations.
  - Focus of examination is on how GPT-4 interprets ER's criticisms of EE.

## 4.2.1 Patterns in Our Dataset

- **Two Primary Patterns of Criticism by ER**:

  <img width="397" alt="image" src="https://github.com/user-attachments/assets/2fd2726d-f738-4d01-b722-6a2aaf9d9973" />

  1. ER questions EE’s spending habits, promoting a redirection of funds to charity (Save the Children).
     - Example: ER asks about wasteful spending on junk food.
  2. ER brings up financially struggling individuals to elicit guilt in EE for inaction.
     - Example: ER highlights children suffering due to lack of donations.

- **GPT-4's Recognition**:

  <img width="397" alt="image" src="https://github.com/user-attachments/assets/40ff17c3-37af-4de8-a6ee-a1bea695c59a" />
  
  - GPT-4 incorrectly identified many of these utterances as non-critical or having different intentions, as detailed in subsequent tables.

## 4.2.2 Artificially Created Dataset
- **Methodology**:
  - Created scenarios testing perception of critical utterances by generating persuasive dialogues where EE hesitates to donate.
  - 90 utterances were judged for their critical nature by both GPT-4 and human annotators.

- **Findings**:
  - Majority of human judgments categorized utterances as motivating rather than critical:
    - 85 as 'motivating', 4 as 'criticizing', 1 as 'confirming donation amount'.
  - GPT-4’s interpretations closely aligned with human judgments for 87 utterances.

- **Contrast Between Critical and Non-Critical**:
  - Examples of utterances categorized as critical had a more sarcastic and obvious tone, while non-critical utterances relied on emotional appeals.
  - Emotional appeals were viewed as strategies to boost donations, whereas sarcastic remarks were recognized for their implicit critique.

- **Further Inquiry**:
  - The difference in how guilt-tripping strategies motivate donations versus being perceived negatively is suggested as a key area for future exploration regarding human versus LLM judgments.

---

# 5 Conclusion

- The study assesses the capability of large language models (LLMs) to detect intentions in multi-turn persuasive dialogues.
- A dataset was created for evaluation, revealing limitations that highlight areas for improvement in intention detection methods.
- Key findings include:
  - Inappropriate labeling may lead to incorrect intention representation due to limited label sets.
  - Multiple interpretations of intentions can complicate the detection task, making singular answers insufficient.
  - The specific dataset is not fully representative of various dialogue types, limiting the generalizability of findings.
- Future research should focus on improving dataset diversity and developing training data for fine-tuning LLMs.
- The study raises ethical concerns regarding the potential misuse of LLMs, particularly in persuading individuals or spreading misinformation.
- Acknowledgments were made for support received and the contributions of various individuals to the research.

---

# 6 Limitations

- **Inappropriate Labeling**: 
  - Some questions in the dataset could not be labeled appropriately due to the limited, predetermined label set.
  - This leads to inaccurate intention descriptions based on misannotated face acts.

- **Multiple Correct Answers**: 
  - The dataset inevitably contains questions where utterances can express multiple correct intentions.
  - This results in models inaccurately selecting intention descriptions due to a lack of a single correct option.

- **Insufficient Dataset for Evaluation**: 
  - The dataset is sparse in face act distributions, lacking examples of less frequent intentions.
  - The exclusive focus on persuasive dialogues limits generalizability; a diverse dataset is necessary for comprehensive analysis.

- **Bias in Generated Data**:
  - The additional experiment used GPT-4 generated conversations, which may reflect biases inherent in the model.
  - This could compromise the validity of the findings based on artificial conversation data.

- **Potential Ethical Concerns**:
  - While the study investigates LLMs' intention detection, insights may later be misused in applications with significant ethical implications.
  - The risk of LLMs misleading individuals or spreading misinformation exists if intent detection capabilities are exploited. 

- **Impact of LLMs’ Knowledge and Biases**:
  - Results may be affected by the aggressive knowledge and various biases of the LLMs employed in the study.

---

# 7 Ethical Considerations

- The study evaluates LLMs' intention detection capabilities without immediate severe ethical implications anticipated from its findings.
- If LLMs become precise in detecting intentions, they may be widely deployed as interactive agents across various fields.
- Potential misuse of LLMs:
  - Could manipulate human intentions leading to deception by malicious entities, posing risks of fraud.
  - Ability to disseminate misinformation, especially on social media, could result in widespread public confusion.
- This research involves models like ChatGPT and GPT-4, which inherently possess biases and aggressive knowledge that may influence results.
- Careful monitoring is necessary as the technology develops and integrates into dialogue systems.
