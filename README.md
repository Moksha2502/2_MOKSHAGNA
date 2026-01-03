
---

<h1 align="center">Drug–Drug Interaction Checker (Graph + RAG)</h1>

## Problem Statement

When a patient is prescribed multiple medications, some drugs may react with each other. These reactions are called **Drug–Drug Interactions (DDIs)**. DDIs can:

* Reduce the effectiveness of treatment
* Cause unexpected side-effects
* Lead to serious or life-threatening complications

Doctors and pharmacists usually check interactions manually using references or tools. This process can be time-consuming and may still result in oversight. Therefore, there is a need for an intelligent system that can automatically detect and explain drug interactions.

---

## Objective

The goal of this project is to build a system that:

* Accepts a list of medicines as input
* Detects risky interaction pairs
* Explains the interaction mechanism in clear and simple language
* Uses both **graph-based relationships** and **RAG (Retrieval-Augmented Generation)** for accuracy and explainability

---

## Dataset

This project uses the **DDInter: Drug–Drug Interaction Database**.

Dataset Link:
[https://www.kaggle.com/datasets/montassarba/drug-drug-interactions-database-ddinter](https://www.kaggle.com/datasets/montassarba/drug-drug-interactions-database-ddinter)

The dataset includes:

* Drug names
* Drug IDs
* Drug-drug interaction pairs
* Description of interactions
* Mechanism or effect details

This information allows us to build a **knowledge graph** representing drug interaction relationships.

---

## System Design

### Step 1: Input

The user provides a list of medicines.
Example:

```
Warfarin, Aspirin, Ibuprofen
```

---

### Step 2: Build a Drug Interaction Graph

A **graph structure** is constructed where:

* Each **node** represents a drug
* Each **edge** represents an interaction between two drugs
* Edge properties store:

  * Risk information
  * Mechanism or interaction description

This allows fast and structured detection of interactions.

---

### Step 3: Interaction Detection

For each drug pair in the input:

* If an interaction exists
  → It is flagged and the mechanism text is retrieved
* If no interaction exists
  → The system reports that no interaction was found

---

### Step 4: RAG-Based Explanation

LangChain is used to retrieve relevant dataset text and convert it into a simple explanation.

Example:

> Warfarin and Aspirin together increase bleeding risk because both reduce the clotting ability of blood.

This ensures responses are **clear, contextual, and explainable**.

---

## Technology Stack

* **Programming Language:** Python
* **Data Handling:** Pandas
* **Graph Processing:** NetworkX or Neo4j (optional)
* **AI / NLP Pipeline:** LangChain (RAG-based explanation)

---

## Example Output

### Input

```
Warfarin, Aspirin, Metformin
```

### Output

**Pair: Warfarin — Aspirin**
Risk Level: High
Interaction Reason: Increased bleeding risk
Explanation: Both drugs interfere with clotting, which increases bleeding tendency.

**Pair: Warfarin — Metformin**
Risk Level: None Detected

**Pair: Aspirin — Metformin**
Risk Level: None Detected

---

## Expected Outcomes

The system aims to:

* Detect risky drug-interaction pairs
* Provide clear explanations behind each interaction
* Improve medication safety
* Support healthcare decision-making
* Ensure explainable and transparent AI results

This solution can be useful for **doctors, pharmacists, hospitals, researchers, and healthcare applications**.

---

## Assumptions

* Users provide valid and correct drug names
* The dataset covers most common interactions
* The current system checks interactions pair-wise
* Severity interpretation is based on dataset descriptions
* Patient-specific conditions such as:

  * age
  * dosage
  * medical history
    are not considered in this version

---

## Future Enhancements

* Add severity-based risk scoring
* Include patient-specific risk assessment
* Develop an interactive web interface
* Support multilingual explanations
* Integrate real-time medical reference databases

---
