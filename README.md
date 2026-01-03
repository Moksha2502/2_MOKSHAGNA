
---

# Drug–Drug Interaction Checker (Graph + RAG)

## Problem Statement

When a patient is prescribed multiple medications, some drugs may react with each other. These reactions are called Drug–Drug Interactions (DDIs). DDIs can:

* Reduce the effectiveness of treatment
* Cause unexpected side-effects
* Lead to serious or life-threatening complications

Doctors and pharmacists usually check interactions manually using references or tools. This can be time-consuming and may still lead to oversight.

### Objective

To build a system that automatically:

1. Takes a list of medicines as input
2. Detects risky interaction pairs
3. Explains the interaction mechanism in simple language
4. Uses both graph-based relationships and RAG (Retrieval-Augmented Generation) for clarity and accuracy

---

## Dataset

We use the DDInter dataset (Drug–Drug Interactions).

Dataset link:
[https://www.kaggle.com/datasets/montassarba/drug-drug-interactions-database-ddinter](https://www.kaggle.com/datasets/montassarba/drug-drug-interactions-database-ddinter)

The dataset contains:

* Drug names
* Drug IDs
* Drug–drug interaction pairs
* Description of interactions
* Mechanism or effect details

This allows us to build a knowledge graph of how drugs interact.

---

## System Design

### Step 1: Input

The user enters a list of medicines, for example:

Warfarin, Aspirin, Ibuprofen

---

### Step 2: Build a Drug Interaction Graph

We use a graph structure where:

* Each node represents a drug
* Each edge represents an interaction between two drugs
* Edge properties store risk details and mechanism text

This helps quickly check whether two drugs interact.

---

### Step 3: Interaction Detection

For every pair of drugs entered by the user, we check if an interaction exists in the dataset.

If an interaction exists:
We flag it and retrieve the mechanism text.

If no interaction exists:
We report that no interaction was found.

---

### Step 4: RAG-Based Explanation

We use LangChain to retrieve relevant text from the dataset and convert it into simple explanation.

So instead of only saying “Interaction detected”, the system explains why the interaction is risky.

Example explanation:
Warfarin and Aspirin together increase bleeding risk because both reduce clotting ability in the body.

---

## Technology Stack

Programming Language: Python
Graph Library: NetworkX or Neo4j (optional)
AI Pipeline: LangChain (RAG-based explanation)
Data Handling: Pandas

---

## Example Output

### Input

Warfarin, Aspirin, Metformin

### Output

Pair: Warfarin — Aspirin
Risk Level: High
Interaction Reason: Increased bleeding risk
Explanation: Both drugs interfere with clotting, which increases bleeding tendency.

Pair: Warfarin — Metformin
Risk Level: None Detected

Pair: Aspirin — Metformin
Risk Level: None Detected

---

## Expected Outcomes

* Detect risky drug interaction pairs
* Provide reason and mechanism behind the interaction
* Ensure responses are explainable
* Improve medication safety
* Support healthcare decision-making

This can be useful for doctors, pharmacists, hospitals, and healthcare applications.

---

## Assumptions

* Users provide correct drug names
* Dataset covers the majority of common interactions
* The system currently checks interactions only between two drugs at a time
* Severity level is based on dataset information
* Patient-specific factors such as age, dosage, or disease conditions are not considered in this version

---

## Future Enhancements

* Add risk severity scoring
* Add patient-specific risk assessment
* Build an easy-to-use web interface
* Support multiple languages
* Integrate real-time medical databases

---

