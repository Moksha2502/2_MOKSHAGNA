
---

# Drug–Drug Interaction Checker (Graph + RAG)

## 1. Problem You Are Solving

When a patient takes multiple medicines together, some drugs **interact** and may cause:

* Reduced treatment effectiveness
* Harmful side-effects
* Serious medical risk (bleeding, heart rhythm issues, etc.)

Doctors usually check this manually in reference tools. This is:

* Slow
* Hard when multiple drugs are taken
* Prone to error

So we are building an **AI-assisted Drug Interaction Checker** that:

1. Takes a list of medicines
2. Detects risky interactions
3. Explains the mechanism in simple language
4. Supports decisions with citations from trusted medical data

---

## 2. Dataset

We use the **DDInter: Drug–Drug Interaction Database**

Dataset link:
[https://www.kaggle.com/datasets/montassarba/drug-drug-interactions-database-ddinter](https://www.kaggle.com/datasets/montassarba/drug-drug-interactions-database-ddinter)

This dataset contains:

* Drug names
* Drug IDs
* Drug–drug interaction pairs
* Details about interaction mechanisms
* Descriptions and notes

This will be our **knowledge base**.

---

## 3. Technology Stack

We will use:

* **Python** — main language
* **Pandas** — data cleaning and loading
* **NetworkX or Neo4j** — to build the drug interaction graph
* **LangChain** — for Retrieval-Augmented Generation (RAG) explanation
* **Embeddings + Vector DB (Chroma/FAISS)** — to fetch relevant text snippets

Optional:

* **FastAPI** — to expose `/check_interactions` API endpoint

---

## 4. System Architecture (Step-by-Step)

### Step 1 — Data Preparation

Load dataset using Pandas.

We extract:

* drug_1
* drug_2
* interaction_description
* mechanism

We **normalize names** (lowercase, remove brackets, trim spaces) so matching is accurate.

---

### Step 2 — Build the Drug Interaction Graph

We create a **Graph where:**

* Each **node = a drug**
* Each **edge = interaction**
* Edge attributes store:

  * interaction description
  * severity (if available)
  * mechanism

This lets us quickly query:

“Does Drug A interact with Drug B?”

NetworkX makes this easy.

---

### Step 3 — Create the Vector Store (for RAG)

We take all **interaction descriptions + mechanism text** and convert them to embeddings.

Store them in:

* ChromaDB or FAISS

This allows **semantic search** when explaining the interaction.

---

### Step 4 — User Input

Example user input:

```
["Warfarin", "Aspirin", "Ibuprofen"]
```

We generate **all unique drug pairs**, for example:

* Warfarin – Aspirin
* Warfarin – Ibuprofen
* Aspirin – Ibuprofen

---

### Step 5 — Interaction Detection (Graph Query)

For each pair:

* Check if an edge exists in the graph
* If yes → flag as risky
* Retrieve mechanism text

This guarantees **fast lookup with high accuracy**

---

### Step 6 — RAG-Based Explanation

To avoid medical ambiguity, we pass the retrieved mechanism text into an LLM using LangChain.

The LLM:

* Summarizes
* Simplifies
* Explains risk in patient-friendly language

Example Output:

```
Warfarin and Aspirin increase bleeding risk.
Both reduce clotting, so taking them together raises bleeding tendency.
```

This makes the system **explainable and trustworthy.**

---


## 6. Example Output (What Judge Will See)

### Input

Warfarin, Aspirin, Metformin

### Output

1. Interaction Detected: Warfarin — Aspirin
   Severity: High
   Mechanism Summary:
   Both reduce blood clotting. When combined, bleeding risk increases significantly.
   Source: DDInter dataset snippet.

2. Interaction: Warfarin — Metformin
   Severity: None detected

3. Interaction: Aspirin — Metformin
   Severity: None detected

---

## 7. Assumptions

* Drug names must match dataset names
* Severity relies on dataset text
* We do not yet factor:

  * dosage
  * age
  * medical history
  * liver/kidney function

(We can mention these as future work)

---

## 8. Evaluation

We will evaluate:

1. Accuracy of detected interaction pairs
2. Relevance of RAG explanations
3. Response time
4. Reduction in hallucinations due to grounding

Optional metric:

* % overlapping interactions vs dataset truth

---

## 9. Future Enhancements

* UI dashboard for hospitals
* Severity-based color tagging
* Patient-specific personalization
* Dose-aware interaction scoring
* Doctor-assist recommendation engine

---

