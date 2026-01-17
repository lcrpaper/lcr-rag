# Conflict Annotation Guidelines

## Document Purpose

These guidelines define the annotation protocol for labeling knowledge conflicts
in retrieval-augmented generation (RAG) systems. Annotators classify conflicts
according to a 4-level taxonomy (L1-L4) based on the type of reasoning required
to resolve the conflict.

---

## 1. Conflict Taxonomy Overview

### L1: Temporal Conflicts
**Definition**: Conflicts resolvable by identifying which information is more recent.

**Characteristics**:
- Explicit temporal markers (dates, years, "as of", "currently")
- Information that changes over time (records, statistics, leadership positions)
- Clear temporal ordering possible between sources

**Examples**:
- "The CEO of Company X is John Smith (2021)" vs "The CEO of Company X is Jane Doe (2023)"
- "The population is 500,000" (2015 census) vs "The population is 520,000" (2023 estimate)

**Resolution Strategy**: Select the most recent information.

### L2: Numerical Conflicts
**Definition**: Conflicts involving measurable quantities resolvable through precision assessment.

**Characteristics**:
- Specific numerical values (counts, percentages, measurements)
- Statistical claims (averages, totals, rankings)
- Quantifiable differences between sources

**Examples**:
- "The tower is 324 meters tall" vs "The tower is 330 meters tall"
- "Sales increased 15%" vs "Sales increased 12%"

**Resolution Strategy**: Assess source reliability, measurement precision, or recency.

### L3: Entity Conflicts
**Definition**: Conflicts involving distinct entities requiring world knowledge to resolve.

**Characteristics**:
- Different named entities for the same role/position
- Contradictory attributions (author, inventor, location)
- Requires background knowledge to adjudicate

**Examples**:
- "The inventor of X was Edison" vs "The inventor of X was Tesla"
- "The capital is City A" vs "The capital is City B"

**Resolution Strategy**: Requires entity verification and contextual knowledge.

### L4: Semantic Conflicts
**Definition**: Conflicts involving interpretive differences or subjective claims.

**Characteristics**:
- Qualitative assessments (good/bad, effective/ineffective)
- Causal claims with different explanations
- Abstract or interpretive disagreements
- Opinion-based statements presented as facts

**Examples**:
- "The policy was successful" vs "The policy failed"
- "Climate change is caused by X" vs "Climate change is caused by Y"

**Resolution Strategy**: Requires deep reasoning, evidence weighing, or source evaluation.

---

## 2. Annotation Procedure

### 2.1 Input Format

Each annotation instance contains:
- **Query**: The user's question
- **Retrieved Documents**: 3-7 documents from retrieval system
- **Document Metadata**: Source, date, reliability indicators (when available)

### 2.2 Annotation Tasks

#### Task 1: Conflict Detection (Binary)
- **Question**: Do the retrieved documents contain conflicting information relevant to answering the query?
- **Labels**: YES (conflict present) / NO (no conflict)

#### Task 2: Conflict Type Classification (4-class)
- **Question**: What type of conflict is present?
- **Labels**: L1 (Temporal) / L2 (Numerical) / L3 (Entity) / L4 (Semantic)
- **Note**: Only annotate if Task 1 = YES

#### Task 3: Gold Answer Identification
- **Question**: What is the correct answer based on available evidence?
- **Output**: Free-text answer or "CANNOT DETERMINE"

### 2.3 Decision Flowchart

```
START
  │
  ▼
Do documents contain conflicting information?
  │
  ├── NO ──► Label: NO_CONFLICT, SKIP remaining tasks
  │
  ▼ YES
  │
Is the conflict resolvable by temporal comparison?
  │
  ├── YES ──► Label: L1 (Temporal)
  │
  ▼ NO
  │
Does the conflict involve specific numerical values?
  │
  ├── YES ──► Label: L2 (Numerical)
  │
  ▼ NO
  │
Does the conflict involve distinct named entities?
  │
  ├── YES ──► Label: L3 (Entity)
  │
  ▼ NO
  │
Is the conflict interpretive/semantic in nature?
  │
  └── YES ──► Label: L4 (Semantic)
```

---

## 3. Edge Cases and Clarifications

### 3.1 Multiple Conflict Types
If multiple conflict types are present, label the **dominant** conflict type:
- The one most relevant to answering the query
- The one requiring the most complex reasoning

### 3.2 Partial Conflicts
If conflict exists in only some documents:
- Still label as conflict if it affects answer reliability
- Note the number of conflicting vs. agreeing documents in metadata

### 3.3 Ambiguous Cases

#### Temporal vs. Numerical
- If a number changes over time (e.g., population statistics), label as **L1** if temporal markers are present
- If comparing measurement precision without temporal context, label as **L2**

#### Entity vs. Semantic
- Concrete entity disagreements (names, places) → **L3**
- Abstract interpretive claims about entities → **L4**

### 3.4 When to Use "CANNOT DETERMINE"

Use this label for gold answers when:
- Both sources appear equally reliable
- Insufficient context to adjudicate
- The conflict reveals genuine uncertainty in the domain

---

## 4. Quality Assurance

### 4.1 Inter-Annotator Agreement

**Target**: Cohen's κ ≥ 0.80 for conflict detection, κ ≥ 0.75 for type classification

**Achieved (from paper)**: κ = 0.83 overall (Appendix F)

### 4.2 Calibration Set

Before beginning annotation, annotators must complete:
1. **Training set**: 50 examples with feedback
2. **Qualification test**: 50 examples, must achieve ≥85% agreement with gold labels

### 4.3 Disagreement Resolution

For cases with annotator disagreement:
1. Independent third annotator provides tie-breaking vote
2. Adjudicator reviews persistent disagreements
3. Ambiguous cases flagged for guideline clarification

---

## 5. Data Collection Protocol

### 5.1 Source Selection

**Primary Sources**:
- NaturalQuestions (Google): Real user queries
- HotpotQA: Multi-hop reasoning queries

**Retrieval**:
- BM25 with parameters k1=0.9, b=0.4 (tuned for conflict detection)
- Top-5 documents per query

### 5.2 Sampling Strategy

To ensure balanced representation:
- 25-30% examples per conflict type (L1-L4)
- Mix of natural-occurring (66%) and augmented (34%) conflicts
- Stratified by domain (science, history, current events, etc.)

### 5.3 Augmentation Protocol

For underrepresented conflict types, augmentation involves:
1. Identifying queries with potential for specific conflict types
2. Retrieving additional documents from diverse sources
3. Injecting controlled conflicts (with clear ground truth)
4. Marking augmented examples in metadata

**Augmentation Rate by Type** (from paper):
- L1 (Temporal): 38% augmented (natural conflicts common)
- L2 (Numerical): 52% augmented (less common naturally)
- L3 (Entity): 29% augmented
- L4 (Semantic): 17% augmented (most naturally occurring)

---

## 6. Ethical Considerations

### 6.1 Annotator Treatment
- Fair compensation (above local minimum wage)
- Clear task expectations and time estimates
- Option to skip distressing content

### 6.2 Content Sensitivity
- Flag content involving violence, discrimination, or misinformation
- Do not annotate content requiring domain expertise (medical, legal) without specialist review

### 6.3 Data Privacy
- Anonymize all annotator identifiers
- Do not collect personal information from query content
- Comply with IRB requirements if applicable

---

## 7. Example Annotations

### Example 1: L1 (Temporal)

**Query**: "Who is the current Prime Minister of the UK?"

**Document 1** (older source):
> "Boris Johnson is the Prime Minister of the United Kingdom."

**Document 2** (more recent source):
> "Rishi Sunak serves as the Prime Minister of the UK."

**Annotation**:
- Conflict: YES
- Type: L1 (Temporal)
- Gold Answer: "Rishi Sunak" (based on recency)
- Reasoning: Clear temporal ordering; Document 2 is more recent.

---

### Example 2: L2 (Numerical)

**Query**: "What is the height of Mount Everest?"

**Document 1**:
> "Mount Everest stands at 8,848 meters above sea level."

**Document 2**:
> "The height of Mount Everest was remeasured at 8,848.86 meters in 2020."

**Annotation**:
- Conflict: YES
- Type: L2 (Numerical)
- Gold Answer: "8,848.86 meters" (more precise, newer measurement)
- Reasoning: Numerical conflict with precision difference.

---

### Example 3: L3 (Entity)

**Query**: "Who invented the telephone?"

**Document 1**:
> "Alexander Graham Bell invented the telephone in 1876."

**Document 2**:
> "Antonio Meucci developed a voice communication device before Bell."

**Annotation**:
- Conflict: YES
- Type: L3 (Entity)
- Gold Answer: "Alexander Graham Bell" (widely credited) or "Disputed"
- Reasoning: Entity attribution conflict requiring historical knowledge.

---

### Example 4: L4 (Semantic)

**Query**: "Is nuclear energy safe?"

**Document 1**:
> "Nuclear power is one of the safest forms of energy generation, with the lowest deaths per TWh."

**Document 2**:
> "Nuclear energy poses significant risks due to potential accidents and radioactive waste."

**Annotation**:
- Conflict: YES
- Type: L4 (Semantic)
- Gold Answer: "Depends on criteria used" or based on specific evidence
- Reasoning: Interpretive disagreement on qualitative assessment.

---

## 8. Revision History

| Version | Changes |
|---------|---------|
| 1.0 | Initial guidelines |
| 1.1 | Added edge case clarifications |
| 1.2 | Updated augmentation rates based on data |
| 2.0 | Final version for paper submission |
