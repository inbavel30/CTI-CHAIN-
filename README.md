# Transformer-Based Cyber Threat Intelligence Extraction and Attack-Chain Reconstruction

A transformer-based NLP pipeline that converts **unstructured Cyber Threat Intelligence (CTI) reports** into structured, evidence-grounded intelligence and reconstructs candidate attack chains.

## Objectives

- Extract entities, events, and relationships from CTI reports.
- Identify temporal relationships and ATT&CK techniques.
- Preserve evidence and provenance.
- Correlate intelligence across multiple reports.
- Reconstruct evidence-grounded attack chains.

## Core Pipeline

```text
CTI Reports
    ↓
Text Processing
    ↓
NER
    ↓
Entity Resolution
    ↓
Event Extraction
    ↓
Relation Extraction
    ↓
Temporal Reasoning
    ↓
MITRE ATT&CK Mapping
    ↓
Evidence & Provenance
    ↓
Cross-Report Correlation
    ↓
Knowledge Graph
    ↓
Attack-Chain Reconstruction
```

## Current Progress

- 50 CTI security reports collected
- 10,492 sentence-level records generated
- Dataset validation completed
- 2,500 CTI annotation candidates generated
- Annotation framework created
- AnnoCTR NER dataset prepared
- Four transformer NER models trained and evaluated

## NER Models

| Model | Test F1 |
|---|---:|
| BERT | 79.15% |
| RoBERTa | 79.53% |
| **CTI-BERT** | **82.00%** |
| SecureBERT | 80.05% |

**CTI-BERT** achieved the highest test F1 and was selected as the primary NER model.

## NER Dataset

Current benchmark entity types:

- LOC
- ORG
- SECTOR

Dataset size: **12,179 records and 268,419 tokens.**

## Technology

- Python
- Natural Language Processing
- Transformer Models
- BERT
- RoBERTa
- CTI-BERT
- SecureBERT
- MITRE ATT&CK
- Knowledge Graphs
- Temporal Reasoning
- Datalog Reasoning

## Project Structure

```text
CTI_ATTACK_CHAIN_PROJECT/
├── data/
│   ├── annotations/
│   ├── processed/
│   └── raw/
├── docs/
├── tools/
├── src/
│   ├── preprocessing/
│   └── ner/
├── outputs/
│   └── ner/
├── requirements-annotation.txt
├── requirements-ner.txt
└── README.md
```

## Next Steps

1. Complete CTI-Chain gold annotation
2. Event extraction
3. Relation extraction
4. Temporal reasoning
5. ATT&CK mapping
6. Entity resolution
7. Cross-report correlation
8. Knowledge graph construction
9. Attack-chain reconstruction
10. End-to-end evaluation and ablation study

## Current Status

**Completed:** CTI collection, preprocessing, annotation framework, NER dataset preparation, four-model benchmark, and CTI-BERT selection.

**In Progress:** CTI-Chain gold-data creation and downstream extraction and reasoning pipeline.

## Expected Outcome

An **evidence-grounded, temporally ordered attack-chain knowledge graph** connecting entities, events, relationships, ATT&CK techniques, campaigns, and supporting CTI evidence across reports.

## Research Areas

**Cyber Threat Intelligence · NLP · Transformer Models · MITRE ATT&CK · Knowledge Graphs · Temporal Reasoning · Attack-Chain Reconstruction**
