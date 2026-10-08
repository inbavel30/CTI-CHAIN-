Transformer-Based Cyber Threat Intelligence Extraction and Attack-Chain Reconstruction from Unstructured Security Reports
1. Project Overview
This project develops a transformer-based Cyber Threat Intelligence (CTI) processing pipeline for converting unstructured security reports into structured, evidence-grounded intelligence and reconstructing attack chains across reports.
The research addresses a practical CTI analysis problem: important intelligence is distributed across narrative reports, making it difficult to consistently identify entities, events, relationships, temporal order, ATT&CK techniques, and connections between reports.
The intended system transforms CTI reports into structured information while preserving evidence and provenance, and ultimately produces candidate attack chains that can be inspected and validated.
Research Question
Can transformer-based NLP automatically transform heterogeneous CTI reports into evidence-grounded entities, events, relationships, temporal constraints, and ATT&CK mappings, and use these structures to reconstruct attack chains across reports?
Core Pipeline
UNSTRUCTURED CTI REPORTS
          |
          v
TEXT PROCESSING AND DATASET CONSTRUCTION
          |
          +-----------------------------+
          |                             |
          v                             v
   NER BENCHMARKING              CTI-CHAIN ANNOTATION
          |                             |
          v                             v
   MODEL COMPARISON                GOLD DATASET
          |                             |
          v                             |
       CTI-BERT                        |
          +-------------+---------------+
                        |
                        v
                ENTITY RESOLUTION
                        |
                        v
                 EVENT EXTRACTION
                        |
                        v
                RELATION EXTRACTION
                        |
                        v
                TEMPORAL REASONING
                        |
                        v
                  ATT&CK MAPPING
                        |
                        v
              EVIDENCE + PROVENANCE
                        |
                        v
                 CAMPAIGN CONTEXT
                        |
                        v
             CROSS-REPORT CORRELATION
                        |
                        v
                 KNOWLEDGE GRAPH
                        |
                        v
             ATTACK UNIT CONSTRUCTION
                 PRE -> BEHAVIOR -> POST
                        |
                        v
                 DATALOG REASONING
                        |
                        v
             CANDIDATE ATTACK CHAINS
                        |
                        v
             CONFIDENCE / UNCERTAINTY
                        |
                        v
            EVIDENCE-GROUNDED OUTPUT
2. Work Completed So Far
Research and system design
- Research problem, objective, and research question defined.
- CTI-Chain architecture designed.
- Evidence-grounded and provenance-aware processing approach established.
- Manual annotation schemas and controlled vocabularies defined for downstream CTI-Chain tasks.
- Research evaluation strategy defined at component and attack-chain levels.
CTI data collection and preprocessing
- 50 CTI security reports collected.
- PDF-to-text extraction completed.
- 10,492 sentence-level records generated.
- Dataset validation completed successfully.
- 2,500 annotation candidates generated for manual CTI-Chain annotation, with 50 candidates per report.
Annotation infrastructure
The annotation framework has been created for:
- Events
- Relations
- Temporal relations
- Campaign context
- Evidence and provenance
- Attack-chain structures
Annotation status values include GOLD, REVIEW, and REJECTED, with certainty and evidence fields maintained separately from model confidence.
NER benchmark
The AnnoCTR NER dataset was prepared and converted into train, validation, and test JSONL files.
Split	Records	Tokens	Entity Tokens
Training	7,570	148,940	2,846
Validation	1,550	54,657	1,028
Test	3,059	64,822	969
Total	12,179	268,419	4,843


The current AnnoCTR benchmark contains the entity categories:
LOC
ORG
SECTOR
with the BIO label space:
O
B-LOC
I-LOC
B-ORG
I-ORG
B-SECTOR
I-SECTOR
3. Transformer Models Trained
Four transformer NER models were trained and evaluated. No DeBERTa or SciBERT model is included in the final benchmark.
Model	Pretrained Model
BERT	google-bert/bert-base-uncased
RoBERTa	FacebookAI/roberta-base
CTI-BERT	ibm-research/CTI-BERT
SecureBERT	ehsanaghaei/SecureBERT


Common NER Configuration
Maximum sequence length : 256
Training batch size     : 4
Evaluation batch size   : 4
Gradient accumulation   : 4
Effective batch size    : 16
Learning rate           : 5e-5
Weight decay            : 0.01
Random seed             : 42
GPU                     : NVIDIA GeForce RTX 3050 Laptop GPU (4 GB)
BERT and RoBERTa were run for 1 epoch. CTI-BERT and SecureBERT were trained for 3 epochs, with the best epoch selected using validation loss.
4. NER Results
Test Performance
Metric	BERT	RoBERTa	CTI-BERT	SecureBERT
Test Accuracy	99.4442%	99.4850%	99.5009%	99.4476%
Test Precision	79.0896%	77.2849%	80.9722%	75.9591%
Test Recall	79.2023%	81.9088%	83.0484%	84.6154%
Test F1	79.1459%	79.5297%	81.9972%	80.0539%
Test Loss	0.018295	0.017180	0.020863	0.018907


CTI-BERT achieved the highest recorded test F1 score, 81.9972%, and is therefore selected as the primary NER model for the next stage of the pipeline.
Validation Performance
Metric	BERT	RoBERTa	CTI-BERT	SecureBERT
Validation Accuracy	98.3998%	98.4238%	98.5007%	98.5462%
Validation Precision	74.9035%	77.3764%	73.9292%	73.0903%
Validation Recall	62.2793%	65.3291%	63.7239%	67.5762%
Validation F1	68.0105%	70.8442%	68.4483%	70.2252%


Training and Runtime Summary
Parameter	BERT	RoBERTa	CTI-BERT	SecureBERT
Epochs Trained	1	1	3	3
Best Epoch	1	1	2	2
Best Validation Loss	0.086594	0.093233	0.089887	0.077813
Parameters	108,897,031	124,060,423	123,856,135	124,060,423
Training Time	98.62 s	102.12 s	318.96 s	531.93 s
Training Samples/sec	76.763	74.126	71.200	42.704


Experiment Output Directories
outputs/ner/
├── bert_base_uncased/
├── roberta_base/
├── cti_bert/
└── securebert/
The experiment directories contain saved models and/or experiment records such as:
metrics.json
results.csv
epoch_history.csv
trainer_log_history.csv
test_metrics.json
The exact files vary by experiment.
5. CTI-BERT Selection
The current NER benchmark produced the following test F1 comparison:
BERT        : 79.1459%
RoBERTa     : 79.5297%
CTI-BERT    : 81.9972%   <- Selected
SecureBERT  : 80.0539%
CTI-BERT is the selected primary NER model because it achieved the highest recorded test F1 among the four benchmarked models.
The recorded CTI-BERT best validation-loss epoch is Epoch 2 with validation loss 0.089887.
6. Important NER Scope Note
The current AnnoCTR benchmark evaluates only LOC, ORG, and SECTOR entities.
The broader CTI-specific entity design includes categories such as:
GROUP
MALWARE
TOOL
SOFTWARE
TECHNIQUE
VULNERABILITY
IP
DOMAIN
FILE
ORGANIZATION
LOCATION
DATE
These richer CTI-specific categories are part of the planned CTI-Chain annotation design, but they are not covered by the current AnnoCTR benchmark results. A separate CTI-specific annotation and training stage is required to evaluate those categories.
7. Current Annotation Framework
The project has structured storage for the CTI-Chain gold data:
data/annotations/
├── attack/
├── events/
├── evidence/
├── ner/
├── relations/
└── temporal/
Event Vocabulary
EXECUTE
DOWNLOAD
DELIVER
EXPLOIT
PERSIST
DISCOVER
COLLECT
EXFILTRATE
COMMUNICATE
CREATE
MODIFY
DELETE
CREDENTIAL_ACCESS
LATERAL_MOVE
IMPACT
Relation Vocabulary
USES
EXECUTES
DOWNLOADS
DELIVERS
EXPLOITS
CREATES
MODIFIES
CONNECTS_TO
COMMUNICATES_WITH
TARGETS
DROPS
LOCATED_AT
USES_TECHNIQUE
NONE
Temporal Vocabulary
BEFORE
AFTER
OVERLAP
CONCURRENT
UNKNOWN
Every annotation is intended to maintain evidence/provenance back to the source report, page, sentence, and relevant text span where applicable.
8. What We Need to Do Next
The immediate next phase is CTI-Chain gold-data creation, followed by implementation and evaluation of the downstream extraction and reasoning pipeline.
Phase 1: Manual CTI-Chain Annotation
Annotate the 50 reports for:
1. Events
2. Relations
3. Temporal relations
4. Campaign context
5. Evidence and provenance
The recommended first milestone is to complete a pilot annotation set for R001 to R010, validate the annotation workflow, and then continue through the remaining reports.
Phase 2: Gold Dataset Validation
Run structural and semantic validation to verify:
- valid report and sentence references;
- valid spans;
- valid relation endpoints;
- valid temporal references;
- valid controlled vocabulary values;
- complete evidence links;
- unique fact identifiers;
- consistent GOLD/REVIEW/REJECTED status.
The existing validation framework is located at:
tools/annotation_validator.py
Phase 3: Downstream Information Extraction
Implement and evaluate:
CTI-BERT NER
      |
      v
ENTITY EXTRACTION
      |
      v
ENTITY RESOLUTION
      |
      v
EVENT EXTRACTION
      |
      v
RELATION EXTRACTION
      |
      v
TEMPORAL REASONING
      |
      v
ATT&CK MAPPING
Phase 4: Evidence and Campaign Modeling
Attach evidence and provenance to generated facts and model campaign context across reports.
Phase 5: Cross-Report Correlation
Correlate entities, events, relationships, ATT&CK techniques, campaigns, and temporal constraints across multiple CTI reports.
Phase 6: Knowledge Graph Construction
Construct a graph containing entities, events, relationships, ATT&CK techniques, campaigns, evidence, and temporal constraints.
Conceptually:
THREAT ACTOR
     |
    USES
     v
  MALWARE
     |
   EXECUTES
     v
  PROCESS
     |
COMMUNICATES_WITH
     v
    C2
     |
 EXFILTRATES
     v
    DATA
Phase 7: Attack Unit Construction
Represent structured behavior as:
PRECONDITION
      |
      v
   BEHAVIOR
      |
      v
POSTCONDITION
Phase 8: Datalog Reasoning and Attack-Chain Reconstruction
Use rule-based reasoning over the structured attack units to identify candidate attack paths while respecting relational and temporal constraints.
The final output should be an evidence-grounded candidate attack chain rather than an unsupported sequence of techniques.
Phase 9: Evaluation and Ablation Study
Evaluate the system at multiple levels:
- NER: precision, recall, F1
- Event extraction: precision, recall, F1
- Relation extraction: precision, recall, F1
- Temporal reasoning: relation-level accuracy/F1
- ATT&CK mapping: mapping accuracy/F1
- Entity resolution: resolution precision/recall/F1
- Attack-chain reconstruction: chain-level correctness, evidence support, and temporal consistency
The ablation study should measure the effect of major components such as:
Temporal Reasoning
ATT&CK Mapping
Cross-Report Correlation
Evidence Constraints
Datalog Reasoning
9. Project Structure
CTI_ATTACK_CHAIN_PROJECT/
|
├── data/
│   ├── annotations/
│   │   ├── attack/
│   │   ├── events/
│   │   ├── evidence/
│   │   ├── ner/
│   │   ├── relations/
│   │   └── temporal/
│   │
│   ├── processed/
│   │   ├── sentences.csv
│   │   └── ner/
│   │       ├── train.jsonl
│   │       ├── validation.jsonl
│   │       ├── test.jsonl
│   │       └── label_map.json
│   │
│   └── raw/
│       ├── ANNOCTR/
│       └── reports/
│           ├── reports.csv
│           └── files/
│
├── docs/
│   └── ANNOTATION_GUIDELINES.md
│
├── tools/
│   ├── annotation_validator.py
│   ├── init_annotations.py
│   ├── create_annotation_candidates.py
│   └── create_ner_pilot.py
│
├── src/
│   ├── preprocessing/
│   │   ├── pdf_to_csv.py
│   │   └── validate_sentences.py
│   │
│   └── ner/
│       ├── prepare_annoctr_ner.py
│       ├── train_bert_ner.py
│       ├── train_roberta_ner.py
│       ├── train_deberta_ner.py
│       └── train_cti_bert_ner.py
│
├── outputs/
│   └── ner/
│       ├── bert_base_uncased/
│       ├── roberta_base/
│       ├── cti_bert/
│       └── securebert/
│
├── requirements-annotation.txt
├── requirements-ner.txt
└── README.md
10. Reproducibility
The NER experiments use fixed model configurations and random seed 42. Experiment outputs record dataset sizes, model names, parameter counts, validation/test metrics, training history, runtime information, and other relevant training details.
Core NER settings:
Seed                    : 42
Max sequence length     : 256
Train batch size        : 4
Evaluation batch size   : 4
Gradient accumulation   : 4
Effective batch size    : 16
Learning rate           : 5e-5
Weight decay            : 0.01
11. Current Project Status
Component	Status
Research problem and objectives	Completed
CTI-Chain architecture	Completed
CTI report collection	Completed
50 CTI reports	Completed
PDF-to-text extraction	Completed
10,492-sentence dataset	Completed
Dataset validation	Completed
Annotation schemas	Completed
Annotation candidate sampling	Completed
AnnoCTR preparation	Completed
BERT NER	Completed
RoBERTa NER	Completed
CTI-BERT NER	Completed
SecureBERT NER	Completed
NER model comparison	Completed
CTI-BERT selection	Completed
Manual CTI-Chain gold annotation	Pending
Event extraction	Pending
Relation extraction	Pending
Temporal reasoning	Pending
ATT&CK mapping	Pending
Entity resolution	Pending
Campaign context modeling	Pending
Cross-report correlation	Pending
Knowledge graph	Pending
Attack unit construction	Pending
Datalog reasoning	Pending
Attack-chain reconstruction	Pending
End-to-end evaluation	Pending
Ablation study	Pending


12. Research Outcome Target
The final research system is intended to produce an:
Evidence-grounded, temporally ordered attack-chain knowledge graph connecting entities, events, relationships, ATT&CK techniques, campaigns, and supporting CTI evidence across reports.

The current project milestone is the completion of the transformer NER benchmark and selection of CTI-BERT. The immediate next milestone is the creation and validation of the CTI-Chain gold dataset, followed by the downstream extraction, reasoning, knowledge-graph, and attack-chain reconstruction components.
Project Information
Title: Transformer-Based Cyber Threat Intelligence Extraction and Attack-Chain Reconstruction from Unstructured Security Reports
Primary Areas: Cyber Threat Intelligence, Natural Language Processing, Transformer Models, MITRE ATT&CK, Knowledge Graphs, Temporal Reasoning, Attack-Chain Reconstruction
