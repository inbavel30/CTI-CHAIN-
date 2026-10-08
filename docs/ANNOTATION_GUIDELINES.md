# CTI-Chain AI Annotation Guidelines

## Ground-truth rule
Annotate only what the CTI source supports. Do not invent entities, events, relations,
temporal order, ATT&CK mappings, preconditions, or postconditions.

Every annotation keeps `report_id` and `sentence_id`. Evidence keeps page and source span
when available.

## NER
GROUP, MALWARE, TOOL, SOFTWARE, TECHNIQUE, VULNERABILITY, IP, DOMAIN, FILE,
ORGANIZATION, LOCATION, DATE.

## Events
EXECUTE, DOWNLOAD, DELIVER, EXPLOIT, PERSIST, DISCOVER, COLLECT, EXFILTRATE,
COMMUNICATE, CREATE, MODIFY, DELETE, CREDENTIAL_ACCESS, LATERAL_MOVE, IMPACT.

## Relations
USES, EXECUTES, DOWNLOADS, DELIVERS, EXPLOITS, CREATES, MODIFIES, CONNECTS_TO,
COMMUNICATES_WITH, TARGETS, DROPS, LOCATED_AT, USES_TECHNIQUE, NONE.

## Temporal
BEFORE, AFTER, OVERLAP, CONCURRENT, UNKNOWN.
Do not infer BEFORE merely from sentence order.

## ATT&CK
Map an event to an ATT&CK technique only when the source supports the mapping.
Use the canonical technique ID.

## Evidence certainty
EXPLICIT = directly stated.
INFERRED = derived from source facts.
UNCERTAIN = ambiguous source.

## Status
GOLD = reviewed ground truth.
REVIEW = needs adjudication.
REJECTED = intentionally excluded.

Only GOLD records should be used for final test-set evaluation.
