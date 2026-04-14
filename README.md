# Healthcare AI Agent

A lightweight healthcare decision-support application that combines:
- symptom-based retrieval from a medical knowledge base,
- doctor recommendation by specialty and city,
- insurance cost estimation,
- patient history memory,
- and an optional AI explanation layer.

The UI is built with Gradio for quick local testing and demos.

## Main Features

- Symptom analysis using sentence embeddings + FAISS nearest-neighbor search.
- Structured health insight output with urgency, suggested tests, and advice.
- Doctor matching with map visualization.
- Insurance cost prediction using a trained Random Forest model.
- Persistent patient visit history in local JSON storage.
- Optional local FLAN-T5 based explanation rewrite with safe fallback.

## Project Structure

- `core/`: diagnostic logic, insurance estimation, doctor lookup, memory, explainability.
- `data/`: source CSV files for medical knowledge, doctors, and insurance training data.
- `models/`: FAISS index and embedding artifacts (and insurance model after training).
- `scripts/`: utilities to build the symptom index and train insurance model.
- `storage/`: local persisted patient memory.
- `ui/`: Gradio interface and app entrypoint.

## Quick Start

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure model artifacts exist:

```bash
python -m scripts.build_knowledge_index
python -m scripts.train_insurance_model
```

4. Run the app:

```bash
python -m ui.app
```

## Notes

- This project is for educational decision-support use only.
- It does **not** provide a medical diagnosis.
- Always consult a licensed healthcare professional.
