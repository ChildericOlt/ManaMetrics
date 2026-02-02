# ManaMetrics - Technical Blueprint

This document outlines the technical strategy for **ManaMetrics**, a hybrid asset valuator for Magic: The Gathering. As CTO, my goal is to ensure this project serves as a high-end showcase of modern ML engineering practices while remaining accessible to stakeholders.

## 🎓 Technical Context for the Team

### 1. BERT: The "Smart Reader"
**BERT** (Bidirectional Encoder Representations from Transformers) is a model that "reads" text. Unlike simple keyword search, it understands:
- **Context**: It knows "Strike" in "Lightning Strike" is a damage spell, not a labor union action.
- **Semantics**: It captures the *power* of a sentence like "Destroy all creatures" vs "Tap target creature".
- **Fine-tuning**: We take a pre-trained BERT and "teach" it the specific vocabulary of Magic (Mana, Scry, Trample).

### 2. XAI (SHAP & Captum): The "Magnifying Glass"
Artificial Intelligence is often a "black box". **XAI** (Explainable AI) tools help us look inside:
- **SHAP**: Used for tabular data. It will tell us exactly how much "Rarity=Mythic" increases the price compared to "Set=Alpha".
- **Captum**: Used for Deep Learning. It will highlight specific words in the card's text that drove the price prediction (e.g., highlighting "Draw three cards").

### 3. Understanding Metrics (Regression vs Classification)
Since we are predicting a **Price** (a continuous number), we use **Regression** metrics:
- **RMSE (Root Mean Square Error)**: The average distance between the real price and our guess. Good for seeing if we are roughly in the ballpark.
- **MAPE (Mean Absolute Percentage Error)**: Much more intuitive. "Our prediction is wrong by 15% on average."
- **Why no ROC/Accuracy?**: These are for "Yes/No" or categories. However, we can use **Top-k Accuracy** to see if our predicted "Top 10 most expensive cards" match the real Top 10.

---

## 👥 Project Governance: Roles & Timeline

### Virtual Team Composition

| Role | Context & Mission | Inputs | Outputs & Format | Key Tasks |
| :--- | :--- | :--- | :--- | :--- |
| **Data Engineer** | Build the "Gold Layer" (clean data) foundation. | Scryfall API (JSON/REST) | `processed_data.parquet`, `dataset_schema.md` | API client with rate limiting; PySpark ETL pipeline; **Dataset Profiling / Documentation** (Schema, distribution, missing values). |
| **ML Researcher** | Hybrid Modeling & Performance. | `processed_data.parquet` | `model_weights.pt`, `metrics.json` | Fine-tune DistilBERT on card text; Train Scikit-Learn baseline; Implement hybrid PyTorch architecture (Fusion). |
| **Backend/X-AI** | Serving & Visual Interaction (The Showroom). | `model_weights.pt`, Card JSON | FastAPI Endpoint, Streamlit Dashboard | Create `/predict` API; Implement Captum/SHAP visual components; Build the Streamlit interface for live demos. |
| **DevOps** | Industrialization & Quality Control. | Source Code, Dockerfile | Docker Image, CI/CD Pipeline | Set up MLflow tracking; Build optimized Docker containers; Configure GitHub Actions for linting/testing. |

### Order of Operation (Roadmap)
1. **Week 1: Infrastructure & Data** (Data Eng + DevOps)
   - Initialize Repo, GitHub Actions, and Scryfall ingestion.
   - **Deliverable**: `processed_data.parquet` + Dataset Documentation.
2. **Week 2: Baseline & Research** (ML Researcher)
   - Scikit-Learn model + SHAP analysis.
   - **Deliverable**: Baseline Metrics + Interpretability Report.
3. **Week 3: Deep Learning Fusion** (ML Researcher + Backend)
   - Hybrid model training (BERT + MLP).
   - **Deliverable**: High-performance model weights.
4. **Week 4: Serving & Vitrine** (Backend + DevOps)
   - FastAPI integration, Streamlit UI, and Docker deployment.
   - **Deliverable**: Fully functional Showcase App with XAI.

---

## Technical Recommendations & Proposed Stack

### 1. Development & Quality Standards
- **Dependency Management**: [Poetry](https://python-poetry.org/) for deterministic builds.
- **Code Quality**: `black`, `isort`, `flake8` enforced via `pre-commit` hooks.
- **CI/CD**: GitHub Actions to run tests and linters on every push.
- **Deployment**: The final code will be hosted on a public GitHub repository with a detailed README and "Vitrine" documentation.

### 2. Multi-Modal Modeling Strategy
- **NLP Component**: Fine-tuning **DistilBERT** (lighter, faster inference) or **RoBERTa** on the Magic ruleset.
- **Tabular Component**: Multi-Layer Perceptron (MLP) for numeric/categorical features.
- **Fusion Layer**: Concatenation of NLP embeddings and MLP output.
- **Custom Loss**: **MSLE** (Mean Squared Logarithmic Error) to handle the price variance ($0.01 to $50,000+).

### 3. "Showroom" Features (The Vitrine)
- **Interactive UI**: A **Streamlit** dashboard for real-time "Price Breakdown" with feature attribution.
- **Data Versioning**: **DVC** (Data Version Control) linked to GitHub to show professional data lineage.

---

## Proposed Changes

### [Component] Repository Initialization & DevOps
#### [MODIFIED] [GitHub Deployment] Host code on GitHub with automated testing.
#### [NEW] [pyproject.toml](file:///Users/childeric/Documents/projects/MTG/pyproject.toml)
#### [NEW] [README.md](file:///Users/childeric/Documents/projects/MTG/README.md)

### [Component] Data Engineering (Phase 1)
#### [NEW] [etl.py](file:///Users/childeric/Documents/projects/MTG/src/data/etl.py)
#### [NEW] [scryfall_client.py](file:///Users/childeric/Documents/projects/MTG/src/data/scryfall_client.py)

### [Component] Modeling (Phase 2 & 3)
#### [NEW] [train_hybrid_model.py](file:///Users/childeric/Documents/projects/MTG/src/models/train_hybrid_model.py)
#### [NEW] [baseline_ml.ipynb](file:///Users/childeric/Documents/projects/MTG/notebooks/01_baseline_ml.ipynb)

---

## Verification Plan

### Automated Tests
- Run `pytest` for ETL unit tests and Scryfall API mocking.
- Functional tests to ensure the `/predict` endpoint returns valid JSON.

### Manual Verification
- Deploy the Streamlit app locally and verify the inference speed is < 100ms.
- Inspect MLflow UI to ensure all hyperparameters and metrics are logged.
