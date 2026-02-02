---
description: Project Context and Implementation Roadmap for ManaMetrics
---

# 🔮 ManaMetrics Project Context

This workflow provides any AI agent working on this repository with the full context of the project, its architecture, and the implementation roadmap.

## 📖 Current Implementation Plan
The detailed implementation plan is located at [implementation_plan.md](file:///Users/childeric/Documents/projects/MTG/docs/implementation_plan.md).

## 🚀 Key Architectures
- **Stack**: PySpark, PyTorch (Lightning), Transformers, FastAPI, Streamlit, MLflow, DVC.
- **Goal**: Predict MTG card prices using a Multi-Modal Hybrid model (Tabular + NLP).
- **Phases**:
  1. **Data Engineering** [COMPLETED]: Scryfall API + PySpark ETL.
  2. **Baseline ML** [IN-PROGRESS]: Scikit-Learn (XGBoost) + SHAP.
  3. **Deep Learning Fusion**: BERT + MLP (PyTorch).
  4. **Serving & Vitrine**: FastAPI + Streamlit.

## 👥 Roles & Standards
- Detailed roles (Input/Output/Format) are defined in the plan.
- Use **Poetry** for all dependencies.
- Follow **Black/Isort/Flake8** standards.

## 📁 Repository Map
- `src/data`: ETL and API clients.
- `src/models`: Model architectures and training scripts.
- `src/serving`: API and UI code.
- `data/`: Parquet and JSON files (tracked by DVC).
- `models/`: Weights and bin files.
