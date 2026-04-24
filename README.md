# Continuous Identity Threat Detection and Response (ITDR) Prototype

A containerized system for detecting identity-based attacks using machine learning and rule-based detection.

## 🎯 Features

- **Simulated Identity Telemetry**: Generates realistic Entra ID/Okta-style logs
- **Attack Simulation**: Password Spray, Impossible Travel, Token Theft, Privilege Escalation
- **Hybrid Detection**: Rule-based + ML (Isolation Forest, Autoencoder)
- **Risk Scoring**: Aggregated 0-100 risk scores with explainability
- **Interactive Dashboard**: Streamlit-based visualization with metrics

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip

### Installation
```bash
pip install -r requirements.txt
```

### Generate Data
```bash
python generate_data.py
```

### Run Dashboard
```bash
streamlit run ui/app.py
```

## 🐳 Docker

```bash
# Build and run
docker-compose up --build

# Access dashboard at http://localhost:8501
```

## 📁 Project Structure

```
itdr_prototype/
├── simulation/          # Log generation & attack simulation
│   ├── schema.py       # Data models (AuthEvent, Identity, etc.)
│   ├── generator.py    # Normal traffic generator
│   └── attack_scenarios.py  # Attack pattern injection
├── detection/           # Detection engine
│   ├── etl.py          # Log loading & preprocessing
│   ├── features.py     # Feature extraction
│   ├── models.py       # ML models (IsolationForest, Autoencoder)
│   ├── rules.py        # Rule-based detection
│   └── scorer.py       # Risk aggregation
├── ui/                  # Dashboard
│   └── app.py          # Streamlit application
├── tests/               # Unit tests
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 📊 Attack Types

| Attack | Detection Method |
|--------|------------------|
| Password Spray | Rule: >5 failed logins from same IP |
| Impossible Travel | Rule: Location change faster than travel |
| Token Theft | ML: Anomalous session behavior |
| Privilege Escalation | Rule: Role change + unusual activity |

## 🧪 Running Tests

```bash
pytest tests/ -v
```

## 📈 Evaluation Metrics

The dashboard displays real-time metrics:
- Precision / Recall / F1 Score
- Confusion Matrix
- Detection rate by attack type

## License

MIT License - Capstone Project
