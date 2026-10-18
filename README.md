# Malaria Cell Classifier — UCS321 Project #4

Biocon Ltd. automated diagnosis of blood samples using CNN + Grad-CAM.

## Folder Structure

```
malaria_project/
├── data/
│   └── cell_images/          ← put downloaded dataset here
│       ├── Parasitized/
│       └── Uninfected/
├── models/                   ← saved .h5 model goes here (auto-created)
├── results/                  ← all plots saved here (auto-created)
├── src/
│   ├── __init__.py
│   ├── config.py             ← all settings in one place
│   ├── data.py               ← loading, preprocessing, augmentation
│   ├── model.py              ← CNN architectures
│   ├── train.py              ← training loop + callbacks
│   ├── evaluate.py           ← metrics, confusion matrix, ROC
│   ├── gradcam.py            ← Grad-CAM explainability
│   ├── pipeline.py           ← full pipeline orchestrator
│   └── app.py                ← Flask prediction API
├── train.py                  ← RUN THIS to train
├── run_app.py                ← RUN THIS to start the Flask API
├── streamlit_app.py          ← Streamlit web UI (calls the Flask API)
├── generate_results.py       ← produces evaluation plots
├── requirements.txt
├── .gitignore
└── README.md
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset from Kaggle and place in data/cell_images/

# 3. Train the model
python train.py

# 4. Start the prediction API (Flask, port 8080)
python run_app.py

# 5. In a second terminal, start the web UI
streamlit run streamlit_app.py
```

## API Usage

```bash
curl -X POST http://localhost:8080/predict \
  -F "file=@path/to/cell.png"
```

Expected output:
```json
{
  "label": "Infected",
  "confidence": 0.9732,
  "probability_infected": 0.9732,
  "probability_healthy": 0.0268,
  "risk_level": "High"
}
```

## Target Results

These are typical figures for CNNs on this dataset, not results measured in this repo. Run `python generate_results.py` after training to produce your own metrics.

| Metric    | Target    |
|-----------|-----------|
| Accuracy  | ~95–97%   |
| AUC-ROC   | ~98–99%   |
| F1 Score  | ~95–97%   |

## Dataset
NIH Malaria Cell Images — ~27,558 images  
https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria
