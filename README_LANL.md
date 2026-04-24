# LANL Dataset Evaluation Instructions

You validate the ITDR model's generalization capabilities using the real-world LANL dataset.

## 1. Setup Data

Download the following files from [https://csr.lanl.gov/data/cyber1/](https://csr.lanl.gov/data/cyber1/) (requires registration):
1. `auth.txt.gz` (7.2GB)
2. `redteam.txt.gz` (4KB)

Place them in this folder:
`data/lanl/`

Folder structure should look like:
```
itdr_prototype/
├── data/
│   └── lanl/
│       ├── auth.txt.gz
│       └── redteam.txt.gz
├── detection/
├── saved_models/
│   └── rba_trained_model.pkl  <-- Generated automatically
└── evaluate_lanl.py
```

## 2. Prepare Model

Run this command to train the model on the RBA dataset and save it to disk:
```powershell
python train_rba_model.py
```
*(This may take 5-10 minutes on the full dataset)*

## 3. Run Evaluation

Run the evaluation script to test the saved model on the LANL data:
```powershell
python evaluate_lanl.py
```

### What happens?
1. The script loads the RBA-trained model weights.
2. It streams the first 50,000 LANL events to "warm up" (learn normal behavior baselines for LANL users).
3. It then predicts attacks on subsequent events using the RBA model's decision logic.
4. It compares predictions against the `redteam.txt` ground truth labels.
5. Finally, it prints Precision, Recall, and F1 scores.

## Troubleshooting
- **Memory Error**: The script streams data chunk-by-chunk to avoid high RAM usage. If you still hit issues, reduce `CHUNK_SIZE` in `evaluate_lanl.py`.
- **File Not Found**: Ensure the folder is named `data\lanl` and files are `.txt.gz` (do not unzip them).
