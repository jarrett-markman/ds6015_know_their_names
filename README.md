# Verité: Know Their Names 

## DS6015: UVA Data Science Capstone

### Marissa Burton, Grace George, Jarrett Markman, Aniyah McWilliams

This pipeline links individuals across the 1870 and 1880 Albemarle County census
records using probabilistic record linkage (Fellegi-Sunter/Expectation-Maximization (**EM**), via the Splink package in ***python***), then boosts match confidence using known family relationships (*relative corroboration*), and
evaluates the model results against a hand-labeled ground truth set.

#### Directory structure

```
├── README.md
├── data/
│   ├── mentions.csv
│   ├── assertions.csv
│   ├── ground_truth.csv
│   ├── predictions.csv (hidden in the .gitignore as it is over 5GB)
│   ├── matches.csv (hidden in the .gitignore as it is over 1GB)
│   └── boosted_preds.parquet
├── eda/ (work in progress)
├── models/
│   ├── em_model.py
│   ├── relative_corroboration.py
│   ├── results.py
│   └── requirements.txt
├── visuals/ (work in progress)
└── notebooks/
    ├── splink_model.ipynb
    ├── relative_corroboration.ipynb
    └── results.ipynb
```

#### Setup and running this pipeline

> **Note:** This assumes that the commands below are ran from from a terminal and that the terminal's current directory is the repo root (main branch/folder).

> **Note:** This pipeline requires a lot of computational power, and in order to actually run our model we utilized UVA's HPC Rivanna. 

```bash
python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -r models/requirements.txt
```

Run the following in order, from the repo root:

```bash
python models/em_model.py
python models/relative_corroboration.py
python models/results.py
```

##### 1. em_model.py
Reads `data/mentions.csv`. Trains the EM (Fellegi-Sunter) record linkage
model and generates match probabilities.
Output: `data/predictions.csv`

##### 2. relative_corroboration.py
Reads `data/predictions.csv` and `data/assertions.csv`. Uses known
family-relationship data to compute a secondary corroboration score for
each candidate match.
Output: `data/matches.csv`, `data/boosted_preds.parquet`

##### 3. results.py
Reads `data/boosted_preds.parquet` and `data/ground_truth.csv`. Evaluates
matches against ground truth and generates plots (probability
distributions, confidence comparisons, calibration).