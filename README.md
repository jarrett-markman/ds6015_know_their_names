# Verité: Know Their Names 

## DS6015: UVA Data Science Capstone

### Marissa Burton, Grace George, Jarrett Markman, Aniyah McWilliams

#### Abstract

The Verité/Know Their Names project is a partnership with the Afro-American Historical Association of Fauquier County​ (AAHA) with the goal of connecting descendants of formerly enslaved laborers with narrative information about their Virginia-based ancestors. It has long  been deemed a nearly impossible task requiring an immense amount of time, energy, and historical context. The introduction of complex entity resolution/record linkage modeling can help fast-track this historical phenomena. Moreover, we test the theory that embedding relational assertions between individuals and available ground truth matches within models can increase confidence for possible matches. To ensure the utmost accuracy of our methodology, our capstone project focuses primarily on census data for Albemarle County, Virginia from 1870 and 1880. We hope to break ground on the Verité mission by using the machine learning (ML) techniques of candidate blocking and comparisons, probabilistic record linkage modeling, and relative corroboration (confidence metric based on Expectation-Maximization (EM) probabilities) to identify likely candidates for matching between these two source documents. 

This pipeline links individuals across the 1870 and 1880 Albemarle County census
records using probabilistic record linkage (Fellegi-Sunter/Expectation-Maximization (**EM**), via the Splink package in ***python***), then boosts match confidence using known relationships (*relative corroboration*), and
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
├── eda/
|   ├── mentions_eda.ipynb 
|   ├── splink_record_match_draft.ipynb 
|   └── assertions_eda.ipynb
├── models/
│   ├── em_model.py
│   ├── relative_corroboration.py
│   ├── results.py
│   └── requirements.txt
└── notebooks/
    ├── exploring_accuracy_across_confidence.ipynb
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

#### Splink Citation
Linacre, R., Lindsay, S., Manassis, T., Slade, Z., Hepworth, T., Kennedy, R., & Bond, A. (2022). Splink: Free software for probabilistic record linkage at scale. International Journal of Population Data Science, 7(3). https://doi.org/10.23889/ijpds.v7i3.1794
