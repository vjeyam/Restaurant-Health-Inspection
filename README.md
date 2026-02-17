# Restaurant-Health-Inspection

## Overview

This project tries to solve if we can proactively identify restaurants that are most likely to receive critical health violations, so inspections can be prioritized more effectively?

![Restaurant Health Inspection](assets/chicago-risk-map.png)

Below we can see the performance of the logistic regression and random forest models in terms of ROC-AUC:

![ROC Curve](assets/roc-curve.png)

| Model         | Test ROC Curve |
|---------------|----------------|
| Logistic Reg  | ~0.56          |
| Random Forest | ~0.63          |

## Setup

If you don't want to run the code and see the results, in the root directory you can run:

```bash
open top100_chicago_risk_map.html
```

To run the code yourself, you can follow these steps:

1. Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/vjeyam/Restaurant-Health-Inspection.git
```

2. Create a virtual environment and activate it:

```bash
conda create -n restaurant_inspection python=3.9
conda activate restaurant_inspection
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Run `main.py` to execute the notebook and generate the results:

```bash
# To run the full pipeline including data ingestion and model training
python src/main.py

# If you want to skip data ingestion and use existing data/models, you can run:
python src/main.py --skip-ingestion
```

## Results

The model found:

- The baseline critical violation rate is 39%, meaning that 39% of all inspections result in a critical violation.

- Restaurants in the highest predicted risk decile show substantially elevated violation rates

- Random Forest improves ROC-AUC from ~0.56 (logistic) to ~0.63

- Both models were better than random guessing (50%)
