# CHF Mortality Risk Visualization Tool
Interactive visualization of clinical risk-factor combinations associated with mortality in congestive heart failure patients.

<img width="932" height="427" alt="image" src="https://github.com/user-attachments/assets/d9669187-116f-4934-b444-576b8f3a026e" />


## Project Overview

This project explores how combinations of clinical variables influence mortality risk in patients with congestive heart failure (CHF). Rather than examining individual predictors in isolation, the analysis investigates how interactions between laboratory values, demographic characteristics, and comorbidities affect mortality outcomes.

An interactive risk explorer was developed to allow users to visualize how mortality changes as additional risk factors are added.

## Repository Structure

- README.md
  
- CHF_exploratory_analysis.ipynb- interactive dashboard for exploring mortality risk across combinations of clinical variables
  
- CHF_visualization_analysis_and_risk_tool_prototype.ipynb-further progressive analysis of risk factors and initial prototype

- Final_CHF_visualization_risk_tool.ipynb-finalized visualization tool

- `images/` — visualizations and figures used throughout the project

- `README.md` — project overview, methodology, findings, and limitations

## Dataset

**Source:** UCI Heart Failure Clinical Records Dataset (299 patients)

**Outcome variable:**

- Death during follow-up

**Clinical features:**

- Age
- Ejection fraction
- Serum creatinine
- Serum sodium
- Platelets
- Anaemia
- Hypertension
- Diabetes
- Smoking status
- Sex

## Data Preprocessing and Feature Engineering
<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/128365f8-5f52-486b-905f-094931efdf67" />

The analysis included:

- Exploratory correlation analysis
- Outlier visualization
- Logistic regression with ROC-AUC scoring
- Transformation of continuous laboratory values into binary risk indicators
- Pairwise, three-way, and four-way interaction analysis

### Clinical thresholds

- Low ejection fraction: < 35%
- High creatinine: > 1.5 mg/dL
- Low sodium: < 135 mEq/L
- Age ≥ 70 years

## Methods
The analysis combined exploratory visualization with quantitative modeling techniques to identify clinically relevant predictors of mortality.

<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/00b1945e-82af-4ea9-8e4e-b770587ddc9c" />



Methods included:

- Pairwise scatterplot matrices to visualize relationships between continuous variables
- Logistic regression with ROC-AUC scoring to rank predictive variable combinations
- Progressive combination analysis to evaluate interaction effects
- Feature engineering using clinically meaningful thresholds
- Correlation analysis and exploratory data visualization

## Key Findings
<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/619451ba-de21-49ff-ab14-fdd654697985" />


- Mortality risk increased substantially as additional high-risk clinical features accumulated.
- Low ejection fraction, elevated creatinine, advanced age, and low sodium consistently appeared in the highest-risk groups.
- Variable interactions provided greater explanatory power than individual predictors alone.
- Progressive combination analysis revealed that some combinations were associated with mortality rates exceeding 80%, although some subgroups contained relatively few patients.
- The effects of low ejection fraction and elevated creatinine became more pronounced in older patients, suggesting important interaction effects.

<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/c3d4fb11-2cb6-4a40-9948-c3ef501a3e74" />




## Interaction Effects and Unexpected Findings

Several variables initially appeared to be strongly associated with mortality during exploratory analysis. Further investigation, however, revealed that some of these relationships were influenced by confounding variables, subgroup size, and interactions with stronger predictors.

<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/271c6bf8-4a31-4fb2-9065-3ccb0aa79cdc" />


Unexpected findings included:

Unexpected findings included:

- Anaemia initially appeared highly predictive, but age stratification suggested that much of the observed effect was attributable to older patients.

- Female patients initially appeared to experience higher mortality across several high-risk combinations; however, many of these subgroups contained only one to four patients, limiting the reliability of the observed differences.

- Smoking demonstrated little association with mortality despite its known cardiovascular risks, suggesting the presence of confounding factors and survivor bias.

- Platelet abnormalities exhibited visual patterns in pairwise analyses but contributed relatively little predictive value when evaluated alongside stronger predictors such as ejection fraction and serum creatinine.

- Variables that appeared predictive in isolation often behaved differently when examined alongside additional clinical risk factors.

<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/4a17d28e-8085-4613-b038-cf2b53495e8f" />
<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/e62e7331-af86-40dd-bce5-517710195f4d" />

Overall, mortality outcomes were better explained by interactions between multiple variables than by any single predictor alone. These findings emphasize the importance of investigating confounding effects and avoiding conclusions based solely on pairwise correlations.


## Limitations

- The dataset contained only 299 patients.
- Several high-risk combinations involved very small subgroups.
- Findings were exploratory and were not externally validated.
- Observed associations should not be interpreted as causal relationships.

## Interactive Risk Explorer
<img width="1132" height="918" alt="image" src="https://github.com/user-attachments/assets/0b0a035b-cd74-4584-b1fb-31e082b59452" />

The interactive dashboard allows users to explore how combinations of clinical variables influence mortality risk in congestive heart failure patients.

Features include:

- Selection of individual risk factors
- Exploration of previously identified high-risk combinations
- Dynamic comparison of mortality rates across patient subgroups
- Investigation of interaction effects between clinical variables
- Comparison of observed mortality against baseline risk

## Technologies and Techniques Used

### Programming and Analysis

- Python
- Jupyter Notebook / Google Colab
- Pandas
- NumPy

### Data Visualization

- Matplotlib
- Seaborn
- Correlation heatmaps
- Scatterplot matrices (pair plots)
- Bar charts
- Line plots
- Interactive dashboards

### Statistical Analysis and Modeling

- Logistic regression
- ROC-AUC analysis
- Pairwise, three-way, and higher-order interaction analysis
- Progressive combination analysis

### Data Processing and Feature Engineering

- Clinical threshold encoding
- Binary feature creation
- Categorical grouping
- Risk-factor aggregation
- Exploratory data analysis (EDA)

### Interactive Components

- ipywidgets
- Interactive risk explorer
- Dynamic filtering and comparison tools

- ## Future Directions

Future work could include:

- Validation using larger heart failure datasets
- Development of predictive machine-learning models
- Time-to-event survival analysis
- Integration of additional clinical variables
- Expansion of the interactive dashboard
