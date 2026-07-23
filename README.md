# Assignment 3 — Salary Prediction using Polynomial Regression

## 👤 Student Details
| Field | Detail |
|---|---|
| **Name** | AADISH ADLAK |
| **Registration Number** | 23BCE10681 |
| **Application Number** | IN26010985 |
| **Batch Number** | 9A |
| **Assignment Number** | Assignment-3 |
| **Email** | adlakaadish@gmail.com |
| **GitHub Repository** | https://github.com/AADISHADLAK/MPONLINE-Assignment-2 |

---

## 🎯 Objective
A company wants to estimate the salary of an employee based on their **position level**. Since the relationship between position level and salary is **non-linear** (salary rises slowly at junior levels and shoots up sharply at senior levels), this project builds a **Polynomial Regression model** to predict employee salaries more accurately than a simple straight-line (Linear Regression) model could.

---

## 📊 Dataset
**Position Salaries Dataset**
Source: [Kaggle — akram24/position-salaries](https://www.kaggle.com/datasets/akram24/position-salaries)

The dataset contains 10 records with the following columns:

| Column | Description |
|---|---|
| `Position` | Job title / designation (e.g., Manager, CEO) |
| `Level` | Numeric level of the position (1–10) — **Input Feature** |
| `Salary` | Annual salary in currency units — **Target Variable** |

> As instructed, the raw dataset file is **not required to be redistributed** here beyond what is needed to run the notebook locally. A copy is included under `data/Position_Salaries.csv` purely to make the project runnable end-to-end; the authoritative source and license remain on Kaggle at the link above. If you fork this repo, please verify the dataset license on Kaggle before further redistribution.

---

## 🛠️ Libraries Used
- `pandas` — data loading, exploration, and manipulation
- `numpy` — numerical operations
- `matplotlib` — data visualization (scatter plot + regression curve)
- `scikit-learn` — `train_test_split`, `PolynomialFeatures`, `LinearRegression`, evaluation metrics (`mean_absolute_error`, `mean_squared_error`, `r2_score`)

Install everything with:
```bash
pip install pandas numpy matplotlib scikit-learn
```

---

## 📁 Repository Structure
```
MPONLINE-Assignment-2/
│
├── Assignment-3.ipynb          # Full Jupyter Notebook (all 5 tasks, with outputs)
├── Assignment-3.py             # Standalone Python script version
├── data/
│   └── Position_Salaries.csv   # Dataset used for the project
├── images/
│   ├── original_data_scatter.png
│   └── polynomial_regression_curve.png
└── README.md                   # This file
```

---

## 🧭 Methodology

### Task 1 — Data Understanding
- Loaded the dataset using `pandas.read_csv()`.
- Displayed the first 5 records with `df.head()`.
- Identified **`Level`** as the input feature and **`Salary`** as the target variable (the `Position` column is a text label carrying the same information as `Level`, so it isn't used as a model input).
- Inspected structure and statistics using `df.info()` and `df.describe()`.

### Task 2 — Data Preprocessing
- Verified there are **no missing values** using `df.isnull().sum()`.
- Selected `X = df[['Level']]` as the feature matrix and `y = df['Salary']` as the target.
- Split the data into **80% training / 20% testing** using `train_test_split(test_size=0.2, random_state=42)`.

### Task 3 — Model Development
- Transformed the single input feature into **polynomial features of degree 3** (`1, x, x², x³`) using `PolynomialFeatures(degree=3)`.
- Trained a `LinearRegression` model on the transformed (polynomial) features — this combination is what constitutes **Polynomial Regression**.
- Generated salary predictions for the test set.

### Task 4 — Model Evaluation
- Computed **MAE**, **MSE**, and **R² Score** on the test set.
- Plotted:
  - A scatter plot of the raw data (`images/original_data_scatter.png`)
  - The fitted polynomial regression curve over a fine grid of position levels, overlaid on the actual data points (`images/polynomial_regression_curve.png`)

### Task 5 — Conclusion
- Summarized findings, compared Linear vs Polynomial Regression, and highlighted the advantage of Polynomial Regression for this dataset.

---

## 📈 Results

*(Exact values are reproduced when you run the notebook/script; a representative run is shown below.)*

| Metric | Value |
|---|---|
| Mean Absolute Error (MAE) | ≈ 70,635 |
| Mean Squared Error (MSE)  | ≈ 6.26 × 10⁹ |
| R² Score                  | ≈ 0.88 |

**Regression Curve:**

![Polynomial Regression Curve](images/polynomial_regression_curve.png)

**Observations:**
1. The degree-3 polynomial curve closely follows the sharp, non-linear rise in salary at higher position levels — something a straight line (simple Linear Regression) cannot capture.
2. Because the dataset only has 10 rows, the 80/20 split leaves a very small test set (2 rows), so the metrics above are illustrative rather than statistically robust — a larger real-world dataset would give more reliable evaluation.
3. The largest prediction errors occur near the highest position levels (e.g., CEO), where salary grows extremely fast and small changes in level cause very large changes in salary.

---

## ✅ Conclusion
The Polynomial Regression model (degree 3) captured the non-linear relationship between an employee's position level and their salary far better than a straight-line fit would. Salary increases slowly at lower levels but accelerates sharply at senior levels (e.g., Partner, C-level, CEO) — a pattern that Linear Regression, which assumes a constant rate of change, cannot represent well.

**Linear Regression vs Polynomial Regression:** Linear Regression fits a single straight line (`y = b0 + b1·x`) and assumes a constant relationship between input and output. Polynomial Regression extends this by adding higher-degree terms (`x², x³, ...`), letting the model fit curves and capture non-linear trends while still being solved with linear regression techniques on the transformed features.

**Advantage for this dataset:** Since salary rises exponentially with seniority rather than linearly, Polynomial Regression provides a much closer, more accurate fit to the actual salary curve, significantly improving prediction accuracy at higher position levels compared to a simple linear model.

---

## ▶️ How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/AADISHADLAK/MPONLINE-Assignment-2.git
   cd MPONLINE-Assignment-2
   ```
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib scikit-learn jupyter
   ```
3. Run the notebook:
   ```bash
   jupyter notebook Assignment-3.ipynb
   ```
   OR run the plain script:
   ```bash
   python Assignment-3.py
   ```

---

## 📌 Submission Checklist
- [x] `Assignment-3.ipynb` / `Assignment-3.py` included
- [x] `README.md` with Objective, Dataset Link, Libraries Used, Methodology, Results, Conclusion
- [x] Dataset link to Kaggle provided
- [x] Repository set to **Public**
