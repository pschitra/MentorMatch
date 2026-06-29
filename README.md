# 🎯 MentorMatch UAE — Analytics Dashboard

> **CTO Analytics Dashboard** · MentorMatch UAE · SP Jain School of Global Management  
> Descriptive Analytics · Diagnostic Analysis · ML Classification · Model Evaluation

---

## 📋 What This Dashboard Does

A 7-section Streamlit analytics dashboard covering the full data science lifecycle:

| Section | Description |
|---------|-------------|
| 🏠 Overview | KPI cards, platform metrics, distribution charts |
| 📊 Descriptive Analytics | 6 interactive cross-tabulation heatmaps against career outcome |
| 🔍 Diagnostic Analysis | Deep dives on Age, Experience, Goals, Industry, Engagement |
| ⚙️ Feature Engineering | Encoding strategy, correlation matrix, feature matrix preview |
| 🤖 ML Classification | KNN, Decision Tree, Random Forest, Gradient Boosting with tunable hyperparameters |
| 📈 Model Performance | ROC curves, confusion matrices, precision/recall/F1/AUC for all 4 models |
| 💡 Key Findings | 8 findings, UAE market insights, 8 product recommendations |

---

## 🚀 Running Locally

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/mentormatch-dashboard.git
cd mentormatch-dashboard
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

The dashboard will open at **http://localhost:8501**

### 5. Load the dataset
- The app will automatically load `MentorMatch_UAE_Clean_Dataset.csv` if it's in the same folder
- Alternatively, upload it via the **sidebar file uploader**

---

## ☁️ Deploying to Streamlit Community Cloud (Free)

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: MentorMatch Analytics Dashboard"
   git remote add origin https://github.com/YOUR_USERNAME/mentormatch-dashboard.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click **New App**
   - Select your repository, branch (`main`), and file (`app.py`)
   - Click **Deploy** — your dashboard will be live in ~2 minutes

3. **Your dashboard URL will be:**
   ```
   https://YOUR_USERNAME-mentormatch-dashboard-app-XXXXX.streamlit.app
   ```

---

## 📂 Repository Structure

```
mentormatch-dashboard/
│
├── app.py                              # Main Streamlit application (1,491 lines)
├── MentorMatch_UAE_Clean_Dataset.csv   # Dataset (502 rows × 21 columns)
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

---

## 🤖 ML Models Included

| Model | Algorithm | Key Hyperparameter |
|-------|-----------|--------------------|
| KNN | K-Nearest Neighbours | K (adjustable 3–21) |
| Decision Tree | CART | Max Depth (adjustable 2–15) |
| Random Forest | Ensemble of Trees | N Estimators (50–300) |
| Gradient Boosting | Sequential Boosting | Learning Rate (0.01–0.3) |

**Evaluation metrics computed for each model:**
- Training Accuracy
- Testing Accuracy  
- 5-Fold Cross-Validation Accuracy (mean ± std)
- Precision (macro average)
- Recall (macro average)
- F1-Score (macro average)
- AUC-ROC
- Confusion Matrix (normalised + raw counts)
- Feature Importance (tree-based models)

---

## 📊 Dataset

**MentorMatch_UAE_Clean_Dataset.csv** — 502 rows × 21 columns

| Column | Type | Description |
|--------|------|-------------|
| Mentee_ID | str | Unique identifier |
| Age | int | Mentee age (22–45) |
| Industry | str | 8 industry categories |
| Experience_Level | str | Entry / Mid / Senior |
| Goal_Type | str | 5 goal categories |
| Corporate_Sponsored | str | Yes/No |
| Months_Subscribed | int | Subscription tenure |
| Sessions_Per_Month | int | 1–4 sessions |
| Mentor_Rating | float | 1.0–5.0 |
| Session_Completion_Rate_Pct | float | % sessions attended |
| NPS_Score | int | −100 to +100 |
| Career_Outcome_Achieved | str | **TARGET** Yes/No (41.2% positive) |
| Engagement_Score | float | Composite 0–100 |

---

## 🛠️ Tech Stack

- **Streamlit** — Dashboard framework
- **Plotly** — Interactive charts
- **Pandas / NumPy** — Data manipulation
- **Scikit-learn** — ML models and evaluation
- **Matplotlib / Seaborn** — Static visualisations

---

*MentorMatch UAE · Data Analytics Project · SP Jain School of Global Management, Dubai*
