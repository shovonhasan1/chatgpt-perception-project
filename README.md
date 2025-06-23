# Student Perception of ChatGPT – Insights from Data

**“Beyond the AI hype: What do students really think about ChatGPT?”**

This project explores how students perceive and use ChatGPT — diving into beliefs about its usefulness, ethical risks, academic reliance, and mimetic influence. By combining survey data, Python-based analysis, hypothesis testing, and Power BI dashboards, we reveal how Gen Z interacts with generative AI in educational spaces.

---

## Project Objective

To analyze real-world student survey data and uncover:
- How students **perceive ChatGPT** (as helpful, risky, or both)
- Whether **mimetic desire** (peer influence) affects adoption
- Which demographics are most likely to **trust or depend on AI**
- Patterns in **AI usage behavior, ethical tension, and future outlook**

---

## Project Structure

student-chatgpt-perception/
├── Hypothesis testing.py ← Python script for statistical tests
├── excel.py ← Excel cleaning & conversion script
├── final dataset.xlsx ← Cleaned survey dataset
├── survey_with_indices.parquet ← Survey + composite indices (belief, ethics)
├── clean_chatgpt_survey.parquet ← Processed version for analysis
├── cleanead_analyzed_dataset_survey.parquet ← Model-ready data
├── DASHBOARD/
│ └── ChatGPT_Report.pbix ← Power BI dashboard (interactive)
│ └── preview.png ← Dashboard preview
└── README.md ← This file

---

## What Was Done

### 1. **Data Cleaning**
- Loaded Excel survey file with `pandas`
- Trimmed whitespace, fixed inconsistent values
- Removed duplicates and incomplete rows (especially on key variables)
- Converted multi-answer fields (e.g., platform use) into exploded rows

### 2. **Composite Index Creation**
- Built **composite perception scores** using Likert-scale responses:
  - **AI Skill Confidence Index**
  - **Employability Belief Index**
  - **Ethical Concern Index**
  - **Mimetic Pressure Index**

### 3. **Hypothesis Testing**
Used statistical methods (t-tests, ANOVA) to analyze:
- Gender-wise difference in employability belief
- Whether perceived usefulness correlates with usage frequency
- Mimetic behavior patterns: do students use ChatGPT because others do?
- Economic category vs perceived AI skill confidence

### 4. **Visualization & Dashboarding**
- Created a Power BI dashboard using cleaned `.xlsx` and `.parquet` files
- KPI cards: average AI skill, employability belief, mimetic pressure
- Bar charts: Score breakdown by gender, platform, and academic background
- Donut and tree charts: Economic background vs belief indexes

---

## Key Insights

- **73% of students** use ChatGPT weekly for academic support
- Students with higher **AI confidence** also had stronger **employability belief**
- Mimetic influence is real: **65% admitted they tried ChatGPT after friends did**
- Students from developing regions had **higher ethical concerns**
- Those who saw ChatGPT as “replacing thinking” scored lower on trust/use scale

---

## Tools & Libraries Used

| Tool            | Purpose                                 |
|-----------------|------------------------------------------|
| Python 3.12      | Core data processing & hypothesis testing |
| `pandas`         | Data manipulation                       |
| `matplotlib`, `seaborn` | Visual exploration                  |
| `scipy.stats`    | t-tests, ANOVA, correlation             |
| `openpyxl`, `xlsxwriter` | Excel reading/writing            |
| Power BI         | Final dashboard & storytelling          |

---

## How to Run This Project

### 1. Clone the Repository
git clone https://github.com/yourusername/student-chatgpt-perception.git
cd student-chatgpt-perception
### 2. Set up Virtual Environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
### 3. Install Requirements
pip install pandas seaborn matplotlib scipy openpyxl xlsxwriter
### 4. Run the Scripts
python Hypothesis\ testing.py
python excel.py
