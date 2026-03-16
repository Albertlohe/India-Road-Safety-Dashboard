#  India Road Safety Dashboard

> **An end-to-end analytics project on India's road accident data (2015–2023) using MoRTH & NCRB datasets, with a focus on motorcycle safety and the AHO/DRL policy impact via Interrupted Time Series analysis.**

---

##  Project Overview

India records over **4,50,000 road accidents annually**, making it one of the worst road safety performers globally. This project builds an interactive analytical dashboard in Jupyter that:

- Visualizes national and state-level road safety trends
- Deep-dives into **motorcycle/two-wheeler** accident patterns
- Applies **Interrupted Time Series (ITS)** regression to evaluate the **April 2017 AHO (Always Headlights On) mandate**
- Generates a comprehensive KPI summary dashboard


---

## Repository Structure

```
india_road_safety_dashboard/
│
├── India_Road_Safety_Dashboard.ipynb   ← Main analysis notebook
│
├── data/
│   ├── morth_accidents.csv             ← National annual accident data (2015–2023)
│   ├── statewise_accidents.csv         ← State-wise breakdown
│   ├── monthly_motorcycle.csv          ← Monthly two-wheeler data (for ITS)
│   └── cause_wise.csv                  ← Accident cause distribution
│
├── requirements.txt                    ← Python dependencies
└── README.md
```

---

## 📊 Dashboard Sections

| Section | Description |
|---------|-------------|
| **1. National Trends** | Accidents, fatalities, injuries (2015–2023) with YoY change |
| **2. Motorcycle Analysis** | Two-wheeler share of accidents and fatalities over time |
| **3. ITS Analysis** | Regression model evaluating AHO mandate impact |
| **4. State-wise Analysis** | Fatality rates and accident counts across major states |
| **5. Cause Analysis** | Over-speeding, drunk driving, mobile use breakdown |
| **6. KPI Dashboard** | Deaths/day, severity index, NH share, injury rate |

---

## Methodology — Interrupted Time Series

**Policy Evaluated:** AHO/DRL mandate requiring all new two-wheelers to have automatic headlights on (April 2017).

**Model:**
```
Y(t) = β0 + β1·t + β2·D + β3·(t − T0)·D + ε
```

| Parameter | Meaning |
|-----------|---------|
| `β1` | Pre-intervention monthly trend |
| `β2` | **Immediate level change** at mandate |
| `β3` | **Slope change** post-mandate |

Robust standard errors (HC3) used to handle heteroscedasticity.

---

## 🛠️ Tech Stack

| Tool | Usage |
|------|-------|
| Python 3.x | Core language |
| Pandas | Data loading, cleaning, feature engineering |
| NumPy | Numerical operations |
| Plotly | Interactive visualizations |
| Statsmodels | ITS regression (OLS with robust SE) |
| Jupyter Notebook | Development environment |

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/india_road_safety_dashboard.git
cd india_road_safety_dashboard
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch Jupyter
```bash
jupyter notebook India_Road_Safety_Dashboard.ipynb
```

---

##  Key Findings

-  **Motorcycles account for ~35% of all road fatalities** — consistently the highest-risk vehicle category
-  **ITS model shows a statistically significant level drop** in motorcycle fatalities post-AHO mandate (April 2017)
-  **COVID-19 caused a 28% drop in accidents in 2020** — but 2021–2023 shows a sharp rebound exceeding pre-pandemic levels
-  **Fatality rate (deaths per accident) is rising** despite absolute accident count declining — severity is worsening
-  **Over-speeding accounts for >60% of all accidents** year-on-year
-  **Bihar and Uttar Pradesh** have the highest fatality rates; **Kerala** consistently lowest (~11%)

---

## Data Sources

- [MoRTH Annual Reports](https://morth.nic.in/) — Ministry of Road Transport & Highways
- [NCRB Accidental Deaths & Suicides in India](https://ncrb.gov.in/) — National Crime Records Bureau

>  **Note:** The CSV files in `/data/` are structured to mirror MoRTH Annual Report format. Replace with official downloaded data for production use.

---

##  Author

**Albertt**  
M.Tech — Safety Engineering & Analytics  
Indian Institute of Technology, Kharagpur  

---

## License

MIT License — free to use with attribution.
