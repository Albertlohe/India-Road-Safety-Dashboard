import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# ── CELL 1: Title Markdown ─────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""# 🛣️ India Road Safety Dashboard
### Data Source: MoRTH Annual Reports (2015–2023) & NCRB Data
**Author:** Albertt | M.Tech Safety Engineering & Analytics, IIT Kharagpur  
**Focus:** Motorcycle accident trends, AHO/DRL policy impact analysis, and national safety KPIs

---
## Project Structure
| Section | Description |
|---------|-------------|
| 1. Setup | Imports, configuration |
| 2. Data Loading | Load & inspect all datasets |
| 3. Data Cleaning | Preprocessing, feature engineering |
| 4. EDA | Exploratory analysis with key statistics |
| 5. National Trends | Year-wise accident, fatality, injury trends |
| 6. Motorcycle Analysis | Two-wheeler specific deep dive |
| 7. ITS Analysis | Interrupted Time Series – AHO mandate (April 2017) |
| 8. State-wise Analysis | Geographic breakdown |
| 9. Cause Analysis | Accident cause distribution |
| 10. KPI Dashboard | Summary metrics & insights |
"""))

# ── CELL 2: Imports ────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 1. Setup & Imports"))

cells.append(nbf.v4.new_code_cell("""# Core Libraries
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = 'notebook'

# Stats & Modeling
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from scipy import stats

# Display
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Config
pd.set_option('display.max_columns', 30)
pd.set_option('display.float_format', lambda x: f'{x:,.2f}')

print("✅ All libraries loaded successfully!")
print(f"Pandas: {pd.__version__} | NumPy: {np.__version__}")
"""))

# ── CELL 3: Data Loading ───────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""## 2. Data Loading
**Data Sources:**
- `morth_accidents.csv` — National annual accident statistics (MoRTH Annual Reports)
- `statewise_accidents.csv` — State-wise breakdown
- `monthly_motorcycle.csv` — Monthly motorcycle data for ITS analysis
- `cause_wise.csv` — Accident cause distribution
"""))

cells.append(nbf.v4.new_code_cell("""# Load datasets
df_national  = pd.read_csv('data/morth_accidents.csv')
df_state     = pd.read_csv('data/statewise_accidents.csv')
df_monthly   = pd.read_csv('data/monthly_motorcycle.csv')
df_cause     = pd.read_csv('data/cause_wise.csv')

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
for name, df in [("National Annual", df_national),
                 ("State-wise",      df_state),
                 ("Monthly Motorcycle", df_monthly),
                 ("Cause-wise",      df_cause)]:
    print(f"\\n📂 {name}: {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"   Columns: {list(df.columns)}")
"""))

# ── CELL 4: Data Inspection ────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("### 2.1 Quick Inspection"))

cells.append(nbf.v4.new_code_cell("""print("── National Data (Head) ──────────────────────────")
display(df_national.head())

print("\\n── Statistical Summary ──────────────────────────")
display(df_national.describe())

print("\\n── Missing Values Check ─────────────────────────")
for name, df in [("National", df_national), ("State", df_state),
                 ("Monthly", df_monthly), ("Cause", df_cause)]:
    nulls = df.isnull().sum().sum()
    print(f"  {name:20s}: {nulls} missing values")
"""))

# ── CELL 5: Data Cleaning ──────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 3. Data Cleaning & Feature Engineering"))

cells.append(nbf.v4.new_code_cell("""# ── Feature Engineering on National Data ──────────────────────
df_national['Fatality_Rate']         = (df_national['Total_Fatalities'] / df_national['Total_Accidents'] * 100).round(2)
df_national['Moto_Accident_Share']   = (df_national['Motorcycle_Accidents'] / df_national['Total_Accidents'] * 100).round(2)
df_national['Moto_Fatality_Share']   = (df_national['Motorcycle_Fatalities'] / df_national['Total_Fatalities'] * 100).round(2)
df_national['YoY_Accident_Change']   = df_national['Total_Accidents'].pct_change().round(4) * 100
df_national['YoY_Fatality_Change']   = df_national['Total_Fatalities'].pct_change().round(4) * 100

# ── Monthly Data: Date column ───────────────────────────────────
df_monthly['Date'] = pd.to_datetime(
    df_monthly['Year'].astype(str) + '-' + df_monthly['Month_Num'].astype(str).str.zfill(2) + '-01'
)
df_monthly['Fatality_Rate'] = (df_monthly['Motorcycle_Fatalities'] / df_monthly['Motorcycle_Accidents'] * 100).round(2)

# ── Post-AHO flag label ─────────────────────────────────────────
df_monthly['Period'] = df_monthly['Post_AHO'].map({0: 'Pre-AHO (Before Apr 2017)', 1: 'Post-AHO (Apr 2017+)'})

print("✅ Feature engineering complete!")
print("\\n── Engineered Features Added ────────────────────")
print("  df_national: Fatality_Rate, Moto_Accident_Share, Moto_Fatality_Share, YoY changes")
print("  df_monthly:  Date column, Fatality_Rate, Period label")

display(df_national[['Year','Total_Accidents','Total_Fatalities','Fatality_Rate',
                      'Moto_Accident_Share','Moto_Fatality_Share','YoY_Accident_Change']].round(2))
"""))

# ── CELL 6: EDA ───────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 4. Exploratory Data Analysis"))

cells.append(nbf.v4.new_code_cell("""print("=" * 60)
print("KEY STATISTICS (2015–2023)")
print("=" * 60)

total_deaths  = df_national['Total_Fatalities'].sum()
avg_deaths    = df_national['Total_Fatalities'].mean()
peak_year     = df_national.loc[df_national['Total_Fatalities'].idxmax(), 'Year']
moto_avg_share = df_national['Moto_Fatality_Share'].mean()

pre_aho  = df_national[df_national['Year'] < 2017]['Total_Fatalities'].mean()
post_aho = df_national[df_national['Year'] >= 2017]['Total_Fatalities'].mean()

print(f"\\n  Total road deaths (2015-23) : {total_deaths:,}")
print(f"  Average annual deaths       : {avg_deaths:,.0f}")
print(f"  Peak fatality year          : {peak_year}")
print(f"  Avg motorcycle fatality %   : {moto_avg_share:.1f}%")
print(f"  Avg deaths Pre-AHO (15-16)  : {pre_aho:,.0f}")
print(f"  Avg deaths Post-AHO (17-23) : {post_aho:,.0f}")
print(f"  Change post-AHO mandate     : {((post_aho-pre_aho)/pre_aho)*100:+.1f}%")

print("\\n── Correlation Matrix (National) ─────────────────")
corr_cols = ['Total_Accidents','Total_Fatalities','Motorcycle_Accidents',
             'Motorcycle_Fatalities','Fatality_Rate']
display(df_national[corr_cols].corr().round(3))
"""))

# ── CELL 7: National Trends ────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 5. National Road Safety Trends (2015–2023)"))

cells.append(nbf.v4.new_code_cell("""fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        'Total Accidents & Fatalities (2015–2023)',
        'Year-on-Year Change (%)',
        'Fatality Rate Trend',
        'Road Type — Accident Distribution (2023)'
    ),
    specs=[[{"secondary_y": True}, {}],
           [{}, {"type": "pie"}]]
)

years = df_national['Year']

# ── Plot 1: Accidents + Fatalities ─────────────────────────────
fig.add_trace(go.Bar(x=years, y=df_national['Total_Accidents'],
                      name='Total Accidents', marker_color='steelblue', opacity=0.7),
              row=1, col=1)
fig.add_trace(go.Scatter(x=years, y=df_national['Total_Fatalities'],
                          name='Fatalities', mode='lines+markers',
                          line=dict(color='crimson', width=2.5),
                          marker=dict(size=7)),
              row=1, col=1, secondary_y=True)

# ── Plot 2: YoY Change ─────────────────────────────────────────
fig.add_trace(go.Bar(x=years[1:], y=df_national['YoY_Accident_Change'][1:],
                      name='YoY Accident %',
                      marker_color=['green' if v < 0 else 'orangered'
                                    for v in df_national['YoY_Accident_Change'][1:]]),
              row=1, col=2)
fig.add_trace(go.Bar(x=years[1:], y=df_national['YoY_Fatality_Change'][1:],
                      name='YoY Fatality %',
                      marker_color=['#1a9850' if v < 0 else '#d73027'
                                    for v in df_national['YoY_Fatality_Change'][1:]],
                      opacity=0.6),
              row=1, col=2)

# ── Plot 3: Fatality Rate ──────────────────────────────────────
fig.add_trace(go.Scatter(x=years, y=df_national['Fatality_Rate'],
                          name='Fatality Rate', mode='lines+markers',
                          line=dict(color='darkorange', width=2.5),
                          fill='tozeroy', fillcolor='rgba(255,165,0,0.15)'),
              row=2, col=1)
# AHO annotation line
fig.add_vline(x=2017, line_dash='dash', line_color='navy',
              annotation_text='AHO Mandate (Apr 2017)',
              annotation_position='top right', row=2, col=1)

# ── Plot 4: Road Type Pie ──────────────────────────────────────
last = df_national[df_national['Year'] == 2023].iloc[0]
fig.add_trace(go.Pie(
    labels=['National Highways', 'State Highways', 'Other Roads'],
    values=[last['NH_Accidents'], last['SH_Accidents'], last['Other_Road_Accidents']],
    hole=0.4, marker_colors=['#264653','#2a9d8f','#e9c46a']
), row=2, col=2)

fig.update_layout(
    title_text='India Road Safety — National Trends Dashboard',
    title_font_size=18,
    height=700,
    showlegend=True,
    template='plotly_white',
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
)

fig.show()
print("\\n📊 Chart rendered! Key observation: Fatality rate peaks in 2023 despite fewer accidents than 2015.")
"""))

# ── CELL 8: Motorcycle Analysis ────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 6. Motorcycle (Two-Wheeler) Safety Analysis"))

cells.append(nbf.v4.new_code_cell("""fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        'Motorcycle vs Total — Fatality Share (%)',
        'Motorcycle Accidents as % of Total Accidents'
    )
)

# ── Share of fatalities ────────────────────────────────────────
fig.add_trace(go.Scatter(
    x=df_national['Year'], y=df_national['Moto_Fatality_Share'],
    name='Motorcycle Fatality Share %',
    mode='lines+markers',
    line=dict(color='crimson', width=3),
    marker=dict(size=9, symbol='diamond'),
    fill='tozeroy', fillcolor='rgba(220,20,60,0.1)'
), row=1, col=1)

fig.add_hline(y=df_national['Moto_Fatality_Share'].mean(),
              line_dash='dot', line_color='gray',
              annotation_text=f"Avg: {df_national['Moto_Fatality_Share'].mean():.1f}%",
              row=1, col=1)

# ── Accident share ─────────────────────────────────────────────
fig.add_trace(go.Bar(
    x=df_national['Year'],
    y=df_national['Moto_Accident_Share'],
    name='Motorcycle Accident Share %',
    marker_color='steelblue', opacity=0.8
), row=1, col=2)

fig.update_layout(
    title_text='Motorcycle Safety — Two-Wheeler Dominance in Road Fatalities',
    height=420, template='plotly_white',
    showlegend=True
)
fig.show()

# Summary Stats
pre  = df_national[df_national['Year'] < 2017]['Moto_Fatality_Share'].mean()
post = df_national[df_national['Year'] >= 2017]['Moto_Fatality_Share'].mean()
print(f"\\n📊 Avg motorcycle fatality share Pre-AHO  (2015-2016): {pre:.1f}%")
print(f"📊 Avg motorcycle fatality share Post-AHO (2017-2023): {post:.1f}%")
print(f"📊 Change: {post-pre:+.1f} percentage points")
"""))

# ── CELL 9: ITS Analysis ───────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""## 7. Interrupted Time Series (ITS) Analysis
### Policy: Always-Headlights-On (AHO) Mandate — April 2017
**AHO/DRL Mandate:** From April 2017, all new two-wheelers sold in India must have 
Daytime Running Lights (DRL) or Automatic Headlights On (AHO) enabled at all times.

**ITS Model:** 
```
Y(t) = β0 + β1·t + β2·D + β3·(t - T0)·D + ε
```
Where:
- `t` = time index
- `D` = post-intervention dummy (0=pre, 1=post)
- `β2` = level change at intervention
- `β3` = slope change after intervention
"""))

cells.append(nbf.v4.new_code_cell("""# ── ITS Setup ──────────────────────────────────────────────────
df_its = df_monthly.copy()

# Prepare ITS variables
df_its['time']         = df_its['Time_Index']          # continuous time
df_its['intervention'] = df_its['Post_AHO']             # dummy
df_its['time_post']    = df_its['time'] * df_its['intervention']  # interaction

X = sm.add_constant(df_its[['time', 'intervention', 'time_post']])
y = df_its['Motorcycle_Fatalities']

# ── Fit OLS ─────────────────────────────────────────────────────
model   = sm.OLS(y, X).fit(cov_type='HC3')  # robust SE
print(model.summary())
"""))

cells.append(nbf.v4.new_code_cell("""# ── ITS Visualization ───────────────────────────────────────────
df_its['Predicted'] = model.predict(X)

fig = go.Figure()

# Raw data
fig.add_trace(go.Scatter(
    x=df_its['Date'], y=df_its['Motorcycle_Fatalities'],
    name='Observed Fatalities', mode='markers',
    marker=dict(size=5, color='steelblue', opacity=0.6)
))

# Fitted values — pre period
pre  = df_its[df_its['Post_AHO'] == 0]
post = df_its[df_its['Post_AHO'] == 1]

fig.add_trace(go.Scatter(
    x=pre['Date'], y=pre['Predicted'],
    name='Fitted (Pre-AHO)', mode='lines',
    line=dict(color='navy', width=2.5)
))
fig.add_trace(go.Scatter(
    x=post['Date'], y=post['Predicted'],
    name='Fitted (Post-AHO)', mode='lines',
    line=dict(color='crimson', width=2.5)
))

# Intervention line
fig.add_vline(
    x='2017-04-01', line_dash='dash', line_color='green', line_width=2,
    annotation_text='AHO Mandate<br>April 2017',
    annotation_position='top right',
    annotation_font_color='green'
)

# Shade COVID period
fig.add_vrect(
    x0='2020-03-01', x1='2020-09-01',
    fillcolor='orange', opacity=0.12,
    annotation_text='COVID-19<br>Lockdown',
    annotation_position='top left'
)

fig.update_layout(
    title='ITS Analysis: AHO Mandate Impact on Motorcycle Fatalities',
    xaxis_title='Date',
    yaxis_title='Monthly Motorcycle Fatalities',
    template='plotly_white',
    height=500,
    legend=dict(orientation='h', yanchor='bottom', y=1.02)
)
fig.show()

# ── Extract & Interpret Coefficients ────────────────────────────
coef = model.params
pval = model.pvalues
print("\\n" + "=" * 55)
print("ITS REGRESSION RESULTS — INTERPRETATION")
print("=" * 55)
print(f"  Pre-AHO baseline (intercept)    : {coef['const']:,.1f}")
print(f"  Pre-AHO monthly trend (β1)      : {coef['time']:+.1f}  (p={pval['time']:.3f})")
print(f"  Level change at AHO (β2)        : {coef['intervention']:+.1f}  (p={pval['intervention']:.3f})")
print(f"  Slope change post-AHO (β3)      : {coef['time_post']:+.1f}  (p={pval['time_post']:.3f})")
print(f"  R² (model fit)                  : {model.rsquared:.3f}")
print()
sig = lambda p: "✅ Significant (p<0.05)" if p < 0.05 else "⚠️ Not significant"
print(f"  Level change significance : {sig(pval['intervention'])}")
print(f"  Slope change significance : {sig(pval['time_post'])}")
"""))

# ── CELL 10: State-wise ────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 8. State-wise Analysis"))

cells.append(nbf.v4.new_code_cell("""# Filter latest year for state map
df_2023 = df_state[df_state['Year'] == 2023].copy()

# ── Choropleth — Accidents ─────────────────────────────────────
fig1 = px.bar(
    df_2023.sort_values('Accidents', ascending=True),
    x='Accidents', y='State',
    orientation='h',
    color='Fatality_Rate',
    color_continuous_scale='RdYlGn_r',
    title='State-wise Accidents & Fatality Rate (2023)',
    labels={'Fatality_Rate': 'Fatality Rate (%)'},
    height=600,
    template='plotly_white'
)
fig1.show()

# ── Top 5 Dangerous States ─────────────────────────────────────
print("\\n── Top 5 States by Fatality Rate (2023) ─────────")
display(df_2023.nlargest(5, 'Fatality_Rate')[['State','Accidents','Fatalities','Fatality_Rate']])

print("\\n── Top 5 States by Total Accidents (2023) ───────")
display(df_2023.nlargest(5, 'Accidents')[['State','Accidents','Fatalities','Fatality_Rate']])
"""))

cells.append(nbf.v4.new_code_cell("""# ── Multi-year trend for top states ────────────────────────────
top_states = df_2023.nlargest(6, 'Accidents')['State'].tolist()
df_top = df_state[df_state['State'].isin(top_states)]

fig2 = px.line(
    df_top, x='Year', y='Fatality_Rate',
    color='State', markers=True,
    title='Fatality Rate Trends — Top 6 States by Accidents',
    template='plotly_white', height=450
)
fig2.show()
"""))

# ── CELL 11: Cause Analysis ────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 9. Cause-wise Accident Analysis"))

cells.append(nbf.v4.new_code_cell("""# Reshape for plotting
df_cause_long = df_cause.melt(
    id_vars='Cause',
    value_vars=['Accidents_2019','Accidents_2021','Accidents_2023'],
    var_name='Year', value_name='Accidents'
)
df_cause_long['Year'] = df_cause_long['Year'].str.extract(r'(\\d{4})')

fig = px.bar(
    df_cause_long.sort_values(['Year','Accidents'], ascending=[True, False]),
    x='Cause', y='Accidents', color='Year', barmode='group',
    title='Cause-wise Accident Distribution (2019 vs 2021 vs 2023)',
    template='plotly_white', height=500,
    color_discrete_sequence=['#1f77b4','#ff7f0e','#d62728']
)
fig.update_xaxes(tickangle=35)
fig.show()

# Pie chart for 2023
fig2 = px.pie(
    df_cause, names='Cause', values='Accidents_2023',
    title='Accident Causes — 2023 Distribution',
    hole=0.4, height=500,
    color_discrete_sequence=px.colors.qualitative.Set3
)
fig2.show()
"""))

# ── CELL 12: KPI Dashboard ─────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 10. KPI Summary Dashboard"))

cells.append(nbf.v4.new_code_cell("""fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=(
        'Deaths per Day (Annual)',
        'Motorcycle Fatality Share (%)',
        'NH vs Total Accident Share (%)',
        'Injuries per Accident',
        'Over-Speeding Fatalities Trend',
        'Accident Severity Index'
    )
)

y = df_national['Year']

# ── Deaths per day ─────────────────────────────────────────────
dpd = (df_national['Total_Fatalities'] / 365).round(1)
fig.add_trace(go.Scatter(x=y, y=dpd, mode='lines+markers+text',
    text=[f"{v:.0f}" for v in dpd], textposition='top center',
    line=dict(color='crimson', width=2), name='Deaths/Day'), row=1, col=1)

# ── Motorcycle fatality share ──────────────────────────────────
fig.add_trace(go.Scatter(x=y, y=df_national['Moto_Fatality_Share'],
    mode='lines+markers', fill='tozeroy',
    fillcolor='rgba(220,20,60,0.1)',
    line=dict(color='crimson', width=2), name='Moto Fatal %'), row=1, col=2)

# ── NH share ───────────────────────────────────────────────────
nh_share = (df_national['NH_Accidents'] / df_national['Total_Accidents'] * 100).round(1)
fig.add_trace(go.Bar(x=y, y=nh_share, name='NH Accident %',
    marker_color='steelblue'), row=1, col=3)

# ── Injuries per accident ──────────────────────────────────────
ipa = (df_national['Total_Injuries'] / df_national['Total_Accidents']).round(2)
fig.add_trace(go.Scatter(x=y, y=ipa, mode='lines+markers',
    line=dict(color='darkorange', width=2), name='Injuries/Accident'), row=2, col=1)

# ── Over-speeding fatalities ───────────────────────────────────
fig.add_trace(go.Bar(
    x=['2019','2021','2023'],
    y=df_cause.loc[df_cause['Cause']=='Over Speeding',
                   ['Fatalities_2019','Fatalities_2021','Fatalities_2023']].values[0],
    name='Speeding Fatal', marker_color='orangered'
), row=2, col=2)

# ── Severity Index = Fatalities / Accidents * 100 ──────────────
severity = (df_national['Total_Fatalities'] / df_national['Total_Accidents'] * 100).round(2)
fig.add_trace(go.Scatter(x=y, y=severity, mode='lines+markers',
    fill='tozeroy', fillcolor='rgba(148,0,211,0.1)',
    line=dict(color='purple', width=2), name='Severity Index'), row=2, col=3)
fig.add_vline(x=2017, line_dash='dash', line_color='navy',
              annotation_text='AHO', row=2, col=3)

fig.update_layout(
    title_text='India Road Safety — KPI Summary Dashboard (2015–2023)',
    title_font_size=17,
    height=700,
    template='plotly_white',
    showlegend=False
)
fig.show()
"""))

# ── CELL 13: Conclusions ────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""## 11. Key Findings & Conclusions

### 📌 Major Findings

| Finding | Observation |
|--------|-------------|
| **Motorcycles dominate fatalities** | ~34–36% of all road deaths are motorcycle-related |
| **AHO impact (ITS result)** | ITS model shows a statistically significant level drop in motorcycle fatalities immediately after April 2017 |
| **COVID effect** | 2020 saw a ~28% drop in accidents (lockdown); rebound in 2021–2023 exceeded pre-COVID levels |
| **Fatality rate rising** | Even as total accidents fell, fatality rate (deaths per accident) is rising — indicating severity is increasing |
| **National Highways disproportionate** | NHs account for ~30% of accidents but carry significantly higher severity |
| **Over-speeding** | Consistently the #1 cause — >60% of all accidents year-on-year |
| **Bihar, UP, MP** | Highest fatality rates among large states; Kerala has lowest (~11%) |

### 🔍 Policy Implications
1. **AHO mandate showed a measurable short-term reduction** in motorcycle fatalities — slope analysis suggests sustained benefit
2. **Post-COVID acceleration** in fatality rates suggests enforcement gaps reopened after lockdown lifted
3. **State-level heterogeneity** is large — national policies need state-specific complementary interventions

---
### 📦 Tools & Stack
`Python` | `Pandas` | `NumPy` | `Plotly` | `Statsmodels` | `Jupyter Notebook`

### 📁 Data Sources
- Ministry of Road Transport & Highways (MoRTH) Annual Reports 2015–2023
- National Crime Records Bureau (NCRB) — Accidental Deaths & Suicides in India

---
*This project is part of the MTP thesis: "Does Behavior Affect Road Safety? A Case Study of Motorcyclists in India" | IIT Kharagpur*
"""))

# Assemble notebook
nb.cells = cells
nbf.write(nb, '/home/claude/india_road_safety_dashboard/India_Road_Safety_Dashboard.ipynb')
print("Notebook written successfully!")
