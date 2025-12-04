import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# ==========================================
# PART 1: HELPER FUNCTIONS & DATA LOADING
# ==========================================

def format_idr(value):
    """
    Formats a number into Indonesian Rupiah style.
    Example: 1000000 -> Rp 1.000.000
    """
    return f"Rp {value:,.0f}".replace(",", ".")

@st.cache_data
def load_mortality_tables():
    """
    Attempts to load the Indonesia 2023 Mortality Tables from CSV.
    Returns two dictionaries: male_table, female_table.
    Structure: {age: qx, ...}
    """
    male_table = {}
    female_table = {}
    loaded_source = "Synthetic (Fallback)"

    # File names must match exactly what is on disk
    male_file = "Male Mortality Indonesia.csv"
    female_file = "Female Mortality Indonesia.csv"

    try:
        if os.path.exists(male_file) and os.path.exists(female_file):
            # --- Load Male Table ---
            # Based on file structure: Header is on line 3 (index 2)
            # Columns: x, Ex, dx, q x̂, px, ex
            df_m = pd.read_csv(male_file, header=2)

            # Clean and Convert Columns
            # Column 0 is Age (x), Column 3 is Mortality Rate (q x̂)
            df_m.iloc[:, 0] = pd.to_numeric(df_m.iloc[:, 0], errors='coerce')
            df_m.iloc[:, 3] = pd.to_numeric(df_m.iloc[:, 3], errors='coerce')

            # Drop rows where Age is NaN (cleanup footer or empty lines)
            df_m = df_m.dropna(subset=[df_m.columns[0]])

            # Create Dictionary {Age: qx}
            male_table = dict(zip(df_m.iloc[:, 0], df_m.iloc[:, 3]))

            # --- Load Female Table ---
            # Based on file structure: Header is on line 2 (index 1)
            df_f = pd.read_csv(female_file, header=1)

            # Clean and Convert Columns
            df_f.iloc[:, 0] = pd.to_numeric(df_f.iloc[:, 0], errors='coerce')
            df_f.iloc[:, 3] = pd.to_numeric(df_f.iloc[:, 3], errors='coerce')
            df_f = df_f.dropna(subset=[df_f.columns[0]])

            # Create Dictionary {Age: qx}
            female_table = dict(zip(df_f.iloc[:, 0], df_f.iloc[:, 3]))

            loaded_source = "Indonesia Mortality Table 2023 (CSV)"
        else:
            raise FileNotFoundError("CSV files not found")

    except Exception as e:
        # FALLBACK: Generate synthetic data if files are missing or broken
        # This ensures the app doesn't crash if files aren't present
        male_table = _generate_synthetic_mortality("Male")
        female_table = _generate_synthetic_mortality("Female")
        loaded_source = f"Synthetic Data (Error loading CSV: {str(e)[:50]}...)"

    return male_table, female_table, loaded_source

def _generate_synthetic_mortality(gender):
    """Fallback generator if CSVs are missing"""
    table = {}
    age_offset = 5 if gender == "Female" else 0
    for age in range(0, 121):
        effective_age = max(0, age - age_offset)
        if effective_age < 30:
            qx = 0.0005
        else:
            qx = 0.0005 * np.exp(0.092 * (effective_age - 30))
        table[age] = min(qx, 1.0)
    return table

def calculate_life_annuity_factor(retirement_age, gender, discount_rate, mortality_tables):
    """
    Calculates the 'Cost of 1 Rupiah'.
    Meaning: How much cash do you need in the bank at age 65 to pay yourself
    Rp 1 per year until you die?
    """
    # Select the correct table
    if gender == "Male":
        table = mortality_tables[0]
    else:
        table = mortality_tables[1]

    total_pv = 0.0
    prob_survival = 1.0 # Start alive

    # Loop from retirement until age 115 (or max in table)
    for t in range(0, 115 - retirement_age):
        current_age = int(retirement_age + t)

        # 1. Discount Factor (Time Value of Money)
        v = 1 / ((1 + discount_rate) ** t)

        # 2. Add to total (PV of payment * Probability of getting it)
        total_pv += prob_survival * v

        # 3. Update survival probability for next year
        # Get q_x from table, default to 1.0 (death) if age not found (end of table)
        q_x = table.get(current_age, 1.0)

        # Safety check for bad data (e.g. negative probabilities)
        if pd.isna(q_x) or q_x < 0: q_x = 1.0

        prob_survival = prob_survival * (1 - q_x)

        if prob_survival < 0.0001: break

    return total_pv

def run_simulation(current_age, retire_age, current_salary,
                   salary_growth, investment_return, inflation,
                   employer_contrib_pct, personal_contrib_pct,
                   target_monthly_income_today_value, gender, mortality_tables):

    years_to_go = retire_age - current_age

    if years_to_go <= 0:
        return None, "Already Retired"

    # --- 1. THE GOAL (What do you need?) ---
    # Adjust target income for inflation.
    future_inflation_factor = (1 + inflation) ** years_to_go
    target_annual_income_future = (target_monthly_income_today_value * 12) * future_inflation_factor

    # Real Discount Rate (approximate) for annuity valuation
    safe_withdrawal_rate_return = 0.04
    annuity_factor = calculate_life_annuity_factor(retire_age, gender, safe_withdrawal_rate_return, mortality_tables)

    # The "Pot" needed
    total_nest_egg_needed = target_annual_income_future * annuity_factor

    # --- 2. THE REALITY (What will you have?) ---
    projected_balance = 0.0
    yearly_salary = current_salary

    total_contribution_rate = employer_contrib_pct + personal_contrib_pct

    # Accumulation Loop
    for year in range(years_to_go):
        contribution = yearly_salary * total_contribution_rate
        projected_balance += contribution

        # Grow investment
        projected_balance *= (1 + investment_return)

        # Raise Salary
        yearly_salary *= (1 + salary_growth)

    # --- 3. THE GAP ---
    shortfall = total_nest_egg_needed - projected_balance

    # --- 4. THE SOLUTION ---
    if shortfall > 0:
        extra_annual_needed = shortfall * investment_return / ((1 + investment_return)**years_to_go - 1)
        extra_monthly_needed = extra_annual_needed / 12
    else:
        extra_monthly_needed = 0

    return {
        "nest_egg_needed": total_nest_egg_needed,
        "projected_balance": projected_balance,
        "shortfall": shortfall,
        "extra_monthly_needed": extra_monthly_needed,
        "future_monthly_target": target_annual_income_future / 12,
        "years_to_go": years_to_go,
        "annuity_factor": annuity_factor
    }, "Success"


# ==========================================
# PART 2: THE USER INTERFACE
# ==========================================

st.set_page_config(page_title="Retirement Calculator IDR", layout="wide", page_icon="🏦")

# Load Data
male_table, female_table, data_source = load_mortality_tables()
mortality_tables = (male_table, female_table)

st.title("🏦 Is Your Pension Enough? (IDR)")
st.markdown("Use this tool to see if your current savings plan will meet your dream retirement income in Indonesia.")

# Status Indicator for Data Source
if "Synthetic" in data_source:
    st.warning(f"⚠️ **Note:** Using synthetic mortality data. To use real Indonesian data, ensure the CSV files are in the same folder. ({data_source})")
else:
    st.success(f"✅ **Verified:** Calculation based on {data_source}")

# --- SIDEBAR INPUTS ---
st.sidebar.header("1. About You")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"], help="Uses Indonesia Mortality Table 2023 for calculations")
current_age = st.sidebar.slider("Current Age", 20, 70, 30)
retire_age = st.sidebar.slider("Target Retirement Age", 50, 80, 65)

# (Graph removed as requested)

st.sidebar.header("2. Income & Goals")
salary_type = st.sidebar.radio("Salary Input", ["Monthly", "Yearly"], horizontal=True)
salary_input = st.sidebar.number_input("Current Gross Salary (IDR)", min_value=0, value=5000000, step=500000)

target_income = st.sidebar.number_input("Desired Monthly Income at Retirement (in today's values)",
                                        min_value=0, value=3000000, step=500000,
                                        help="How much purchasing power do you want per month when you retire? e.g., Rp 3.000.000")

st.sidebar.header("3. Contributions")
employer_match = st.sidebar.slider("Employer Contribution (%)", 0.0, 20.0, 5.0, step=0.5) / 100
personal_contrib = st.sidebar.slider("Your Contribution (%)", 0.0, 50.0, 5.0, step=0.5) / 100

st.sidebar.header("4. Economic Assumptions")
invest_return = st.sidebar.number_input("Expected Investment Return (%)", 1.0, 15.0, 7.0, step=0.5) / 100
salary_growth = st.sidebar.number_input("Expected Salary Raises (%)", 0.0, 10.0, 3.0, step=0.5) / 100
inflation = st.sidebar.number_input("Expected Inflation (%)", 0.0, 10.0, 2.5, step=0.5) / 100

# --- NORMALIZATION ---
annual_salary = salary_input * 12 if salary_type == "Monthly" else salary_input

# --- CALCULATION ---
results, status = run_simulation(
    current_age, retire_age, annual_salary,
    salary_growth, invest_return, inflation,
    employer_match, personal_contrib,
    target_income, gender, mortality_tables
)

if status != "Success":
    st.error(status)
else:
    # --- MAIN DISPLAY ---

    # 1. The Big Verdict
    if results['shortfall'] <= 0:
        st.success(f"🎉 Congratulations! You are on track to exceed your goal by {format_idr(abs(results['shortfall']))}!")
    else:
        st.warning(f"⚠️ You have a projected gap of {format_idr(results['shortfall'])}.")

    st.divider()

    # 2. Key Metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Total Fund Needed",
            value=format_idr(results['nest_egg_needed']),
            help=f"Based on real Indonesian mortality data ({gender}), you need this much to sustain your income."
        )

    with col2:
        # Calculate delta string
        if results['shortfall'] > 0:
            delta_str = f"-{format_idr(results['shortfall'])}"
        else:
            delta_str = f"+{format_idr(abs(results['shortfall']))}"

        st.metric(
            label="Projected Fund Balance",
            value=format_idr(results['projected_balance']),
            delta=delta_str,
        )

    with col3:
        if results['shortfall'] > 0:
            st.metric(
                label="Extra Monthly Savings Needed",
                value=format_idr(results['extra_monthly_needed']),
                delta_color="inverse",
            )
        else:
            st.metric(label="Status", value="Fully Funded! 🚀")

    # 3. Visuals
    st.subheader("Visual Analysis")

    col_viz1, col_viz2 = st.columns(2)

    with col_viz1:
        # Gauge Chart for Fund Adequacy
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = results['projected_balance'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Retirement Readiness"},
            delta = {'reference': results['nest_egg_needed']},
            gauge = {
                'axis': {'range': [0, max(results['nest_egg_needed'] * 1.2, results['projected_balance'])]},
                'bar': {'color': "green" if results['shortfall'] <= 0 else "orange"},
                'steps': [
                    {'range': [0, results['nest_egg_needed']], 'color': "lightgray"},
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': results['nest_egg_needed']
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

    with col_viz2:
        # Explanation of "Cost of 1 IDR"
        st.info(f"""
        **The "Cost of Rp 1":**
        Based on the loaded actuarial tables (Indonesia 2023) for a {gender} retiring at {retire_age}:

        To pay yourself **Rp 1** (or 1 Rupiah) every year for life, you need to have **{results['annuity_factor']:.2f}** saved up at the moment you retire.

        This factor accounts for the probability of living to 80, 90, 100, etc., using specific data for Indonesia.
        """)

    # 4. Reality Check Section
    st.markdown("---")
    st.info(f"""
    **Reality Check:**
    You requested a retirement income of **{format_idr(target_income)}/month** in today's money.
    Due to {inflation*100:.1f}% inflation over the next {results['years_to_go']} years,
    you will actually need to receive **{format_idr(results['future_monthly_target'])}/month** when you retire to buy the exact same amount of groceries/goods.
    """)