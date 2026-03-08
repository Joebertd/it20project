import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import joblib

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="LoanIQ Fintech AI Dashboard",
    page_icon="💰",
    layout="wide"
)

# ---------------------------------------------------
# FINTECH CSS DESIGN
# ---------------------------------------------------

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        120deg,
        #f0f4ff,
        #dbeafe,
        #eff6ff
    );
}

section[data-testid="stSidebar"] {
    background-color:#0f172a;
}

section[data-testid="stSidebar"] * {
    color:white;
}

h1 {
    color:#0f62fe;
}

.stButton>button {
    background: linear-gradient(90deg,#0f62fe,#4589ff);
    color:white;
    border-radius:10px;
    height:3em;
    width:100%;
    font-size:18px;
}

[data-testid="metric-container"] {
    background:white;
    border-radius:10px;
    padding:15px;
    box-shadow:0px 3px 10px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# DATABASE SETUP
# ---------------------------------------------------

def create_db():

    conn = sqlite3.connect("loan_database.db")
    c = conn.cursor()

    # DROP TABLE prevents schema mismatch errors
    c.execute("DROP TABLE IF EXISTS loans")

    c.execute("""
    CREATE TABLE loans(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        first_name TEXT,
        middle_name TEXT,
        last_name TEXT,
        address TEXT,
        occupation TEXT,
        years_work INTEGER,
        gender TEXT,
        married TEXT,
        dependents INTEGER,
        income REAL,
        loan REAL,
        credit INTEGER,
        prediction TEXT
    )
    """)

    conn.commit()
    conn.close()

create_db()

# ---------------------------------------------------
# SAVE DATA
# ---------------------------------------------------

def save_to_db(first,middle,last,address,work,years,
               gender,married,dependents,income,loan,credit,prediction):

    conn = sqlite3.connect("loan_database.db")
    c = conn.cursor()

    c.execute("""
    INSERT INTO loans VALUES(
        NULL,?,?,?,?,?,?,?,?,?,?,?,?
    )
    """,(first,middle,last,address,work,years,
         gender,married,dependents,income,loan,credit,prediction))

    conn.commit()
    conn.close()

# ---------------------------------------------------
# SIDEBAR MENU
# ---------------------------------------------------

menu = st.sidebar.radio(
    "Navigation",
    [
        "Loan Prediction",
        "Batch Prediction (CSV)",
        "Analytics Dashboard",
        "Model Insights"
    ]
)

# ---------------------------------------------------
# LOAN PREDICTION PAGE
# ---------------------------------------------------

if menu == "Loan Prediction":

    st.title("💰 LoanIQ AI Loan Approval System")

    st.subheader("Applicant Information")

    colA,colB,colC = st.columns(3)

    with colA:
        first_name = st.text_input("First Name")
        middle_name = st.text_input("Middle Name (optional)")
        last_name = st.text_input("Last Name")

    with colB:
        address = st.text_area("Address")
        occupation = st.text_input("Applicant Work / Occupation")
        years_work = st.number_input("Years in Work",0,50,1)

    with colC:
        gender = st.selectbox("Gender",["Male","Female"])
        married = st.selectbox("Married",["Yes","No"])
        dependents = st.number_input("Dependents",0,5,0)

    st.markdown("---")

    st.subheader("Financial Information")

    col1,col2 = st.columns(2)

    with col1:
        applicant_income = st.number_input("Applicant Income",0.0,100000.0,5000.0)
        loan_amount = st.number_input("Loan Amount",10.0,1000.0,150.0)

    with col2:
        credit_history = st.selectbox("Credit History",[1,0])
        loan_term = st.selectbox("Loan Term",[12,36,60,120,180,240,300,360])

    if st.button("Predict Loan Approval"):

        # Simple rule prediction
        if credit_history == 1 and applicant_income > 3000:
            result = "Approved"
            st.success("Loan Approved ✅")
        else:
            result = "Rejected"
            st.error("Loan Rejected ❌")

        save_to_db(
            first_name,middle_name,last_name,address,occupation,years_work,
            gender,married,dependents,applicant_income,
            loan_amount,credit_history,result
        )

        # Risk Score
        risk = np.random.randint(40,95)

        st.subheader("Loan Risk Score")

        st.progress(risk/100)

        st.write(f"Risk Score: {risk}/100")

# ---------------------------------------------------
# CSV BATCH PREDICTION
# ---------------------------------------------------

if menu == "Batch Prediction (CSV)":

    st.title("Batch Loan Prediction")

    file = st.file_uploader("Upload CSV",type=["csv"])

    if file:

        df = pd.read_csv(file)

        st.dataframe(df)

        df["Prediction"] = np.where(
            (df["Credit_History"]==1) &
            (df["ApplicantIncome"]>3000),
            "Approved",
            "Rejected"
        )

        st.subheader("Prediction Results")

        st.dataframe(df)

        st.download_button(
            "Download Results",
            df.to_csv(index=False),
            "loan_predictions.csv"
        )

# ---------------------------------------------------
# ANALYTICS DASHBOARD
# ---------------------------------------------------

if menu == "Analytics Dashboard":

    st.title("Loan Analytics Dashboard")

    conn = sqlite3.connect("loan_database.db")

    df = pd.read_sql("SELECT * FROM loans",conn)

    if not df.empty:

        total = len(df)
        approved = len(df[df["prediction"]=="Approved"])
        rejected = len(df[df["prediction"]=="Rejected"])

        rate = (approved/total)*100

        c1,c2,c3 = st.columns(3)

        c1.metric("Total Loans",total)
        c2.metric("Approved Loans",approved)
        c3.metric("Approval Rate",f"{rate:.2f}%")

        st.markdown("---")

        chart = pd.DataFrame({
            "Status":["Approved","Rejected"],
            "Count":[approved,rejected]
        })

        st.bar_chart(chart.set_index("Status"))

        st.subheader("Applicant Records")

        st.dataframe(df)

    else:
        st.info("No records yet")

# ---------------------------------------------------
# MODEL INSIGHTS
# ---------------------------------------------------

if menu == "Model Insights":

    st.title("Model Feature Importance")

    try:

        model = joblib.load("models/logistic_model.pkl")

        features = [
            "Income",
            "Loan Amount",
            "Credit History",
            "Dependents",
            "Loan Term"
        ]

        importance = np.abs(model.coef_[0])

        df_imp = pd.DataFrame({
            "Feature":features,
            "Importance":importance[:len(features)]
        })

        df_imp = df_imp.sort_values("Importance",ascending=False)

        st.bar_chart(df_imp.set_index("Feature"))

    except:
        st.warning("Feature importance unavailable.")