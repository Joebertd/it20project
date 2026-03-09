import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import joblib
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="LoanIQ Fintech AI Dashboard",
    page_icon="💰",
    layout="wide"
)

# ---------------------------------------------------
# CSS DESIGN
# ---------------------------------------------------

st.markdown("""
<style>

.stApp {
    background: linear-gradient(120deg,#f0f4ff,#dbeafe,#eff6ff);
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
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("models/logistic_model.pkl")

model = load_model()

# ---------------------------------------------------
# DATABASE
# ---------------------------------------------------

def init_db():

    conn = sqlite3.connect("loan_database.db")
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS loans(
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

init_db()

# ---------------------------------------------------
# SAVE RECORD FUNCTION
# ---------------------------------------------------

def save_to_db(first,middle,last,address,work,years,
               gender,married,dependents,income,loan,credit,prediction):

    conn = sqlite3.connect("loan_database.db")
    c = conn.cursor()

    c.execute("""
    INSERT INTO loans(
        first_name,middle_name,last_name,address,occupation,years_work,
        gender,married,dependents,income,loan,credit,prediction
    )
    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
    """,(first,middle,last,address,work,years,
         gender,married,dependents,income,loan,credit,prediction))

    conn.commit()
    conn.close()

# ---------------------------------------------------
# SIDEBAR
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
# LOAN PREDICTION
# ---------------------------------------------------

if menu == "Loan Prediction":

    st.title("💰 LoanIQ AI Loan Approval")

    col1,col2,col3 = st.columns(3)

    with col1:
        first_name = st.text_input("First Name")
        middle_name = st.text_input("Middle Name")
        last_name = st.text_input("Last Name")

    with col2:
        address = st.text_area("Address")
        occupation = st.text_input("Occupation")
        years_work = st.number_input("Years in Work",0,40,1)

    with col3:
        gender = st.selectbox("Gender",["Male","Female"])
        married = st.selectbox("Married",["Yes","No"])
        dependents = st.number_input("Dependents",0,5,0)

    st.markdown("---")

    colA,colB = st.columns(2)

    with colA:
        income = st.number_input("Income",0.0,100000.0,5000.0)
        loan = st.number_input("Loan Amount",10.0,20000.0,150.0)

    with colB:
        credit = st.selectbox("Credit History",[1,0])
        term = st.selectbox("Loan Term",[12,36,60,120,180,240,300,360])

    if st.button("Predict Loan Approval"):

        features = [[income,loan,credit,dependents,term]]

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        result = "Approved" if prediction == 1 else "Rejected"

        if result == "Approved":
            st.success("Loan Approved")
        else:
            st.error("Loan Rejected")

        save_to_db(
            first_name,middle_name,last_name,address,occupation,years_work,
            gender,married,dependents,income,loan,credit,result
        )

        # AI Probability Gauge

        st.subheader("AI Approval Probability")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability*100,
            title={'text':"Approval Probability"},
            gauge={
                'axis':{'range':[0,100]},
                'steps':[
                    {'range':[0,40],'color':'red'},
                    {'range':[40,70],'color':'orange'},
                    {'range':[70,100],'color':'green'}
                ]
            }
        ))

        st.plotly_chart(fig)

# ---------------------------------------------------
# CSV BATCH PREDICTION
# ---------------------------------------------------

if menu == "Batch Prediction (CSV)":

    st.title("Batch Loan Prediction")

    file = st.file_uploader("Upload CSV",type=["csv"])

    if file:

        df = pd.read_csv(file)

        st.subheader("Uploaded Dataset")
        st.dataframe(df)

        df.columns = df.columns.str.lower()

        credit_col = None
        income_col = None
        loan_col = None

        for col in df.columns:

            if "credit" in col:
                credit_col = col

            if "income" in col or "salary" in col:
                income_col = col

            if "loan" in col:
                loan_col = col

        if credit_col and income_col:

            df["prediction"] = np.where(
                (df[credit_col]==1) &
                (df[income_col]>3000),
                "Approved",
                "Rejected"
            )

            st.subheader("Prediction Results")
            st.dataframe(df)

            conn = sqlite3.connect("loan_database.db")
            cursor = conn.cursor()

            for _,row in df.iterrows():

                cursor.execute("""
                INSERT INTO loans(
                    first_name,middle_name,last_name,address,
                    occupation,years_work,gender,married,
                    dependents,income,loan,credit,prediction
                )
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,(

                    row.get("first_name",""),
                    row.get("middle_name",""),
                    row.get("last_name",""),
                    row.get("address",""),
                    row.get("occupation",""),
                    row.get("years_work",0),
                    row.get("gender",""),
                    row.get("married",""),
                    row.get("dependents",0),
                    row.get(income_col,0),
                    row.get(loan_col,0),
                    row.get(credit_col,0),
                    row["prediction"]

                ))

            conn.commit()
            conn.close()

            st.success("Batch results saved to database")

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

        st.subheader("Loan Risk Heatmap")

        fig,ax = plt.subplots()

        sns.heatmap(
            df[["income","loan","credit"]].corr(),
            annot=True,
            cmap="coolwarm"
        )

        st.pyplot(fig)

        st.subheader("Records")
        st.dataframe(df)

    else:

        st.info("No records yet")

# ---------------------------------------------------
# MODEL INSIGHTS
# ---------------------------------------------------

if menu == "Model Insights":

    st.title("Model Evaluation")

    data = pd.read_csv("loan_data_clean.csv")

    # normalize column names
    data.columns = data.columns.str.strip().str.lower()

    st.write("Detected Dataset Columns:", list(data.columns))

    # detect target column
    target_col = None

    for col in data.columns:
        if "status" in col or "approved" in col:
            target_col = col
            break

    if target_col is None:
        st.error("Target column not found in dataset.")
        st.stop()

    st.success(f"Detected target column: {target_col}")

    # remove categorical text columns
    drop_cols = [
        "employment_status",
        "education",
        "marital_status"
    ]

    data = data.drop(columns=[c for c in drop_cols if c in data.columns])

    # split features and labels
    X = data.drop(columns=[target_col])
    y = data[target_col]

    try:

        pred = model.predict(X)

        acc = accuracy_score(y, pred)

        st.metric("Model Accuracy", f"{acc*100:.2f}%")

        cm = confusion_matrix(y, pred)

        fig, ax = plt.subplots()

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Rejected","Approved"],
            yticklabels=["Rejected","Approved"]
        )

        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        st.pyplot(fig)

    except Exception as e:

        st.error("Model prediction failed.")
        st.write(e)

    # ---------------------------------------------------
    # FEATURE IMPORTANCE
    # ---------------------------------------------------

    st.subheader("Feature Importance")

    try:

        importance = np.abs(model.coef_[0])

        features = X.columns

        df_imp = pd.DataFrame({
            "Feature": features,
            "Importance": importance[:len(features)]
        })

        df_imp = df_imp.sort_values("Importance", ascending=False)

        st.bar_chart(df_imp.set_index("Feature"))

    except:
        st.warning("Feature importance unavailable for this model.")