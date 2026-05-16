import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans



# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Mall Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)



# =========================
# CUSTOM CSS
# =========================


st.markdown("""
<style>

/* DARK THEME BACKGROUND */
.stApp {
    background: linear-gradient(135deg, #0f172a, #111827, #1e1b4b);
    color: white;
}

/* MAIN TEXT */
h1, h2, h3, h4, h5, h6, p, label {
    color: white !important;
}

/* SIDEBAR GLASS EFFECT */
section[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.85);
    backdrop-filter: blur(15px);
    border-right: 1px solid rgba(255,255,255,0.15);
}

/* GRADIENT KPI CARDS */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #7c3aed, #2563eb);
    border-radius: 20px;
    padding: 22px;
    color: white;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.35);
    border: 1px solid rgba(255,255,255,0.25);
    animation: fadeIn 0.8s ease-in-out;
}

/* GLASS TABLE / CONTAINER EFFECT */
div[data-testid="stDataFrame"],
div[data-testid="stTable"],
div[data-testid="stPlotlyChart"],
div[data-testid="stPyplot"] {
    background: rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 15px;
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0px 8px 25px rgba(0,0,0,0.25);
    animation: slideUp 0.8s ease-in-out;
}

/* BUTTON STYLE */
.stButton>button {
    background: linear-gradient(135deg, #ec4899, #8b5cf6);
    color: white;
    border-radius: 14px;
    height: 50px;
    width: 100%;
    font-size: 18px;
    border: none;
    box-shadow: 0px 5px 15px rgba(236,72,153,0.4);
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.03);
    box-shadow: 0px 8px 20px rgba(236,72,153,0.6);
}

/* DOWNLOAD BUTTON */
.stDownloadButton>button {
    background: linear-gradient(135deg, #22c55e, #16a34a);
    color: white;
    border-radius: 14px;
    height: 50px;
    width: 100%;
    border: none;
    font-size: 18px;
    transition: 0.3s;
}

.stDownloadButton>button:hover {
    transform: scale(1.03);
}

/* INPUT BOX */
input, textarea {
    background-color: rgba(255,255,255,0.12) !important;
    color: white !important;
    border-radius: 12px !important;
}

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 12px;
    border: 1px solid rgba(255,255,255,0.18);
}

/* ANIMATION */
@keyframes fadeIn {
    from {
        opacity: 0;
        transform: scale(0.95);
    }
    to {
        opacity: 1;
        transform: scale(1);
    }
}

@keyframes slideUp {
    from {
        opacity: 0;
        transform: translateY(25px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* FOOTER */
.footer {
    text-align: center;
    padding: 25px;
    border-radius: 18px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    backdrop-filter: blur(15px);
}

</style>
""", unsafe_allow_html=True)


# =========================
# TITLE
# =========================
st.title("🛍️ Mall Customer Segmentation")

st.write("""
This Streamlit app uses **K-Means Clustering** to segment mall customers
based on their **Annual Income** and **Spending Score**.
""")

# =========================
# SIDEBAR
# =========================
st.sidebar.title("📂 Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("data/Mall_Customers.csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ File loaded successfully!")
else:
    df = load_data()
    st.sidebar.success("✅ Default dataset loaded!")

# =========================
# REQUIRED COLUMNS CHECK
# =========================
required_cols = ["Annual Income (k$)", "Spending Score (1-100)"]

if not all(col in df.columns for col in required_cols):
    st.error("Dataset must contain required columns.")
    st.stop()

# =========================
# CONTROLS
# =========================
st.sidebar.title("⚙️ Controls")

k = st.sidebar.slider(
    "Select number of clusters (K)",
    min_value=2,
    max_value=10,
    value=5
)

# =========================
# DATASET OVERVIEW
# =========================
st.header("1️⃣ Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Rows", df.shape[0])
col2.metric("Total Columns", df.shape[1])

col3.metric(
    "Avg Income",
    round(df["Annual Income (k$)"].mean(), 2)
)

col4.metric(
    "Avg Spending",
    round(df["Spending Score (1-100)"].mean(), 2)
)

st.subheader("📄 First 10 Rows")

st.dataframe(df.head(10), use_container_width=True)

# =========================
# K-MEANS
# =========================
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X)

# =========================
# CUSTOMER TYPE
# =========================
def customer_type(row):

    income = row["Annual Income (k$)"]
    score = row["Spending Score (1-100)"]

    if income >= 70 and score >= 60:
        return "VIP Customer"

    elif income >= 70 and score < 40:
        return "High Income Low Spending"

    elif income < 40 and score >= 60:
        return "Low Income High Spending"

    elif income < 40 and score < 40:
        return "Low Value Customer"

    else:
        return "Average Customer"

df["Customer Type"] = df.apply(customer_type, axis=1)

# =========================
# SCATTER PLOT
# =========================
st.header("2️⃣Scatter Plot - Customer Segmentation Graph")

fig, ax = plt.subplots(figsize=(10, 6))

colors = [
    "purple",
    "green",
    "red",
    "orange",
    "blue",
    "pink",
    "cyan",
    "yellow",
    "brown",
    "gray"
]

for cluster in sorted(df["Cluster"].unique()):

    cluster_data = df[df["Cluster"] == cluster]

    ax.scatter(
        cluster_data["Annual Income (k$)"],
        cluster_data["Spending Score (1-100)"],
        s=90,
        color=colors[cluster],
        label=f"Cluster {cluster}"
    )

centers = kmeans.cluster_centers_

ax.scatter(
    centers[:, 0],
    centers[:, 1],
    s=300,
    c="black",
    marker="X",
    label="Centroids"
)

ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score (1-100)")
ax.set_title("K-Means Customer Segmentation")

ax.legend()

st.pyplot(fig)


# =========================
# HISTOGRAM
# =========================

st.header("📊 Age Distribution Histogram")

fig4, ax4 = plt.subplots()

ax4.hist(
    df["Age"],
    bins=10,
    color="skyblue"
)

ax4.set_title("Age Distribution")

st.pyplot(fig4)


# =========================
# HEATMAP
# =========================

import seaborn as sns

st.header("🔥 Correlation Heatmap")

fig5, ax5 = plt.subplots(figsize=(8,5))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm",
    ax=ax5
)

st.pyplot(fig5)

# =========================
# BOXPLOT
# =========================

st.header("📦 Income Boxplot")

fig6, ax6 = plt.subplots()

ax6.boxplot(df["Annual Income (k$)"])

ax6.set_title("Annual Income Boxplot")

st.pyplot(fig6)

# =========================
# LINE CHART
# =========================

st.header("📈 Income Trend Line Chart")

fig7, ax7 = plt.subplots(figsize=(10,5))

ax7.plot(
    df["Annual Income (k$)"],
    color="green",
    linewidth=3
)

ax7.set_title("Income Trend")

st.pyplot(fig7)

# =========================
# BAR CHART
# =========================
st.header("3️⃣Bar Chart - Customer Type Count")

type_count = df["Customer Type"].value_counts()

fig3, ax3 = plt.subplots(figsize=(10,5))

colors = ["purple", "green", "orange", "red", "blue"]

ax3.bar(
    type_count.index,
    type_count.values,
    color=colors
)

ax3.set_title("Customer Type Count")
ax3.set_xlabel("Customer Type")
ax3.set_ylabel("Count")

plt.xticks(rotation=10)

st.pyplot(fig3)

# =========================
# PIE CHART
# =========================
st.header("4️⃣Pie Chart - Cluster Distribution")

cluster_count = df["Cluster"].value_counts()

fig2, ax2 = plt.subplots(figsize=(7, 7))

ax2.pie(
    cluster_count,
    labels=[f"Cluster {i}" for i in cluster_count.index],
    autopct="%1.1f%%",
    startangle=90
)

ax2.set_title("Cluster Distribution")

st.pyplot(fig2)

# =========================
# CUSTOMER INSIGHTS
# =========================
st.header("5️⃣ Customer Insights")

high_spending = df[df["Spending Score (1-100)"] >= 70]

low_income = df[df["Annual Income (k$)"] <= 40]

vip_customers = df[df["Customer Type"] == "VIP Customer"]

c1, c2, c3 = st.columns(3)

c1.metric(
    "High Spending Customers",
    len(high_spending)
)

c2.metric(
    "Low Income Customers",
    len(low_income)
)

c3.metric(
    "VIP Customers",
    len(vip_customers)
)

st.subheader("🌟 VIP Customers")

st.dataframe(
    vip_customers.head(10),
    use_container_width=True
)

# =========================
# PREDICTION SECTION
# =========================
st.header("6️⃣ Predict New Customer Segment")

income_input = st.number_input(
    "Enter Annual Income (k$)",
    min_value=0,
    max_value=200,
    value=60
)

score_input = st.number_input(
    "Enter Spending Score (1-100)",
    min_value=1,
    max_value=100,
    value=50
)

if st.button("Predict Cluster"):

    new_customer = [[income_input, score_input]]

    predicted_cluster = kmeans.predict(new_customer)[0]

    if income_input >= 70 and score_input >= 60:
        predicted_type = "VIP Customer"

    elif income_input >= 70 and score_input < 40:
        predicted_type = "High Income Low Spending"

    elif income_input < 40 and score_input >= 60:
        predicted_type = "Low Income High Spending"

    elif income_input < 40 and score_input < 40:
        predicted_type = "Low Value Customer"

    else:
        predicted_type = "Average Customer"

    st.success(
        f"✅ This customer belongs to Cluster {predicted_cluster}"
    )

    st.info(
        f"Customer Type: {predicted_type}"
    )

# =========================
# FINAL DATA
# =========================
st.header("7️⃣ Final Dataset With Clusters")

st.dataframe(df, use_container_width=True)

# =========================
# DOWNLOAD BUTTON
# =========================
csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Result CSV",
    data=csv,
    file_name="customer_segmentation_result.csv",
    mime="text/csv"
)

# =========================
# FOOTER
# =========================
st.markdown("---")

st.markdown(
    """
    <div class="footer">
        <h3>Made with ❤️ using Streamlit</h3>
        <p>Developed by: Israt Zahan Eva</p>
    </div>
    """,
    unsafe_allow_html=True
)