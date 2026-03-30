import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data

df = pd.read_csv("data/churn.csv")

# Cleaning

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Graph 1: Tenure Distribution

plt.figure()
plt.hist(df["tenure"], bins=30)
plt.title("Tenure Distribution")
plt.xlabel("Tenure")
plt.ylabel("Customers")
plt.savefig("tenure_distribution.png")

# Graph 2: Tenure vs Churn

plt.figure()
sns.boxplot(x="Churn", y="tenure", data=df)
plt.title("Tenure vs Churn")
plt.savefig("tenure_vs_churn.png")


# Graph 3: Churn Rate by Tenure Group

df["tenure_group"] = pd.cut(
    df["tenure"], bins=[0, 12, 24, 48, 72], labels=["0-12", "12-24", "24-48", "48-72"]
)

churn_rate = df.groupby("tenure_group")["Churn"].mean()

plt.figure()
churn_rate.plot(kind="bar")
plt.title("Churn Rate by Tenure Group")
plt.ylabel("Churn Rate")
plt.savefig("churn_rate.png")

print("EDA graphs saved successfully ✅")
