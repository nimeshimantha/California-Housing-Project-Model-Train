# Import Libraries
import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
import seaborn as sns  # Added for heatmap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load Dataset
data = pd.read_csv("housing_data.csv")

# Display First 10 Rows
print("First 10 rows:")
print(data.head(10))

# Descriptive Statistics
print("\nDescriptive Statistics:")
for col in ["median_income", "total_rooms", "population"]:
    print(f"\n{col}:")
    print("Mean:", data[col].mean())
    print("Median:", data[col].median())
    print("Std Dev:", data[col].std())
    print("Min:", data[col].min())
    print("Max:", data[col].max())

# Check Missing Values
print("\nMissing Values Check:")
print(data.isnull().sum())

# Handle Missing Values (Median Imputation for numeric columns)
if data.isnull().any().any():
    for column in data.columns:
        if data[column].isnull().any() and is_numeric_dtype(data[column]):
            column_median = data[column].median()
            data[column] = data[column].fillna(column_median)
            print(f"Filled missing values in '{column}' with median value {column_median:.2f}")

# Histogram of Target Variable
plt.hist(data["median_house_value"], bins=50, color="skyblue", edgecolor="black")
plt.title("Histogram of Median House Value")
plt.xlabel("House Value ($1000s)")
plt.ylabel("Frequency")
plt.show()

# Scatter Plot: Median Income vs House Value
plt.scatter(data["median_income"], data["median_house_value"], alpha=0.5, color="green")
plt.title("Median Income vs House Value")
plt.xlabel("Median Income")
plt.ylabel("House Value")
plt.show()

# Geographical Scatter: Latitude vs Longitude
plt.scatter(
    data["longitude"], data["latitude"],
    c=data["median_house_value"], cmap="viridis", alpha=0.5
)
plt.colorbar(label="House Value")
plt.title("Geographical Scatter (Latitude vs Longitude)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# Correlation Matrix (numeric only)
numeric_data = data.select_dtypes(include=["float64", "int64"])
print("\nCorrelation with Median House Value:")
print(numeric_data.corr()["median_house_value"].sort_values(ascending=False))

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_data.corr(), cmap="coolwarm", annot=True, fmt=".2f")
plt.title("Correlation Heatmap of Numeric Features")
plt.show()

# Convert Categorical Column 'ocean_proximity' into Numbers
if "ocean_proximity" in data.columns:
    data = pd.get_dummies(data, columns=["ocean_proximity"])

print("\nDataset after encoding categorical column:")
print(data.head())

# efine Features (X) and Target (y)
X = data.drop("median_house_value", axis=1)
y = data["median_house_value"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression Model
lr = LinearRegression()
lr.fit(X_train, y_train)

print("\nLinear Regression Coefficients:", lr.coef_)
print("Intercept:", lr.intercept_)

y_pred_lr = lr.predict(X_test)
print("Linear Regression MSE:", mean_squared_error(y_test, y_pred_lr))
print("Linear Regression R²:", r2_score(y_test, y_pred_lr))

# Decision Tree Model
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)
print("\nDecision Tree MSE:", mean_squared_error(y_test, y_pred_dt))
print("Decision Tree R²:", r2_score(y_test, y_pred_dt))

# Compare Models
print("\n--- Model Comparison ---")
if r2_score(y_test, y_pred_dt) > r2_score(y_test, y_pred_lr):
    print("Decision Tree performs better (higher R² score).")
else:
    print("Linear Regression performs better (higher R² score).")

