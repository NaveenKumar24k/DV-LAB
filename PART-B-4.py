import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# Load the dataset
df = pd.read_csv("sales.csv")

# Display the first few rows of the dataset to confirm it's loaded correctly
print(df.head())

# Define predictors (independent variables) and target (dependent variable)
X = df[["AdvertisingExpenditure", "Competition", "StoreLocation"]]
Y = df["SalesRevenue"]

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, Y_train)

# Retrieve the coefficients and intercept
coefficients = model.coef_
intercept = model.intercept_
print(f"Coefficients = {coefficients}\nIntercept = {intercept}")

# Predict the sales revenue for the test set
Y_pred = model.predict(X_test)

# Plot the relationships between predictors and sales revenue
predictors = ["AdvertisingExpenditure", "Competition", "StoreLocation"]

for predictor in predictors:
    plt.figure(figsize=(8, 6))
    plt.title(f"{predictor} vs SalesRevenue")
    plt.xlabel(predictor)
    plt.ylabel("SalesRevenue")
    plt.scatter(X_test[predictor], Y_test, color="b", label="Actual Data")
    plt.scatter(X_test[predictor], Y_pred, color="r", label="Predicted Data")
    plt.legend()
    plt.show()

# Perform statistical analysis using statsmodels
X_with_const = sm.add_constant(X)  # Add constant for intercept
model_sm = sm.OLS(Y, X_with_const).fit()

# Print the summary of the model
print(model_sm.summary())

# Perform t-tests for each predictor
for predictor in predictors:
    t_statistic = model_sm.tvalues[predictor]
    p_value_t = model_sm.pvalues[predictor]

    print(f"t-statistic for {predictor} = {t_statistic}")

    if p_value_t < 0.05:
        print(f"{predictor} is a statistically significant predictor of SalesRevenue.")
    else:
        print(f"{predictor} is NOT a statistically significant predictor of SalesRevenue.")

# Perform an F-test for overall model significance
f_statistic = model_sm.fvalue
p_value_f = model_sm.f_pvalue

print(f"F-statistic for the overall model = {f_statistic}")
if p_value_f < 0.05:
    print("The overall model is statistically significant.")
else:
    print("The overall model is NOT statistically significant.")

# Analyze individual predictors in isolation (optional)
for predictor in predictors:
    X_with_const_single = sm.add_constant(X[[predictor]])  # Use only one predictor
    model_single = sm.OLS(Y, X_with_const_single).fit()

    f_statistic_single = model_single.fvalue
    p_value_f_single = model_single.f_pvalue

    print(f"F-statistic for {predictor} = {f_statistic_single}")

    if p_value_f_single < 0.05:
        print(f"{predictor} is a statistically significant predictor of SalesRevenue when considered alone.")
    else:
        print(f"{predictor} is NOT a statistically significant predictor of SalesRevenue when considered alone.")
