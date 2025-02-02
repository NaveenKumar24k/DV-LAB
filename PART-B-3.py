import matplotlib.pyplot as plt  
import pandas as pd  
from sklearn.metrics import r2_score, mean_squared_error  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression  
import statsmodels.api as sm  

# Step 1: Load the dataset  
df = pd.read_csv("student_data.csv")  

# Display the first few rows of the dataset  
print(df.head())  

# Step 2: Visualization to understand the relationship  
X = df[['StudyHours']]  
y = df['ExamScore']  

plt.scatter(X, y, color="b", marker="*")  
plt.xlabel("Study Hours")  
plt.ylabel("Exam Scores")  
plt.title("Study Hours vs Exam Scores")  
plt.show()  

# Step 3: Split the dataset into training and testing sets  
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.25)  

# Step 4: Create and train the Linear Regression model  
model = LinearRegression()  
model.fit(X_train, y_train)  

# Step 5: Make predictions on the test set  
y_pred = model.predict(X_test)  

# Visualize actual vs predicted values  
plt.title("Comparing Actual and Predicted Exam Scores")  
plt.xlabel("Study Hours")  
plt.ylabel("Exam Scores")  
plt.scatter(X_test, y_test, color="b", label="Actual Data")  
plt.plot(X_test, y_pred, color="r", label="Predicted Data")  
plt.legend()  
plt.show()  

# Step 6: Model Evaluation  
r2 = r2_score(y_test, y_pred)  
mse = mean_squared_error(y_test, y_pred)  

print(f"Mean Squared Error = {mse}\nr^2 = {r2}")  

# Step 7: Model Diagnosis  
# Add constant for OLS regression  
X_const = sm.add_constant(X)  
model_sm = sm.OLS(y, X_const).fit()  

# Print summary to check coefficients and significance  
print(model_sm.summary())  

# Step 8: Analyze Residuals  
# Calculate residuals  
residuals = y - model.predict(X)  

# Plot residuals  
plt.figure(figsize=(10, 6))  
plt.scatter(model.predict(X), residuals, color="b")  
plt.axhline(0, color='red', linestyle='--')  
plt.xlabel("Predicted Exam Scores")  
plt.ylabel("Residuals")  
plt.title("Residuals Analysis")  
plt.show()
