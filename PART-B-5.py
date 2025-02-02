import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# Load the Iris dataset from CSV
df = pd.read_csv("iris_dataset.csv")
print(df.head())  # Display the first few rows
print(df.tail())  # Display the last few rows

# Rename columns if necessary to match expected feature names
# Assuming the CSV has columns: sepal_length, sepal_width, petal_length, petal_width, species
df.columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'target']

# Convert categorical target labels to numerical if needed
label_encoder = LabelEncoder()
df['target'] = label_encoder.fit_transform(df['target'])

# Split the DataFrame into different species for visualization
df0 = df[df.target == 0]
df1 = df[df.target == 1]
df2 = df[df.target == 2]

# Scatter plot for Sepal Length vs Sepal Width
plt.figure(figsize=(10, 5))
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'], color="green", marker='+', label='Setosa')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'], color="blue", marker='.', label='Versicolor')
plt.scatter(df2['sepal length (cm)'], df2['sepal width (cm)'], color="red", marker='o', label='Virginica')
plt.title('Sepal Dimensions')
plt.legend()

# Scatter plot for Petal Length vs Petal Width
plt.figure(figsize=(10, 5))
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'], color="green", marker='+', label='Setosa')
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'], color="blue", marker='.', label='Versicolor')
plt.scatter(df2['petal length (cm)'], df2['petal width (cm)'], color="red", marker='o', label='Virginica')
plt.title('Petal Dimensions')
plt.legend()
plt.show()

# Prepare data for model training
X = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
y = df.target  # Target variable (species)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Create and train the k-NN classifier
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

# Evaluate the model's performance
accuracy = knn.score(X_test, y_test)
print(f'Accuracy of the k-NN classifier: {accuracy * 100}')
