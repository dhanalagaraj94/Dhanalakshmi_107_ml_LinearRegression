# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

# Load the synthetic healthcare dataset
df = pd.read_csv('synthetic_healthcare_data.csv')

# Data Preprocessing
# Handle Missing Data
# Checking for missing values
print(df.isnull().sum())

# Handle missing categorical values manually by filling with the most frequent value
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Smoker'] = df['Smoker'].fillna(df['Smoker'].mode()[0])
df['Region'] = df['Region'].fillna(df['Region'].mode()[0])
df['Chronic Disease'] = df['Chronic Disease'].fillna(df['Chronic Disease'].mode()[0])

# Encode Categorical Features
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Smoker'] = label_encoder.fit_transform(df['Smoker'])
df['Region'] = label_encoder.fit_transform(df['Region'])
df['Chronic Disease'] = label_encoder.fit_transform(df['Chronic Disease'])

# Normalize Numerical Features
scaler = StandardScaler()
df[['Age', 'BMI', 'Treatment Cost']] = scaler.fit_transform(df[['Age', 'BMI', 'Treatment Cost']])

# Split Data for Supervised Learning
# Split into features (X) and target (y)
X = df.drop('Treatment Cost', axis=1)
y = df['Treatment Cost']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Linear Regression
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)

# Predict on the test set
y_pred_lr = linear_regressor.predict(X_test)

# Calculate RMSE for Linear Regression
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
print(f"RMSE for Linear Regression: {rmse_lr}")

# Random Forest Regressor
random_forest = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = random_forest.predict(X_test)

# Calculate RMSE for Random Forest
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print(f"RMSE for Random Forest: {rmse_rf}")

# Apply PCA for dimensionality reduction to visualize clusters
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Fit KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_pca)

#Visualize KMeans Clustering Results
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Cluster'], cmap='viridis')
plt.title('KMeans Clustering of Patients')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.show()


# Visualize the relationship between Age and Treatment Cost
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Treatment Cost', data=df, color='blue')
plt.title('Age vs Treatment Cost')
plt.xlabel('Age')
plt.ylabel('Treatment Cost')
plt.show()
#
#Visualize the relationship between BMI and Treatment Cost
plt.figure(figsize=(10, 6))
sns.scatterplot(x='BMI', y='Treatment Cost', data=df, color='green')
plt.title('BMI vs Treatment Cost')
plt.xlabel('BMI')
plt.ylabel('Treatment Cost')
plt.show()
#
# Feature Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Feature Correlation Heatmap')
plt.show()

#  Distribution of Treatment Cost (Target Variable)
plt.figure(figsize=(10, 6))
sns.histplot(df['Treatment Cost'], bins=30, kde=True, color='purple')
plt.title('Distribution of Treatment Cost')
plt.xlabel('Treatment Cost')
plt.ylabel('Frequency')
plt.show()

#Distribution of Gender (Categorical Feature)
plt.figure(figsize=(10, 6))
sns.countplot(x='Gender', data=df, palette='Set2')
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()
