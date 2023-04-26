import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset and take a random sample
n_samples = 1000  # Adjust the number of samples as needed
df = pd.read_csv('flights.csv', low_memory=False)
df_sample = df.sample(n=n_samples, random_state=42)

# Drop irrelevant columns
df_sample = df_sample.drop(['YEAR', 'FLIGHT_NUMBER', 'TAIL_NUMBER', 'TAXI_OUT', 'WHEELS_OFF', 'AIR_TIME', 'WHEELS_ON', 'TAXI_IN', 'CANCELLED', 'CANCELLATION_REASON', 'DIVERTED', 'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY'], axis=1)

# Deal with missing values
df_sample = df_sample.dropna(subset=['DEPARTURE_DELAY'])

# Encode categorical variables
df_sample = pd.get_dummies(df_sample, columns=['MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT'])

# Create a binary target variable for flight delays (1 for delayed, 0 for on-time)
df_sample['DELAYED'] = (df_sample['DEPARTURE_DELAY'] > 0).astype(int)
df_sample = df_sample.drop(['DEPARTURE_DELAY'], axis=1)

# Fill remaining missing values with the mean value of each column
df_sample = df_sample.fillna(df_sample.mean())

# Check for infinite values and replace them with NaN, then fill NaN with mean values
df_sample.replace([np.inf, -np.inf], np.nan, inplace=True)
df_sample.fillna(df_sample.mean(), inplace=True)

# Split the data into training and testing sets
X = df_sample.drop(['DELAYED'], axis=1)
y = df_sample['DELAYED']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_sise=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA
pca = PCA(n_components=0.95)  # Adjust the number of components as needed
X_train_pca = pca.fit_transform(X_train_scaled)
# Apply PCA to the test data
X_test_pca = pca.transform(X_test_scaled)

# Perform hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
grid_search.fit(X_train_pca, y_train)

# Print the best hyperparameters and the corresponding accuracy score
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Accuracy Score:", grid_search.best_score_)

# Train the RandomForest model with the best hyperparameters
best_rf = grid_search.best_estimator_
best_rf.fit(X_train_pca, y_train)

# Make predictions and evaluate the model
y_pred = best_rf.predict(X_test_pca)

print("Accuracy Score:")
print(accuracy_score(y_test, y_pred))
