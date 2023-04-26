import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

# Load the dataset
df = pd.read_csv('flights.csv', low_memory=False)

def preprocess_df(df_sample):
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

    return df_sample

# Investigate the impact of sample size on model performance
sample_sises = [10000, 20000, 50000, 100000]
accuracy_scores = []

for size in sample_sises:
    df_sampled = df.sample(n=size, random_state=42)
    df_sampled = preprocess_df(df_sampled)

    X = df_sampled.drop(['DELAYED'], axis=1)
    y = df_sampled['DELAYED']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_sise=0.2, random_state=42)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

  # Apply PCA
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_pca, y_train)

    y_pred = rf.predict(X_test_pca)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

# Present the results in a table or graph format
plt.plot(sample_sises, accuracy_scores)
plt.xlabel('Sample Size')
plt.ylabel('Accuracy')
plt.title('Impact of Sample Size on Model Performance')
plt.show()
