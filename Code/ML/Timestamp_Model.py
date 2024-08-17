import pandas as pd
from sklearn.ensemble import IsolationForest
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import joblib

data = pd.read_csv('/ProM-Ex/BPI_2015.csv')

data['time:timestamp'] = pd.to_datetime(data['time:timestamp'], format='%Y-%m-%d %H:%M:%S.%f%z', dayfirst=True)
#the above format can be changed to 'mixed' too.

data['hour'] = data['time:timestamp'].dt.hour
data['day_of_week'] = data['time:timestamp'].dt.dayofweek
data['day_of_month'] = data['time:timestamp'].dt.day
data['month'] = data['time:timestamp'].dt.month
data['year'] = data['time:timestamp'].dt.year
data['time_diff'] = data['time:timestamp'].diff().dt.total_seconds().fillna(0)


features = ['hour', 'day_of_week', 'day_of_month', 'month', 'year', 'time_diff']
X = data[features]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


model = IsolationForest(contamination=0.01, random_state=42)
model.fit(X_scaled)


predictions = model.predict(X_scaled)
anomalies = data[predictions == -1]


activity_column_candidates = ["Activity", "Action","ActivityID", "Event", "Description", "concept:name"]  


if any(candidate in data.columns for candidate in activity_column_candidates):
    for candidate in activity_column_candidates:
        if candidate in data.columns:
            activity_column = candidate
            break
else:
    print("Error: None of the potential activity column names ('Activity', 'Action', 'Event', 'Description') were found in the data. Activity information will be unavailable.")
    

joblib.dump(model, "timestamp_anomaly_activity_detection_model.pkl")
