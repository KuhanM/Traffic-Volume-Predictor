import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the dataset
data = pd.read_csv('traffic_data.csv')

# Fill missing values for rain and snow
data['rain_1h'] = data['rain_1h'].fillna(0)
data['snow_1h'] = data['snow_1h'].fillna(0)

# Convert date_time to datetime and extract hour and day_of_week
data['date_time'] = pd.to_datetime(data['date_time'])
data['hour'] = data['date_time'].dt.hour
data['day_of_week'] = data['date_time'].dt.dayofweek

# Drop original date_time column
data = data.drop('date_time', axis=1)

# Separate target variable and features
X = data.drop('traffic_volume', axis=1)
y = data['traffic_volume']

# Define categorical and numerical columns
categorical_cols = ['holiday', 'weather_main', 'weather_description', 'day_of_week']
numerical_cols = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'hour']

# Preprocessing for numerical data
numerical_transformer = StandardScaler()

# Preprocessing for categorical data
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])

# Define the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Create and evaluate the pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Save the model
joblib.dump(pipeline, 'traffic_volume_model.pkl')
