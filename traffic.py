import streamlit as st
import pandas as pd
import joblib

# Load the dataset
data = pd.read_csv('traffic_data.csv')

# Identify categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns.tolist()

# Create a dictionary to store the unique values for each categorical column
unique_values = {col: data[col].unique().tolist() for col in categorical_columns}

# Load the lightweight model
model = joblib.load('traffic_volume_model.pkl')

# Add custom CSS to style the background image and other elements
st.markdown("""
    <style>
    .stApp {
        background-image: url('https://marketingaccesspass.com/wp-content/uploads/2015/10/Podcast-Website-Design-Background-Image.jpg');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: #f0f0f0;  /* Light text color to contrast with dark background */
    }
    .stApp > header, .stApp > footer {
        background-color: rgba(0, 0, 0, 0); /* Transparent header and footer */
    }
    .stApp .main {
        background-color: rgba(0, 0, 0, 0.6); /* Dark transparent background */
        border-radius: 10px;
        padding: 2rem;
    }
    .sidebar .sidebar-content {
        background-color: rgba(0, 0, 0, 0.5); /* Darker transparent sidebar */
        border-radius: 10px;
    }
    .stButton button {
        font-size: 20px;
        padding: 10px 20px;
        background-color: #4CAF50; /* Green background */
        color: white; /* White text */
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #45a049; /* Darker green on hover */
    }
    .stNumberInput input, .stSelectbox select {
        color: #000000; /* Dark text for inputs */
        background-color: rgba(255, 255, 255, 0.9); /* Light background for inputs */
        border: 1px solid #CCCCCC; /* Light border */
        border-radius: 5px;
    }
    h1, h2, p, label {
        color: #f0f0f0;  /* Light text color for better readability */
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit app title and description
st.markdown("""
    <h1 style='font-size:40px;'>Traffic Volume Predictor</h1>
    <p style='font-size:20px;'>This application predicts traffic volume based on various weather and time-related features.</p>
    <p style='font-size:20px;'>Fill in the inputs below and click "Predict" to see the estimated traffic volume.</p>
""", unsafe_allow_html=True)

# Sidebar for user inputs
st.sidebar.header("Input Parameters")

# Numeric input fields
temp = st.sidebar.number_input("Temperature (K)", min_value=0.0, max_value=500.0, value=300.0, step=0.1)
rain_1h = st.sidebar.number_input("Rain in 1h (mm)", min_value=0.0, max_value=200.0, value=0.0, step=0.1)
snow_1h = st.sidebar.number_input("Snow in 1h (mm)", min_value=0.0, max_value=200.0, value=0.0, step=0.1)
clouds_all = st.sidebar.number_input("Clouds all (%)", min_value=0, max_value=100, value=0)
hour = st.sidebar.slider("Hour", min_value=0, max_value=23, value=12)
day_of_week = st.sidebar.slider("Day of Week (0=Monday, 6=Sunday)", min_value=0, max_value=6, value=0)

# Create dropdowns dynamically for each categorical column
dropdown_selections = {}
for col in categorical_columns:
    dropdown_selections[col] = st.sidebar.selectbox(f"Select {col}", unique_values[col])

# Display selected values in the main area
st.markdown("<h2 style='font-size:30px;'>Selected Input Values</h2>", unsafe_allow_html=True)
st.write({
    'Temperature (K)': temp,
    'Rain in 1h (mm)': rain_1h,
    'Snow in 1h (mm)': snow_1h,
    'Clouds all (%)': clouds_all,
    'Hour': hour,
    'Day of Week': day_of_week,
    **{col: dropdown_selections[col] for col in categorical_columns}
})

# Prepare input data for prediction
input_data = pd.DataFrame({
    'temp': [temp],
    'rain_1h': [rain_1h],
    'snow_1h': [snow_1h],
    'clouds_all': [clouds_all],
    'hour': [hour],
    'day_of_week': [day_of_week],
    **{col: [dropdown_selections[col]] for col in categorical_columns}  # Include categorical selections
})

# Prediction button
if st.button("Predict Traffic Volume"):
    try:
        # Make the prediction
        prediction = model.predict(input_data)
        st.success(f"**Predicted Traffic Volume:** {round(prediction[0], 2)} vehicles/hour")
    except Exception as e:
        st.error(f"Error in making prediction: {e}")

# Footer
st.markdown("""
    <p style='font-size:18px; color:white;'>*This model is powered by machine learning to provide real-time traffic volume predictions based on historical data.*</p>
""", unsafe_allow_html=True)
