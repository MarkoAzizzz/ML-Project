import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model=joblib.load(open("boxer",'rb')) 

# Prediction function
def predict(features):
    prediction = model.predict(features)
    return prediction

# Page configuration
st.set_page_config(
    page_title='Boxing Match Outcome Prediction',
    page_icon='🥊',
    initial_sidebar_state='collapsed'
)

# Home page
st.write('# Boxing Match Outcome Prediction')
st.subheader('Enter boxing match statistics for outcome prediction')

# User inputs
punch_power_diff = st.number_input("Enter difference in estimated punch power:", min_value=-1000.000000, format="%.6f")
punch_resistance_diff = st.number_input("Enter difference in estimated punch resistance:", min_value=-1000.000000, format="%.6f")
ability_to_take_punch_diff = st.number_input("Enter difference in estimated ability to take punch:", min_value=-1000.000000, format="%.6f")
rounds_boxed_diff = st.number_input("Enter difference in rounds boxed:", min_value=-1000.000000, format="%.6f")
round_ko_percentage_diff = st.number_input("Enter difference in round KO percentage:", min_value=-1000.000000, format="%.6f")
has_been_ko_percentage_diff = st.number_input("Enter difference in has-been-KO percentage:", min_value=-1000.000000, format="%.6f")
avg_weight_diff = st.number_input("Enter difference in average weight:", min_value=-1000.000000, format="%.6f")
punch_power_ratio = st.number_input("Enter ratio of estimated punch power:", min_value=-1000.000000, format="%.6f")
punch_resistance_ratio = st.number_input("Enter ratio of estimated punch resistance:", min_value=-1000.000000, format="%.6f")
ability_to_take_punch_ratio = st.number_input("Enter ratio of estimated ability to take punch:", min_value=-1000.000000, format="%.6f")
rounds_boxed_ratio = st.number_input("Enter ratio of rounds boxed:", min_value=-1000.000000, format="%.6f")
round_ko_percentage_ratio = st.number_input("Enter ratio of round KO percentage:", min_value=-1000.000000, format="%.6f")
has_been_ko_percentage_ratio = st.number_input("Enter ratio of has-been-KO percentage:", min_value=-1000.000000, format="%.6f")
avg_weight_ratio = st.number_input("Enter ratio of average weight:", min_value=-1000.000000, format="%.6f")
punch_power_x_punch_resistance_1 = st.number_input("Enter product of estimated punch power and punch resistance for opponent 1:", min_value=-1000.000000, format="%.6f")
punch_power_x_punch_resistance_2 = st.number_input("Enter product of estimated punch power and punch resistance for opponent 2:", min_value=-1000.000000, format="%.6f")


# Concatenate features
features = np.array([punch_power_diff, punch_resistance_diff, ability_to_take_punch_diff,
                     rounds_boxed_diff, round_ko_percentage_diff, has_been_ko_percentage_diff,
                     avg_weight_diff, punch_power_ratio, punch_resistance_ratio,
                     ability_to_take_punch_ratio, rounds_boxed_ratio, round_ko_percentage_ratio,
                     has_been_ko_percentage_ratio, avg_weight_ratio, punch_power_x_punch_resistance_1,
                     punch_power_x_punch_resistance_2]).reshape(1, -1)

# Prediction
if st.button("Predict"):
    result = predict(features)
    st.write(f'The predicted outcome of the boxing match is: *{result[0]}*')

# About page
st.write("# About")
st.write("This app predicts the outcome of a boxing match based on the provided match statistics.")

# Contact page
st.write("# Contact")
st.write("For inquiries, please contact us at contact@example.com")

# Read the fighters dataset from CSV
fighters_df = pd.read_csv('fighters.csv')

# Read the popular matches dataset from CSV
matches_df = pd.read_csv('popular_matches.csv')


