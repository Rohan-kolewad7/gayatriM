import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle

# Load the ML model
with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define function to preprocess user input
def preprocess_input(ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity):
    input_data = {
        'ph': [float(ph)],
        'Hardness': [float(hardness)],
        'Solids': [float(solids)],
        'Chloramines': [float(chloramines)],
        'Sulfate': [float(sulfate)],
        'Conductivity': [float(conductivity)],
        'Organic_carbon': [float(organic_carbon)],
        'Trihalomethanes': [float(trihalomethanes)],
        'Turbidity': [float(turbidity)]
    }
    input_df = pd.DataFrame(input_data)
    return input_df

# Define function to make prediction and display result
def predict_potability(input_df):
    prediction = model.predict(input_df)
    return prediction[0]

# Define function to plot graph for a single parameter
def plot_single_parameter_graph(data, parameter, acceptable_range, ax):
    sns.barplot(data=data, x='Feature', y='Value', ax=ax)
    ax.set_title('Input Data - {}'.format(parameter))
    ax.set_xlabel('Feature')
    ax.set_ylabel('Value')
    ax.tick_params(axis='x', rotation=45)

    # Highlight the parameter value compared to the acceptable range
    parameter_value = data.loc[data['Feature'] == parameter.capitalize(), 'Value'].values[0]
    ax.axhline(y=acceptable_range[0], color='r', linestyle='--', label='Acceptable Range')
    ax.axhline(y=acceptable_range[1], color='r', linestyle='--')
    ax.axhline(y=parameter_value, color='b', linestyle='-', label='User {} Level'.format(parameter.capitalize()))
    ax.legend()

# Set Streamlit app configuration
st.set_page_config(layout='wide')

# Add title and description
st.title('Water Potability Prediction')
st.markdown('Enter the values for various features to predict the potability of water.')

# Add input boxes for user input
col1, col2, col3, col4 = st.columns(4)
with col1:
    ph = st.number_input('pH', min_value=0.0, max_value=14.0, step=0.1)
with col2:
    hardness = st.number_input('Hardness', min_value=0.0)
with col3:
    solids = st.number_input('Solids', min_value=0.0)
with col4:
    chloramines = st.number_input('Chloramines', min_value=0.0)

col5, col6, col7, col8 = st.columns(4)
with col5:
    sulfate = st.number_input('Sulfate', min_value=0.0)
with col6:
    conductivity = st.number_input('Conductivity', min_value=0.0)
with col7:
    organic_carbon = st.number_input('Organic Carbon', min_value=0.0)
with col8:
    trihalomethanes = st.number_input('Trihalomethanes', min_value=0.0)

turbidity = st.number_input('Turbidity', min_value=0.0)

# Add submit button
if st.button('Submit'):
    # Preprocess user input
    input_df = preprocess_input(ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity)
    
    # Display prediction
    prediction = predict_potability(input_df)
    if prediction == 0:
        st.subheader('Water Potability Prediction:')
        st.write('Water is not drinkable')
        
        # Define the acceptable range for each parameter
        acceptable_ranges = {
            'ph': (6.5, 8.5),
            'Solids': (0.0, 500.0),
            'Turbidity': (0.0, 10.0),
            'Chloramines': (0.0, 4.0)
        }
        
        # Display solution and information about parameters
        st.subheader('Solution:')
        st.markdown('Based on the input data, the water is predicted to be not potable. Below are some potential issues:')
        st.markdown('- pH level outside the acceptable range (6.5 - 8.5)')
        st.markdown('- Solids concentration above the acceptable limit (0 - 500)')
        st.markdown('- Turbidity level outside the acceptable range (0 - 10)')
        st.markdown('- Chloramines concentration above the acceptable limit (0 - 4)')
        
        # Create a column layout for the graphs
        col1, col2, col3, col4 = st.columns(4)

        # Plot the first parameter graph (pH)
        parameter1 = 'ph'
        graph_data1 = pd.DataFrame({
            'Feature': [parameter1.capitalize()],
            'Value': [input_df[parameter1][0]]
        })
        acceptable_range1 = acceptable_ranges[parameter1]
        with col1:
            fig, ax = plt.subplots(figsize=(6, 6))
            plot_single_parameter_graph(graph_data1, parameter1.capitalize(), acceptable_range1, ax)
            st.pyplot(fig)

        # Plot the second parameter graph (Solids)
        parameter2 = 'Solids'
        graph_data2 = pd.DataFrame({
            'Feature': [parameter2.capitalize()],
            'Value': [input_df[parameter2][0]]
        })
        acceptable_range2 = acceptable_ranges[parameter2]
        with col2:
            fig, ax = plt.subplots(figsize=(6, 6))
            plot_single_parameter_graph(graph_data2, parameter2.capitalize(), acceptable_range2, ax)
            st.pyplot(fig)

        # Plot the third parameter graph (Turbidity)
        parameter3 = 'Turbidity'
        graph_data3 = pd.DataFrame({
            'Feature': [parameter3.capitalize()],
            'Value': [input_df[parameter3][0]]
        })
        acceptable_range3 = acceptable_ranges[parameter3]
        with col3:
            fig, ax = plt.subplots(figsize=(6, 6))
            plot_single_parameter_graph(graph_data3, parameter3.capitalize(), acceptable_range3, ax)
            st.pyplot(fig)

        # Plot the fourth parameter graph (Chloramines)
        parameter4 = 'Chloramines'
        graph_data4 = pd.DataFrame({
            'Feature': [parameter4.capitalize()],
            'Value': [input_df[parameter4][0]]
        })
        acceptable_range4 = acceptable_ranges[parameter4]
        with col4:
            fig, ax = plt.subplots(figsize=(6, 6))
            plot_single_parameter_graph(graph_data4, parameter4.capitalize(), acceptable_range4, ax)
            st.pyplot(fig)
    else:
        st.subheader('Water Potability Prediction:')
        st.write('Water is drinkable')
