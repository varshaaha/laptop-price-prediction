
import streamlit as st
import pickle 
import numpy as np
import pandas as pd 

# Load the model and dataframe
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Streamlit interface
st.title("Laptop Price Predictor")

company = st.selectbox('Brand', df['Company'].unique())
type = st.selectbox('Type', df['TypeName'].unique())
ram = st.selectbox('RAM(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
weight = st.number_input('Weight of the Laptop')
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
ips = st.selectbox('IPS', ['No', 'Yes'])
screen_size = st.number_input('Screen Size')
resolution = st.selectbox('Screen Resolution', [
    '1920x1080', '1366x768', '1600x900', '3840x2160', 
    '3200x1800', '2880x1800', '2560x1600', '2560x1440', 
    '2304x1440'
])
cpu = st.selectbox('CPU', df['Cpu brand'].unique())
hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])
gpu = st.selectbox('GPU', df['Gpu brand'].unique())
os = st.selectbox('OS', df['os'].unique())

if st.button('Predict Price'):
    try:
        # Convert inputs to appropriate formats
        touchscreen = 1 if touchscreen == 'Yes' else 0
        ips = 1 if ips == 'Yes' else 0
        X_res, Y_res = map(int, resolution.split('x'))
        
        # Calculate ppi
        if screen_size != 0:
            ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size
        else:
            st.error("Error: Screen size cannot be zero.")
            raise ValueError("Screen size cannot be zero.")
        
       
        data = {
            'Company': company,
            'TypeName': type,
            'Ram': ram,
            'Weight': weight,
            'Touchscreen': touchscreen,
            'Ips': ips,
            'ppi': ppi,
            'Cpu brand': cpu,
            'HDD': hdd,
            'SSD': ssd,
            'Gpu brand': gpu,
            'os': os
        }

        query_df = pd.DataFrame(data, index=[0])

        # Reshape query for prediction
    
        print(query_df)
        predicted_price = int(np.exp(pipe.predict(query_df)[0]))
        
        st.title(f"The predicted price of this configuration is { int(predicted_price)}")
    
    except ValueError as e:
        st.error(f"Error occurred: {e}")
    except Exception as e:
        st.error(f"Unexpected error occurred: {e}")

