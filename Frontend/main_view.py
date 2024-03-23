import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import json
import matplotlib.pyplot as plt


st.write("this App is not for commercial use")

df = pd.read_csv("TestingData/compression_algorithms_test_dataset_10.csv")

metric = st.selectbox('Select a metric to visualize',
                      ['Compression Ratio', 'Decoding Time(s)', 'Decompression Accuracy(%)',
                       'Encoding Time(s)', 'Entropy', 'Memory Usage for Processing(MB)',
                       'Redundancy Reduction', 'Space Saving(%)'])

# Generate the interactive chart
fig = px.bar(df, x='Algorithm', y=metric, color='Algorithm',
             labels={'Algorithm': 'Compression Algorithm'},
             title=f'Compression Algorithm {metric}')

# Display the chart
st.plotly_chart(fig)

#sideBar

with st.sidebar:
     st.write("Make a Prediction based on the metrcis")
     add_Entropy = st.number_input("Entopy",value=1.2339821616590991)
     Original_size = st.number_input("Original Size(bits)", value=384208174.87159073)
     compressed_size = st.number_input("Compressed Size(bits)", value=46418303.3892862)
     compression_ratio = st.number_input("Compression Ratio", value=1.2081550283109528)


#sennd metrcis via Api

if st.button('Predict'):
    url = 'http://127.0.0.1:8000/predict/'
    data = {"feature": [Original_size, compressed_size, compression_ratio, add_Entropy]}
    #send data to FastAPI Backend
    response = requests.post(url , json=data)


    with st.status("make the prediction....", expanded=True) as status:
        st.write("connecting to the model")
        time.sleep(2)
        st.write("Sending data...")
        time.sleep(2)
        st.write("Analyse...")
        time.sleep(3)
    st.success('Done!')

    if response.status_code == 200:
        # Display the result returned from FastAPI
        result = response.content
        st.write(result)
        # Step 1: Decode the byte string
        str_data = result.decode('utf-8')

        # Step 2: Parse the JSON string
        data = json.loads(str_data)

        # Extracting prediction values
        prediction = data['prediction'][0]  # Assuming there's only one prediction

        # Creating a DataFrame to hold the data
        df = pd.DataFrame([prediction], columns=['X', 'Y'])

        # Displaying the data using Streamlit
        st.write("Displaying Prediction Data:")
        st.dataframe(df)

        # Plotting the data
        fig, ax = plt.subplots()
        ax.scatter(df['X'], df['Y'])
        ax.set_xlabel('X Value')
        ax.set_ylabel('Y Value')
        ax.set_title('Prediction Data Plot')

        # Displaying the plot in Streamlit
        st.pyplot(fig)

    else:
        st.error('Failed to get response from FastAPI backend.')