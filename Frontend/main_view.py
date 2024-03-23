import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests


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

    with st.spinner('Wait for it...'):
        st.image('images/ball.gif')
        time.sleep(5)
    st.success('Done!')

    if response.status_code == 200:
        # Display the result returned from FastAPI
        result = response.content
        st.success(f"Result from FastAPI: {result}")
    else:
        st.error('Failed to get response from FastAPI backend.')