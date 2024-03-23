import streamlit as st
import pandas as pd
import numpy as np

st.write("this App is not for commercial use")

loadfile = pd.read_csv("TestingData/compression_algorithms_test_dataset_10.csv")

st.plotly_chart(loadfile , use_container_width=True)

