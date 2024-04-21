import pandas as pd
from sklearn.ensemble import IsolationForest
import plotly.express as px
from collections import Counter
import re
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'D:\\Work\\Development\\Machine learning Tasks\\ML3\\Queries.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(data.head())

# Check for null values
print(data.isnull().sum())

# Check column information
print(data.info())

# Descriptive statistics of the dataset
print(data.describe())

# Convert CTR from percentage string to float
data['CTR'] = data['CTR'].str.rstrip('%').astype(float) / 100.0

# Display the updated dataset
print(data.head())

def clean_and_split_queries(data):
    all_queries = ' '.join(data['Top queries'])
    words = re.findall(r'\w+', all_queries.lower())
    word_count = Counter(words)
    return word_count

word_frequencies = clean_and_split_queries(data)
print(word_frequencies.most_common(10))

freq_df = pd.DataFrame(word_frequencies.most_common(20), columns=['Word', 'Frequency'])
fig = px.bar(freq_df, x='Word', y='Frequency', title='Top 20 Common Words in Search Queries')
fig.show()

top_clicks = data.sort_values(by='Clicks', ascending=False).head(10)
top_impressions = data.sort_values(by='Impressions', ascending=False).head(10)
highest_ctrs = data.sort_values(by='CTR', ascending=False).head(10)
lowest_ctrs = data.sort_values(by='CTR', ascending=True).head(10)
print(top_clicks, top_impressions, highest_ctrs, lowest_ctrs)

correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Metrics')
plt.show()

iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
iso_forest.fit(data[['Clicks', 'Impressions', 'CTR', 'Position']])
data['anomaly'] = iso_forest.predict(data[['Clicks', 'Impressions', 'CTR', 'Position']])
anomalies = data[data['anomaly'] == -1]
print(anomalies)
