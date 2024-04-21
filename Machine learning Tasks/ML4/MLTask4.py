import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = 'D:\\Work\\Development\\Machine learning Tasks\\ML4\\userbehaviour.csv'
user_data = pd.read_csv(file_path)

# Display basic data information
print("Data head and information:")
print(user_data.head())
print(user_data.info())
print(user_data.describe())

# Q.2: Screen time statistics
screen_time_stats = {
    "Highest Screen Time (hrs)": user_data['Average Screen Time'].max(),
    "Lowest Screen Time (hrs)": user_data['Average Screen Time'].min(),
    "Average Screen Time (hrs)": user_data['Average Screen Time'].mean()
}
print("Screen Time Stats:", screen_time_stats)

# Q.3: Spending statistics
spending_stats = {
    "Highest Spent on App (INR)": user_data['Average Spent on App (INR)'].max(),
    "Lowest Spent on App (INR)": user_data['Average Spent on App (INR)'].min(),
    "Average Spent on App (INR)": user_data['Average Spent on App (INR)'].mean()
}
print("Spending Stats:", spending_stats)

# Q.4: Relationship between spending capacity and screen time by status
plt.figure(figsize=(14, 7))
active_users = user_data[user_data['Status'] == 'Installed']
uninstalled_users = user_data[user_data['Status'] == 'Uninstalled']

# Active users plot
plt.subplot(1, 2, 1)
sns.scatterplot(x='Average Screen Time', y='Average Spent on App (INR)', data=active_users, color='blue', label='Active Users')
plt.title('Active Users: Screen Time vs Spending')
plt.xlabel('Average Screen Time (hrs)')
plt.ylabel('Average Spent on App (INR)')

# Uninstalled users plot
plt.subplot(1, 2, 2)
sns.scatterplot(x='Average Screen Time', y='Average Spent on App (INR)', data=uninstalled_users, color='red', label='Uninstalled Users')
plt.title('Uninstalled Users: Screen Time vs Spending')
plt.xlabel('Average Screen Time (hrs)')
plt.ylabel('Average Spent on App (INR)')
plt.tight_layout()
plt.show()

# Q.5: Relationship between ratings and screen time
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Ratings', y='Average Screen Time', data=user_data, color='green')
plt.title('Relationship Between Ratings and Screen Time')
plt.xlabel('Ratings')
plt.ylabel('Average Screen Time (hrs)')
plt.show()

# Q.6: Cluster analysis
features = user_data[['Average Screen Time', 'Average Spent on App (INR)', 'Ratings', 'New Password Request', 'Last Visited Minutes']]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(features_scaled)
user_data['Cluster'] = clusters
cluster_counts = user_data['Cluster'].value_counts()
print("Cluster Counts:", cluster_counts)

# Q.7: Visualize user segmentation
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Average Screen Time', y='Average Spent on App (INR)', hue='Cluster', palette='viridis', data=user_data, s=50)
plt.title('User Segmentation: Screen Time vs Spending')
plt.xlabel('Average Screen Time (hrs)')
plt.ylabel('Average Spent on App (INR)')
plt.legend(title='Cluster')
plt.show()
