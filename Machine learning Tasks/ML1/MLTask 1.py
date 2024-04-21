import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

def load_data(file_path):
    df = pd.read_csv(file_path)
    print(df.head())
    return df

def check_null_values(df):
    null_counts = df.isnull().sum()
    print("Count of Null :", null_counts)

def show_info(df):
    df.info()

def show_descriptive_stats(df):
    print("Descriptive Stats:", df.describe())

def plot_distribution(df):
    sb.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    sb.histplot(data=df, x='Transaction_Amount', kde=False, bins=30, color='red', element="step", palette='viridis')
    plt.title('Distribution of Transaction Amount')
    plt.xlabel('Transaction amount')
    plt.ylabel("Frequency")
    plt.show()

def plot_distByAccount(df, column):
    filter_df = df[df['Account_Type'].isin(['Savings', 'Current'])]
    plt.figure(figsize=(10,6))
    sb.histplot(data=df, x=column, bins=30, hue='Account_Type', multiple='stack', kde=True, palette='Set1')
    plt.title('Distribution of Transactions Amount by Account Type')
    plt.xlabel('Transaction Amount')
    plt.ylabel('Frequency')
    plt.show()

def AveTransaction(df):
    avg_trans = df.groupby('Age')['Transaction_Amount'].mean().reset_index()
    plt.figure(figsize=(10,6))
    sb.scatterplot(data=df, x='Age', y='Transaction_Amount', hue=30, alpha=0.6)
    sb.lineplot(data=avg_trans, x='Age', y='Transaction_Amount', hue=30, markers=True, dashes=False)
    plt.title('Average Transaction Amount vs Age')
    plt.xlabel('Age')
    plt.ylabel('Transaction Amount')
    plt.show()

def TransByDay(df):
    trans_count = df[Day_of_Week].value_counts().sort_index()
    transactions_count_df = transactions_count.reset_index()
    transactions_count_df.columns = [Day_of_Week, 'Transaction_Count']
    ordered_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    transactions_count_df[Day_of_Week] = pd.Categorical(transactions_count_df[Day_of_Week], categories=ordered_days, ordered=True)
    transactions_count_df = transactions_count_df.sort_values(by=Day_of_Week)
    plt.figure(figsize=(10, 6))
    sb.barplot(data=transactions_count_df, x=Day_of_Week, y='Transaction_Count', color='skyblue')
    plt.title('Count of Transactions by Day of the Week')
    plt.xlabel('Day of the Week')
    plt.ylabel('Number of Transactions')
    plt.xticks(rotation=45)  # Rotate the labels if they overlap
    plt.show()

def plot_correlation_matrix(df):
    corr_matrix = df.corr()
    plt.figure(figsize=(12, 10))
    sb.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title('Correlation Matrix of All Columns')
    plt.show()

def visualize_anomalies(df, x_column, y_column, anomaly_column):

    plt.figure(figsize=(10, 6))
    # Plot normal points
    sb.scatterplot(data=df[df[anomaly_column]], x=x_column, y=y_column, color='blue', label='Normal', s=50)
    # Plot anomalies
    sb.scatterplot(data=df[df[anomaly_column]], x=x_column, y=y_column, color='red', label='Anomaly', s=100, marker='x')
    plt.title('Anomaly Visualization')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.legend()
    plt.show()

def calculate_anomaly_ratio(df, anomaly_column):
    total_points = len(df) 
    anomaly_count = df[anomaly_column].sum() 
    anomaly_ratio = anomaly_count / total_points
    print(f"Total data points: {total_points}")
    print(f"Number of anomalies: {anomaly_count}")
    print(f"Ratio of anomalies: {anomaly_ratio:.2f}")
    return anomaly_ratio

def isolationforest(df):
    features = ['Feature1', 'Feature2', 'FeatureN']
    X = df[features]
    model = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
    model.fit(X)
    df['anomaly_score'] = model.predict(X)
    df['Is_Anomaly'] = (df['anomaly_score'] == -1).astype(int)

def classificationreport(df):
    true_labels = df['True_Label']
    predicted_labels = df['Is_Anomaly']
    report = classification_report(true_labels, predicted_labels, target_names=['Normal', 'Anomaly'])
    print(report)

def anomalies(df):
    features = df.drop(columns=['True_Label'], errors='ignore')  
    model = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
    model.fit(features)
    predictions = model.predict(features)
    df['Is_Anomaly'] = (predictions == -1).astype(int)
    anomalies = df[df['Is_Anomaly'] == 1]
    print("Detected Anomalies:")
    print(anomalies)
    anomalies = df[df['Is_Anomaly'] == 1][['Feature1', 'Feature2', 'Is_Anomaly']]
    print("Anomalies with selected features:")
    print(anomalies)



# Replace with your actual file path
file_path = r'D:\Work\Development\Machine learning Tasks\ML1\transaction_anomalies_dataset.csv'

# Function calls
df = load_data(file_path)
check_null_values(df)
show_info(df)
show_descriptive_stats(df)
# plot_distribution(df)
# plot_distByAccount(df, 'Transaction_Amount')
AveTransaction(df)
TransByDay(df)
anomaly_ratio = calculate_anomaly_ratio(df, 'Is_Anomaly')