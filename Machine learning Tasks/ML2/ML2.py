import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Data Loading
file_path = 'D:\Work\Development\Machine learning Tasks\ML2\tips.csv'  
tips_data = pd.read_csv(file_path)

# Data Checking
print("Data Info:")
print(tips_data.info())
print("Descriptive Statistics:")
print(tips_data.describe())
print("Null Values:")
print(tips_data.isnull().sum())

# Question 1: Analysis of tips data characteristics
# This includes visualizing the relationship between total bill, size, and tip amount by day

def plot_relationships():
    plt.figure(figsize=(14, 5))
    # Total bill vs Tip
    plt.subplot(1, 3, 1)
    sns.scatterplot(data=tips_data, x='total_bill', y='tip')
    plt.title('Tip vs Total Bill')

    # Size vs Tip
    plt.subplot(1, 3, 2)
    sns.scatterplot(data=tips_data, x='size', y='tip')
    plt.title('Tip vs Size')

    # Day vs Tip
    plt.subplot(1, 3, 3)
    sns.boxplot(data=tips_data, x='day', y='tip')
    plt.title('Tip by Day of the Week')
    plt.tight_layout()
    plt.show()

plot_relationships()

# Question 2: Tips by gender and day
def tips_by_gender_day():
    plt.figure(figsize=(14, 5))
    # Total bill vs Tip by Gender
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=tips_data, x='total_bill', y='tip', hue='sex')
    plt.title('Tip vs Total Bill by Gender')

    # Size vs Tip by Gender
    plt.subplot(1, 2, 2)
    sns.boxplot(data=tips_data, x='size', y='tip', hue='sex')
    plt.title('Tip by Size and Gender')
    plt.tight_layout()
    plt.show()

tips_by_gender_day()

# Question 3: Tips by time of meal
def tips_by_meal_time():
    plt.figure(figsize=(14, 5))
    # Total bill vs Tip by Meal Time
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=tips_data, x='total_bill', y='tip', hue='time')
    plt.title('Tip vs Total Bill by Meal Time')

    # Size vs Tip by Meal Time
    plt.subplot(1, 2, 2)
    sns.boxplot(data=tips_data, x='size', y='tip', hue='time')
    plt.title('Tip by Size and Meal Time')
    plt.tight_layout()
    plt.show()

tips_by_meal_time()

# Model Training for Predicting Tips
def train_tip_predictor():
    # Encoding categorical variables
    tips_data['sex'] = tips_data['sex'].astype('category').cat.codes
    tips_data['smoker'] = tips_data['smoker'].astype('category').cat.codes
    tips_data['day'] = tips_data['day'].astype('category').cat.codes
    tips_data['time'] = tips_data['time'].astype('category').cat.codes

    # Preparing the dataset for training
    X = tips_data.drop('tip', axis=1)
    y = tips_data['tip']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Model Evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error:", mse)
    print("R² Score:", r2)

    # Sample prediction
    sample_input = X_test.iloc[0]
    predicted_tip = model.predict([sample_input])
    print("Predicted Tip for sample input:", predicted_tip)

train_tip_predictor()
