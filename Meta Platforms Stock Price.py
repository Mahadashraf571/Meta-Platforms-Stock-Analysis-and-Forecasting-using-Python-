#!/usr/bin/env python
# coding: utf-8

# **Exploratory Data Analysis (EDA) and Visualization**

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
file_path = 'C:/Users/mahad/Downloads/archive/META.csv'
meta_stock_data = pd.read_csv(file_path)

# Convert Date column to datetime
meta_stock_data['Date'] = pd.to_datetime(meta_stock_data['Date'], format='%d/%m/%Y')

# Sort by Date
meta_stock_data.sort_values('Date', inplace=True)

# Display the first few rows
meta_stock_data.head()


# **Historical Trends and Patterns**

# In[4]:


plt.figure(figsize=(14, 7))
plt.plot(meta_stock_data['Date'], meta_stock_data['Close'], label='Closing Price')
plt.title('Meta Platforms Daily Closing Prices')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()


# **Moving Averages**

# In[5]:


# Calculate moving averages
meta_stock_data['MA50'] = meta_stock_data['Close'].rolling(window=50).mean()
meta_stock_data['MA200'] = meta_stock_data['Close'].rolling(window=200).mean()

# Plot closing prices and moving averages
plt.figure(figsize=(14, 7))
plt.plot(meta_stock_data['Date'], meta_stock_data['Close'], label='Closing Price')
plt.plot(meta_stock_data['Date'], meta_stock_data['MA50'], label='50-Day Moving Average')
plt.plot(meta_stock_data['Date'], meta_stock_data['MA200'], label='200-Day Moving Average')
plt.title('Meta Platforms Closing Prices and Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()


# **Distribution of Daily Returns and Volatility**

# In[6]:


# Calculate daily returns
meta_stock_data['Daily Return'] = meta_stock_data['Close'].pct_change()

# Plot histogram of daily returns
plt.figure(figsize=(14, 7))
plt.hist(meta_stock_data['Daily Return'].dropna(), bins=100, alpha=0.75)
plt.title('Distribution of Daily Returns')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.show()

# Calculate volatility (standard deviation of daily returns)
volatility = meta_stock_data['Daily Return'].std()
print(f'Volatility (Standard Deviation of Daily Returns): {volatility:.4f}')


# **Predictive Modeling**

# In[7]:


from sklearn.model_selection import train_test_split

# Create features and target
features = meta_stock_data[['Open', 'High', 'Low', 'Volume']]
target = meta_stock_data['Close']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)


# **Train and Evaluate Models**

# In[8]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Initialize models
linear_model = LinearRegression()
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train models
linear_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Make predictions
linear_preds = linear_model.predict(X_test)
rf_preds = rf_model.predict(X_test)

# Evaluate models
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

linear_mae, linear_mse, linear_rmse = evaluate_model(y_test, linear_preds)
rf_mae, rf_mse, rf_rmse = evaluate_model(y_test, rf_preds)

print(f'Linear Regression - MAE: {linear_mae:.2f}, MSE: {linear_mse:.2f}, RMSE: {linear_rmse:.2f}')
print(f'Random Forest - MAE: {rf_mae:.2f}, MSE: {rf_mse:.2f}, RMSE: {rf_rmse:.2f}')


# **Sentiment Analysis and Stock Price Correlation**

# In[11]:


pip install requests


# **Sentiment Analysis Using News Articles**

# In[10]:


import requests
import pandas as pd

# Replace with your NewsAPI key
api_key = '9c42e869bdb44562829f3403a036aab7'
query = 'Meta Platforms OR Facebook'
url = f'https://newsapi.org/v2/everything?q={query}&apiKey={api_key}'

response = requests.get(url)
news_data = response.json()

# Extract relevant data
articles = news_data['articles']
news_list = []

for article in articles:
    news_item = {
        'source': article['source']['name'],
        'author': article['author'],
        'title': article['title'],
        'description': article['description'],
        'url': article['url'],
        'publishedAt': article['publishedAt'],
        'content': article['content']
    }
    news_list.append(news_item)

# Create a DataFrame
news_df = pd.DataFrame(news_list)
news_df.head()


# **Volatility Forecasting**

# In[14]:


pip install arch


# In[15]:


from arch import arch_model

# Remove NaN values
returns = meta_stock_data['Daily Return'].dropna()

# Fit GARCH model
garch_model = arch_model(returns, vol='Garch', p=1, q=1)
garch_fit = garch_model.fit(disp='off')

# Forecast volatility
volatility_forecast = garch_fit.forecast(horizon=30)
volatility_forecast = volatility_forecast.variance[-1:]

# Display forecasted volatility
print(volatility_forecast)


# In[16]:


pip install numpy cvxopt


# **Portfolio Optimization**

# In[17]:


import numpy as np
import cvxopt as opt
from cvxopt import blas, solvers
import matplotlib.pyplot as plt

# Create a function to generate random portfolios
def generate_random_portfolios(num_portfolios, means, cov_matrix):
    results = np.zeros((3, num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(len(means))
        weights /= np.sum(weights)
        portfolio_return = np.sum(means * weights)
        portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        results[0,i] = portfolio_return
        results[1,i] = portfolio_stddev
        results[2,i] = results[0,i] / results[1,i]  # Sharpe ratio
        weights_record.append(weights)
    return results, weights_record

# Mean returns and covariance matrix for portfolio optimization
returns = meta_stock_data[['Close']].pct_change().mean()
cov_matrix = meta_stock_data[['Close']].pct_change().cov()

# Generate random portfolios
num_portfolios = 50000
results, weights_record = generate_random_portfolios(num_portfolios, returns, cov_matrix)

# Plot the efficient frontier
plt.figure(figsize=(14, 7))
plt.scatter(results[1,:], results[0,:], c=results[2,:], cmap='viridis', marker='o')
plt.colorbar(label='Sharpe ratio')
plt.title('Efficient Frontier')
plt.xlabel('Risk')
plt.ylabel('Return')
plt.show()


# In[ ]:




