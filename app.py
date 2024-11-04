from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/optimize', methods=['POST'])
def optimize_portfolio():
    # Get JSON data from the POST request
    data = request.json
    tickers = data['tickers']  # List of tickers from the user
    weights = np.array(data['weights'])  # Portfolio weights

    # Get start date and end date from the user input
    start_date = data['start_date']  # Start date in 'MM/DD/YYYY' format
    end_date = data['end_date']  # End date in 'MM/DD/YYYY' format

    # Convert start and end dates to the required format ('YYYY-MM-DD')
    start_date = datetime.strptime(start_date, "%m/%d/%Y").strftime("%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%m/%d/%Y").strftime("%Y-%m-%d")

    adj_close_df = pd.DataFrame()

    for ticker in tickers:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        adj_close_df[ticker] = stock_data['Adj Close']

    # Calculate log returns and covariance matrix
    log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()
    cov_matrix = log_returns.cov() * 252

    # Normalize weights (in case they don't sum to 1)
    weights = weights / np.sum(weights)

    # Define portfolio functions
    def standard_deviation(weights, cov_matrix):
        variance = weights.T @ cov_matrix @ weights
        return np.sqrt(variance)

    def expected_return(weights, log_returns):
        return np.sum(log_returns.mean() * weights) * 252

    # Risk-free rate example (10-year Treasury)
    risk_free_rate = 0.03

    def sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
        return (expected_return(weights, log_returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)

    # Calculate portfolio metrics
    portfolio_return = expected_return(weights, log_returns)
    portfolio_risk = standard_deviation(weights, cov_matrix)
    portfolio_sharpe = sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)

    # Return results as JSON
    return jsonify({
        'expected_return': portfolio_return,
        'risk': portfolio_risk,
        'sharpe_ratio': portfolio_sharpe
    })

if __name__ == '__main__':
    app.run()