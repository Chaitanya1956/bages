from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/predict', methods=['GET'])
def predict():
    try:
        ticker = request.args.get('ticker', 'AAPL').upper()
        years = float(request.args.get('years', 2))
        
        print(f"Predicting for {ticker} with {years} years of data...")
        
        # Fetch data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=years*365)).strftime('%Y-%m-%d')
        
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if len(data) < 60:
            return jsonify({'error': 'Not enough data'}), 400
        
        # Create features
        data['SMA_20'] = data['Close'].rolling(20).mean()
        data['SMA_50'] = data['Close'].rolling(50).mean()
        data['RSI'] = calculate_rsi(data['Close'])
        data['Momentum'] = data['Close'].pct_change(5)
        
        # Drop NaN values
        data = data.dropna()
        
        if len(data) < 10:
            return jsonify({'error': 'Not enough data after cleaning'}), 400
        
        # Prepare features and target
        features = data[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 'Momentum']]
        target = data['Close'].shift(-1)  # Predict next day
        
        # Remove last row (no target)
        features = features[:-1]
        target = target[:-1]
        
        # Train model
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(features, target)
        
        # Predict
        last_features = features.iloc[-1:].values
        predicted_price = model.predict(last_features)[0]
        current_price = float(data['Close'].iloc[-1])
        change_pct = (predicted_price - current_price) / current_price * 100
        
        print(f"Prediction successful: {current_price} -> {predicted_price}")
        
        # Convert to JSON-serializable format
        dates_list = [date.strftime('%Y-%m-%d') for date in data.index]
        prices_list = [float(price) for price in data['Close'].values]
        
        return jsonify({
            'success': True,
            'currentPrice': current_price,
            'predictedPrice': float(predicted_price),
            'changePct': float(change_pct),
            'dates': dates_list,
            'prices': prices_list,
            'ticker': ticker
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e), 'success': False}), 500

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')