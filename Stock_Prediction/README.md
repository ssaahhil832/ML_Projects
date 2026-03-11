# Stock Market Prediction with LSTM

This project trains an LSTM model on real market data and compares predicted vs actual prices.

## What it does
- Downloads real price data from Yahoo Finance
- Trains an LSTM to predict the next-day close
- Plots predicted vs actual close prices
- Saves the trained model and scaler

## Quick start
1. Create a virtual environment and install deps:
   - `pip install -r requirements.txt`

2. Train the model (default is AAPL, 5y+ daily data):
   - `python train_lstm.py`

3. Train on a specific asset and range:
   - `python train_lstm.py --ticker MSFT --start 2014-01-01 --end 2024-12-31 --lookback 60`

Outputs are saved in the `outputs` folder:
- `pred_vs_actual.png`
- `training_loss.png`
- `lstm_model.keras`
- `scaler.pkl`

## Notes
- This is not financial advice. Real markets are noisy and non-stationary.
- The model predicts the next-day close based on historical prices only.
- If you want stronger results, add more features (volume, indicators, macro data).
