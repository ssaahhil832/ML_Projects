import argparse
import math
import os
from datetime import datetime

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, LSTM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an LSTM model to predict next-day close price."
    )
    parser.add_argument("--ticker", default="AAPL", help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--start", default="2015-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--interval", default="1d", help="Data interval, e.g. 1d")
    parser.add_argument("--lookback", type=int, default=60, help="Window size in days")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience")
    return parser.parse_args()


def download_data(ticker: str, start: str, end: str | None, interval: str) -> pd.DataFrame:
    data = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    if data.empty:
        raise ValueError("No data returned. Check ticker or date range.")
    data = data[["Close"]].dropna()
    return data


def create_sequences(values: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    x_list, y_list = [], []
    for i in range(lookback, len(values)):
        x_list.append(values[i - lookback : i])
        y_list.append(values[i])
    return np.array(x_list), np.array(y_list)


def build_model(input_shape: tuple[int, int]) -> Sequential:
    model = Sequential(
        [
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def main() -> None:
    args = parse_args()
    data = download_data(args.ticker, args.start, args.end, args.interval)

    values = data["Close"].values.reshape(-1, 1)
    split_index = int(len(values) * (1 - args.test_size))
    if split_index <= args.lookback:
        raise ValueError("Not enough data for the chosen lookback and test size.")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(values[:split_index])
    scaled = scaler.transform(values)

    x_all, y_all = create_sequences(scaled, args.lookback)
    train_end = split_index - args.lookback
    x_train, y_train = x_all[:train_end], y_all[:train_end]
    x_test, y_test = x_all[train_end:], y_all[train_end:]

    model = build_model((x_train.shape[1], x_train.shape[2]))
    callbacks = [
        EarlyStopping(patience=args.patience, restore_best_weights=True),
        ReduceLROnPlateau(patience=max(2, args.patience // 2), factor=0.5),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
        shuffle=False,
    )

    preds_scaled = model.predict(x_test, verbose=0)
    preds = scaler.inverse_transform(preds_scaled)
    y_test_inv = scaler.inverse_transform(y_test)

    rmse = math.sqrt(mean_squared_error(y_test_inv, preds))
    mae = mean_absolute_error(y_test_inv, preds)

    dates = data.index[args.lookback :]
    test_dates = dates[train_end:]

    os.makedirs("outputs", exist_ok=True)
    model.save(os.path.join("outputs", "lstm_model.keras"))
    joblib.dump(scaler, os.path.join("outputs", "scaler.pkl"))

    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, y_test_inv, label="Actual", linewidth=1.2)
    plt.plot(test_dates, preds, label="Predicted", linewidth=1.2)
    plt.title(f"{args.ticker} Actual vs Predicted Close")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "pred_vs_actual.png"))

    plt.figure(figsize=(10, 4))
    plt.plot(history.history["loss"], label="Train loss")
    plt.plot(history.history["val_loss"], label="Val loss")
    plt.title("Training history")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "training_loss.png"))

    last_window = scaled[-args.lookback :].reshape(1, args.lookback, 1)
    next_day_scaled = model.predict(last_window, verbose=0)
    next_day = scaler.inverse_transform(next_day_scaled)[0][0]

    last_date = data.index[-1].date()
    print(f"Ticker: {args.ticker}")
    print(f"Last available date: {last_date}")
    print(f"Next-day predicted close: {next_day:.2f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print("Saved outputs in ./outputs")


if __name__ == "__main__":
    main()
