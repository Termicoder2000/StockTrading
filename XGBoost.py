import pandas as pd
import numpy as np
from market_data import load_close
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import os

os.makedirs("images/ml_model", exist_ok=True)

ticker = "AAPL" 
start_date = '2020-01-01'
end_date = '2026-11-01'
prices = load_close(ticker, start_date, end_date)
df = pd.DataFrame({'Close': prices}).dropna()

# -------------------------------------------------------------
# 1. ADVANCED FEATURE ENGINEERING
# -------------------------------------------------------------
df['Return_1d'] = df['Close'].pct_change(1)
df['Return_2d'] = df['Close'].pct_change(2)
df['Return_3d'] = df['Close'].pct_change(3)
df['Volatility'] = df['Return_1d'].rolling(window=10).std()

# RSI (14 periods)
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# MACD
ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = ema_12 - ema_26
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

# -------------------------------------------------------------
# 2. MULTI-CLASS TARGET WITH MARGIN (DEAD ZONE)
# -------------------------------------------------------------
future_return = df['Close'].pct_change().shift(-1)
margin = 0.005  # 0.5% threshold margin

# Map classes for XGBoost: 
# XGBoost expects class labels starting from 0, so we map:
#  -1 (Short) -> 0
#   0 (Cash)  -> 1
#   1 (Long)  -> 2
df['Target'] = 1  # Default to Cash
df.loc[future_return > margin, 'Target'] = 2   # Long
df.loc[future_return < -margin, 'Target'] = 0  # Short

df = df.dropna()

features = ['Return_1d', 'Return_2d', 'Return_3d', 'Volatility', 'RSI', 'MACD', 'MACD_Hist']
X = df[features]
y = df['Target']

# Time-Series Split (Chronological to prevent look-ahead bias)
split_index = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# 3. Train XGBoost Classifier
print("Training XGBoost Classifier...")
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.05,
    random_state=42,
    eval_metric='mlogloss'
)
model.fit(X_train, y_train)

# 4. Predictions & Evaluation
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"\nTest Accuracy (XGBoost 3-Class): {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=['Short (0)', 'Cash (1)', 'Long (2)']))

# 5. Backtest with Transaction Costs & Position Mapping
df_test = df.iloc[split_index:].copy()
df_test['ML_Signal'] = predictions

# Remap XGBoost labels back to trading positions: 0 -> -1, 1 -> 0, 2 -> 1
signal_mapping = {0: -1, 1: 0, 2: 1}
df_test['Mapped_Position'] = df_test['ML_Signal'].map(signal_mapping)

# Shift position to avoid look-ahead bias
df_test['Position_Shifted'] = df_test['Mapped_Position'].shift(1).fillna(0)

cost_per_trade = 0.001  # 0.1% transaction cost per position change magnitude
df_test['Market_Return'] = df_test['Close'].pct_change()

# Strategy gross return: Position (-1, 0, or 1) * Market Return
df_test['Strategy_Gross'] = df_test['Position_Shifted'] * df_test['Market_Return']

# Transaction costs apply whenever the position changes
df_test['Trade_Changes'] = df_test['Position_Shifted'].diff().abs().fillna(0)
df_test['Transaction_Costs'] = df_test['Trade_Changes'] * cost_per_trade

# Net Strategy Return
df_test['Strategy_Net'] = df_test['Strategy_Gross'] - df_test['Transaction_Costs']

# Cumulative performance
df_test['Cumulative_Market'] = (1 + df_test['Market_Return'].dropna()).cumprod()
df_test['Cumulative_ML_Strategy'] = (1 + df_test['Strategy_Net'].dropna()).cumprod()

# 6. Plot Results
plt.figure(figsize=(12, 6))
plt.plot(df_test.index, df_test['Cumulative_Market'], label=f'Buy & Hold ({ticker})', color='gray', alpha=0.7)
plt.plot(df_test.index, df_test['Cumulative_ML_Strategy'], label=f'XGBoost Strategy (3-Class with {margin*100}% Margin)', color='darkorange', linewidth=2)
plt.title(f'XGBoost ML Trading Strategy - {ticker}')
plt.xlabel('Date')
plt.ylabel('Growth of $1')
plt.legend()
plt.grid(True)
plt.tight_layout()

plot_path = "images/ml_model/xgboost_performance.png"
plt.savefig(plot_path)
plt.close()
print(f"\nChart saved to '{plot_path}'!")