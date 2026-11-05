import matplotlib
matplotlib.use('Agg')  # Prevents GUI/Tkinter errors when saving plots

import pandas as pd
import numpy as np
from market_data import load_close
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os

# Create base output directory for images
output_base_dir = "images/ml_models"
os.makedirs(output_base_dir, exist_ok=True)

ticker = "AAPL"
start_date = '2020-01-01'
end_date = '2026-11-01'
prices = load_close(ticker, start_date, end_date)
df = pd.DataFrame({'Close': prices}).dropna()

# -------------------------------------------------------------
# 1. FEATURE ENGINEERING (RSI, MACD, Returns, Volatility)
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

# Map classes for classifiers (0: Short, 1: Cash, 2: Long)
df['Target'] = 1  
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

# -------------------------------------------------------------
# 3. DEFINE MODELS TO TEST
# -------------------------------------------------------------
models = {
    "Random_Forest": RandomForestClassifier(n_estimators=100, min_samples_split=50, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.05, random_state=42, eval_metric='mlogloss'),
    "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=42)
}

comparison_results = []
cost_per_trade = 0.001  # 0.1% transaction cost
signal_mapping = {0: -1, 1: 0, 2: 1}  # Map model output back to positions

# Test dataset framework for backtesting
df_test = df.iloc[split_index:].copy()
df_test['Market_Return'] = df_test['Close'].pct_change()

cumulative_market = (1 + df_test['Market_Return']).cumprod()

# -------------------------------------------------------------
# 4. TRAINING, EVALUATION & BACKTESTING LOOP
# -------------------------------------------------------------
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    
    # Predict
    predictions = model.predict(X_test.loc[df_test.index])
    y_true = y_test.loc[df_test.index]
    
    # Classification Metrics (using 'macro' average for multi-class balance)
    acc = accuracy_score(y_true, predictions)
    prec = precision_score(y_true, predictions, average='macro', zero_division=0)
    rec = recall_score(y_true, predictions, average='macro', zero_division=0)
    f1 = f1_score(y_true, predictions, average='macro', zero_division=0)
    
    # Backtest calculations
    df_test[f'{name}_Signal'] = predictions
    df_test[f'{name}_Position'] = df_test[f'{name}_Signal'].map(signal_mapping)
    df_test[f'{name}_Position_Shifted'] = df_test[f'{name}_Position'].shift(1).fillna(0)
    
    # Returns & Costs
    gross_ret = df_test[f'{name}_Position_Shifted'] * df_test['Market_Return']
    position_change = df_test[f'{name}_Position_Shifted'].diff().fillna(0)
    trade_changes = position_change.abs()
    trans_costs = trade_changes * cost_per_trade
    net_ret = gross_ret - trans_costs
    executed_actions = pd.Series('Hold', index=df_test.index)
    executed_actions.loc[position_change > 0] = 'Buy'
    executed_actions.loc[position_change < 0] = 'Sell'
    
    cum_strategy = (1 + net_ret).cumprod()
    total_strat_return = (cum_strategy.iloc[-1] - 1) * 100
    
    # Save results for DataFrame
    comparison_results.append({
        'Model': name,
        'Accuracy_%': round(acc * 100, 2),
        'Precision_%': round(prec * 100, 2),
        'Recall_%': round(rec * 100, 2),
        'F1_Score_%': round(f1 * 100, 2),
        'Total_Return_%': round(total_strat_return, 2)
    })
    
    # Plot individual model performance
    plt.figure(figsize=(12, 6))
    plt.plot(df_test.index, cumulative_market, label=f'Buy & Hold ({ticker})', color='gray', alpha=0.7)
    plt.plot(df_test.index, cum_strategy, label=f'{name} Strategy (Net)', linewidth=2)
    action_styles = {
        'Buy': ('green', '^'),
        'Sell': ('red', 'v'),
        'Hold': ('darkorange', 'o')
    }
    for action, (color, marker) in action_styles.items():
        action_mask = executed_actions == action
        plt.scatter(
            df_test.index[action_mask],
            cum_strategy.loc[action_mask],
            label=action,
            color=color,
            marker=marker,
            s=24 if action != 'Hold' else 12,
            alpha=0.8 if action != 'Hold' else 0.35,
            zorder=3
        )
    plt.title(f'ML Strategy Performance: {name} - {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Growth of $1')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plot_path = os.path.join(output_base_dir, f"{name.lower()}_performance.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot to '{plot_path}'")

# -------------------------------------------------------------
# 5. DATAFRAME EXPORT & DISPLAY
# -------------------------------------------------------------
df_scores = pd.DataFrame(comparison_results)
csv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csv")
os.makedirs(csv_dir, exist_ok=True)
csv_filename = os.path.join(csv_dir, "model_comparison_results.csv")
df_scores.to_csv(csv_filename, index=False)

print("\n--- Comprehensive Model Comparison Results ---")
print(df_scores.to_string(index=False))
print(f"\nResults dataframe successfully saved to '{csv_filename}'!")
print(f"All individual model plots saved in the '{output_base_dir}/' folder!")