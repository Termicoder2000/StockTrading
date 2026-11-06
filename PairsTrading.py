import matplotlib
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot to prevent GUI/Tkinter errors

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Your list of stock pairs
pairs_tickers_only = [
    ("BMW.DE", "MBG.DE"),
    ("KO", "PEP"),
    ("NESN.SW", "ULVR.L"),
    ("ALV.DE", "MUV2.DE"),
    ("JPM", "BAC"),
    ("NTRS", "STT"),
    ("XOM", "CVX"),
    ("ROG.SW", "NOVN.SW"),
    ("V", "MA")
]

# Create base directory for images
output_base_dir = "images"
os.makedirs(output_base_dir, exist_ok=True)

cost_per_trade = 0.001  # 0.1% transaction costs per position change
window = 30
TIME_LIMIT = 30  # Maximum days in trade

results = []

for t1, t2 in pairs_tickers_only:
    pair_name = f"{t1}_{t2}"
    pair_dir = os.path.join(output_base_dir, pair_name)
    os.makedirs(pair_dir, exist_ok=True)
    
    print(f"Processing and saving plots for: {pair_name} ...")
    
    try:
        # 1. Download data
        data = yf.download([t1, t2], start='2024-01-01', end='2026-11-01', progress=False)['Close']
        
        if isinstance(data.columns, pd.MultiIndex):
            s1 = data['Close'][t1]
            s2 = data['Close'][t2]
        else:
            s1 = data[t1]
            s2 = data[t2]
            
        df_prices = pd.DataFrame({t1: s1, t2: s2}).dropna()
        if len(df_prices) < 50:
            print(f"Skipped (not enough data): {pair_name}")
            continue
            
        # Normalization & Spread
        norm1 = (df_prices[t1] / df_prices[t1].iloc[0]) * 100
        norm2 = (df_prices[t2] / df_prices[t2].iloc[0]) * 100
        spread = norm1 - norm2
        
        r_mean = spread.rolling(window=window).mean()
        r_std = spread.rolling(window=window).std()
        zscore = (spread - r_mean) / r_std
        
        # Calculate daily spread return difference
        daily_diff_returns = df_prices[t1].pct_change() - df_prices[t2].pct_change()

        # -------------------------------------------------------------
        # STRATEGY 1: UNSECURED (Only Mean Reversion Exit)
        # Z-Score measures how many standard deviations the spread is from its mean.
        # Position flag (+1/-1/0) acts as a directional multiplier for spread returns.
        # -------------------------------------------------------------
        pos_unsecured = []
        cur_pos_uns = 0
        for z in zscore:
            if np.isnan(z):
                pos_unsecured.append(0)
                continue
            if cur_pos_uns == 0:
                if z > 2.0:
                    cur_pos_uns = -1
                elif z < -2.0:
                    cur_pos_uns = 1
            else:
                if -0.5 <= z <= 0.5:
                    cur_pos_uns = 0
            pos_unsecured.append(cur_pos_uns)

        sig_uns = pd.Series(pos_unsecured, index=zscore.index)
        strat_ret_uns = sig_uns.shift(1) * daily_diff_returns
        costs_uns = sig_uns.diff().abs().fillna(0) * cost_per_trade
        net_returns_uns = strat_ret_uns - costs_uns
        net_uns = (1 + net_returns_uns.dropna()).prod() - 1

        # -------------------------------------------------------------
        # STRATEGY 2: SECURED (Stop-Loss and Time-Exit)
        # - Stop-Loss: Exits if Z diverges beyond +/- 3.0 (prevents blowups)
        # - Time-Exit: Exits if trade exceeds TIME_LIMIT days without reverting
        # -------------------------------------------------------------
        pos_secured = []
        cur_pos_sec = 0
        days_in_trade = 0
        for z in zscore:
            if np.isnan(z):
                pos_secured.append(0)
                continue
            if cur_pos_sec == 0:
                if z > 2.0: 
                    cur_pos_sec = -1  # Short t1 / Long t2
                    days_in_trade = 0
                elif z < -2.0: 
                    cur_pos_sec = 1   # Long t1 / Short t2
                    days_in_trade = 0
            else:
                days_in_trade += 1
                hit_mean = (-0.5 <= z <= 0.5)
                hit_stop_loss = (abs(z) > 3.0)
                hit_time_limit = (days_in_trade >= TIME_LIMIT)
                
                if hit_mean or hit_stop_loss or hit_time_limit:
                    cur_pos_sec = 0
                    days_in_trade = 0
            pos_secured.append(cur_pos_sec)

        sig_sec = pd.Series(pos_secured, index=zscore.index)
        strat_ret_sec = sig_sec.shift(1) * daily_diff_returns
        costs_sec = sig_sec.diff().abs().fillna(0) * cost_per_trade
        net_returns_sec = strat_ret_sec - costs_sec
        net_sec = (1 + net_returns_sec.dropna()).prod() - 1

        if (t1, t2) == ("JPM", "BAC"):
            cumulative_uns = (1 + net_returns_uns.fillna(0)).cumprod() - 1
            cumulative_sec = (1 + net_returns_sec.fillna(0)).cumprod() - 1

            plt.figure(figsize=(12, 6))
            plt.plot(
                cumulative_uns.index,
                cumulative_uns * 100,
                label=f'Unsecured (Full Risk): {net_uns * 100:.2f}%',
                linewidth=2
            )
            plt.plot(
                cumulative_sec.index,
                cumulative_sec * 100,
                label=f'Secured (Stop-Loss + Time Limit): {net_sec * 100:.2f}%',
                linewidth=2
            )
            plt.axhline(0, color='black', linestyle='--', alpha=0.5)
            plt.title('JPM/BAC Pair Trading: Cumulative Net Return')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Return (%)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(pair_dir, 'strategy_performance.png'))
            plt.close()

        # Store performance comparison metrics
        results.append({
            'Pair': f"{t1} - {t2}",
            'Unsecured_Return_%': round(net_uns * 100, 2),
            'Secured_Return_%': round(net_sec * 100, 2)
        })

        # -------------------------------------------------------------
        # PLOT 1: Normalized Asset Prices
        # -------------------------------------------------------------
        plt.figure(figsize=(12, 6))
        plt.plot(df_prices.index, norm1, label=f'{t1} (Normalized)', color='blue', alpha=0.8)
        plt.plot(df_prices.index, norm2, label=f'{t2} (Normalized)', color='orange', alpha=0.8)
        
        plt.title(f'Normalized Prices: {t1} vs {t2}')
        plt.xlabel('Date')
        plt.ylabel('Normalized Price (Start = 100)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(pair_dir, 'prices_and_trades.png'))
        plt.close()

        # Helper function for plotting Z-score signals cleanly
        def save_zscore_plot(signal_series, title_suffix, filename):
            plt.figure(figsize=(12, 6))
            plt.plot(zscore.index, zscore, label='Z-Score', color='purple', alpha=0.7)
            plt.axhline(0, color='black', linestyle='--', alpha=0.5)
            plt.axhline(2.0, color='red', linestyle=':', label='Entry Threshold (+2 / -2)')
            plt.axhline(-2.0, color='red', linestyle=':')
            plt.axhline(3.0, color='darkred', linestyle='--', label='Stop-Loss (+3 / -3)')
            plt.axhline(-3.0, color='darkred', linestyle='--')

            # Filter entry trades
            entry_long = zscore[(signal_series == 1) & (signal_series.shift(1) == 0)]
            entry_short = zscore[(signal_series == -1) & (signal_series.shift(1) == 0)]

            plt.scatter(entry_long.index, entry_long, color='green', marker='^', s=120, label='Entry Long', zorder=5)
            plt.scatter(entry_short.index, entry_short, color='red', marker='v', s=120, label='Entry Short', zorder=5)

            plt.title(f'Z-Score & Signals ({title_suffix}): {t1} vs {t2}')
            plt.xlabel('Date')
            plt.ylabel('Z-Score')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(pair_dir, filename))
            plt.close()

        # -------------------------------------------------------------
        # PLOT 2: Z-Score Signals (Unsecured)
        # -------------------------------------------------------------
        save_zscore_plot(sig_uns, 'Unsecured Strategy', 'zscore_signals_unsecured.png')

        # -------------------------------------------------------------
        # PLOT 3: Z-Score Signals (Secured)
        # -------------------------------------------------------------
        save_zscore_plot(sig_sec, 'Secured Strategy', 'zscore_signals_secured.png')

    except Exception as e:
        print(f"Error for pair {t1}/{t2}: {e}")

# Save comparison results to CSV and print output
df_results = pd.DataFrame(results)
csv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csv")
os.makedirs(csv_dir, exist_ok=True)
csv_path = os.path.join(csv_dir, "backtest_results.csv")
df_results.to_csv(csv_path, index=False)

print("\n--- Backtest Results Comparison (Net of Transaction Costs) ---")
print(df_results.to_string(index=False))
print(f"\nResults table successfully saved to '{csv_path}'!")
print("All pair plots (including JPM/BAC strategy performance) successfully saved in the 'images/' folder!")