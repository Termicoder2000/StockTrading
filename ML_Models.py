import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from market_data import load_close


TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "BAC", "XOM", "CVX"]
PAIRS = [("JPM", "BAC"), ("XOM", "CVX")]
START_DATE = "2020-01-01"
END_DATE = "2026-11-01"
TRAIN_FRACTION = 0.8
HORIZON_DAYS = 5
EXECUTION_LAG_DAYS = 1
REBALANCE_EVERY_DAYS = 5
TOP_BOTTOM_FRACTION = 0.2
WALK_FORWARD_SPLITS = 3
COST_PER_SIDE = 0.001
SLIPPAGE_PER_SIDE = 0.0005
SHORT_BORROW_RATE_ANNUAL = 0.03
MIN_EDGE_CANDIDATES = (
    2 * (COST_PER_SIDE + SLIPPAGE_PER_SIDE)
    + SHORT_BORROW_RATE_ANNUAL * HORIZON_DAYS / 252,
    0.005,
    0.0075,
    0.01,
)
BUDGET_EDGE_CANDIDATES = (0.0, 0.001, 0.0025, MIN_EDGE_CANDIDATES[0], 0.005)
STARTING_CAPITAL = 10_000.0
MIN_HOLDINGS = 3
MAX_HOLDINGS = len(TICKERS)
MAX_POSITION_WEIGHT = 0.2
FEATURES = [
    "Return_1d",
    "Return_2d",
    "Return_5d",
    "Return_10d",
    "Return_20d",
    "Volatility_5d",
    "Volatility_20d",
    "RSI_14",
    "MACD_Hist",
    "SPY_Return_1d",
    "QQQ_Return_1d",
    "Market_Volatility_20d",
    "Peer_Return_1d",
    "Pair_ZScore",
    "Pair_Return_Difference",
    "Pair_Context_Available",
]
TARGET = "Forward_Excess_Return_5d"


def build_feature_table(
    prices: pd.DataFrame, tickers: list[str] = TICKERS
) -> pd.DataFrame:
    """Build point-in-time features and five-session relative-return labels."""
    returns = prices[tickers].pct_change(fill_method=None)
    entry_prices = prices[tickers].shift(-EXECUTION_LAG_DAYS)
    exit_prices = prices[tickers].shift(-(EXECUTION_LAG_DAYS + HORIZON_DAYS))
    forward_horizon_returns = exit_prices.div(entry_prices) - 1
    forward_universe_return = forward_horizon_returns.mean(axis=1)
    spy_returns = prices["SPY"].pct_change(fill_method=None)
    qqq_returns = prices["QQQ"].pct_change(fill_method=None)
    rows = []

    pair_context = {}
    for first, second in PAIRS:
        if first not in tickers or second not in tickers:
            continue
        log_spread = np.log(prices[first]) - np.log(prices[second])
        rolling_mean = log_spread.rolling(60, min_periods=40).mean()
        rolling_std = log_spread.rolling(60, min_periods=40).std()
        spread_zscore = (log_spread - rolling_mean) / rolling_std
        spread_return = returns[first] - returns[second]
        pair_context[first] = (spread_zscore, spread_return)
        pair_context[second] = (-spread_zscore, -spread_return)

    for ticker in tickers:
        close = prices[ticker]
        asset_returns = returns[ticker]
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        macd_hist = (macd - macd.ewm(span=9, adjust=False).mean()) / close
        delta = close.diff()
        gains = delta.clip(lower=0).rolling(14).mean()
        losses = -delta.clip(upper=0).rolling(14).mean()
        rsi = 100 - 100 / (1 + gains / losses)
        peer_returns = returns.drop(columns=ticker).mean(axis=1)

        if ticker in pair_context:
            pair_zscore, pair_return_difference = pair_context[ticker]
            pair_available = 1.0
        else:
            pair_zscore = pd.Series(0.0, index=prices.index)
            pair_return_difference = pd.Series(0.0, index=prices.index)
            pair_available = 0.0

        features = pd.DataFrame(
            {
                "Return_1d": asset_returns,
                "Return_2d": close.pct_change(2, fill_method=None),
                "Return_5d": close.pct_change(5, fill_method=None),
                "Return_10d": close.pct_change(10, fill_method=None),
                "Return_20d": close.pct_change(20, fill_method=None),
                "Volatility_5d": asset_returns.rolling(5).std(),
                "Volatility_20d": asset_returns.rolling(20).std(),
                "RSI_14": rsi,
                "MACD_Hist": macd_hist,
                "SPY_Return_1d": spy_returns,
                "QQQ_Return_1d": qqq_returns,
                "Market_Volatility_20d": spy_returns.rolling(20).std(),
                "Peer_Return_1d": peer_returns,
                "Pair_ZScore": pair_zscore,
                "Pair_Return_Difference": pair_return_difference,
                "Pair_Context_Available": pair_available,
                "Forward_Return_1d": asset_returns.shift(-1),
                TARGET: forward_horizon_returns[ticker] - forward_universe_return,
            },
            index=prices.index,
        )
        features.insert(0, "Date", prices.index)
        features.insert(1, "Ticker", ticker)
        rows.append(features)

    dataset = pd.concat(rows, ignore_index=True)
    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    dataset = dataset.dropna(subset=FEATURES + ["Forward_Return_1d"])
    return dataset.set_index(["Date", "Ticker"]).sort_index()


def build_market_neutral_weights(
    signals: pd.DataFrame, volatility: pd.DataFrame
) -> pd.DataFrame:
    """Allocate half of gross exposure to each side, inverse-volatility weighted."""
    weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)
    for date in signals.index:
        row_signals = signals.loc[date]
        row_volatility = volatility.loc[date]
        long_assets = row_signals.index[(row_signals > 0) & row_volatility.notna()]
        short_assets = row_signals.index[(row_signals < 0) & row_volatility.notna()]
        if len(long_assets) == 0 or len(short_assets) == 0:
            continue

        for assets, direction in ((long_assets, 1.0), (short_assets, -1.0)):
            inverse_volatility = 1 / row_volatility.loc[assets].clip(lower=1e-6)
            weights.loc[date, assets] = (
                direction * 0.5 * inverse_volatility / inverse_volatility.sum()
            )
    return weights


def build_weekly_weights(
    predictions: pd.Series,
    feature_data: pd.DataFrame,
    daily_dates: pd.DatetimeIndex,
    min_edge: float,
) -> pd.DataFrame:
    prediction_frame = predictions.rename("Score").to_frame().reset_index()
    rebalance_dates = prediction_frame["Date"].drop_duplicates().sort_values()
    signals = pd.DataFrame(0.0, index=rebalance_dates, columns=TICKERS)
    number_per_side = max(1, int(len(TICKERS) * TOP_BOTTOM_FRACTION))

    for date, rows in prediction_frame.groupby("Date"):
        scores = rows.set_index("Ticker")["Score"].reindex(TICKERS)
        longs = scores.nlargest(number_per_side)
        shorts = scores.nsmallest(number_per_side)
        signals.loc[date, longs.index[longs > min_edge]] = 1.0
        signals.loc[date, shorts.index[shorts < -min_edge]] = -1.0

    volatility = (
        feature_data["Volatility_20d"]
        .unstack("Ticker")
        .reindex(index=rebalance_dates, columns=TICKERS)
    )
    rebalance_weights = build_market_neutral_weights(signals, volatility)
    return rebalance_weights.reindex(daily_dates).ffill().fillna(0.0)


def _return_metrics(
    strategy_returns: pd.Series,
    weights: pd.DataFrame,
    benchmark_growth: pd.Series,
    equal_weight_growth: pd.Series,
) -> tuple[dict, pd.Series]:
    growth = (1 + strategy_returns).cumprod()
    drawdown = growth / growth.cummax() - 1
    daily_volatility = strategy_returns.std(ddof=1)
    sharpe = (
        np.sqrt(252) * strategy_returns.mean() / daily_volatility
        if daily_volatility > 0
        else 0.0
    )
    weight_changes = weights.diff()
    if not weight_changes.empty:
        weight_changes.iloc[0] = weights.iloc[0]
    metrics = {
        "Strategy_Return_%": round((growth.iloc[-1] - 1) * 100, 2),
        "SPY_Return_%": round((benchmark_growth.iloc[-1] - 1) * 100, 2),
        "Equal_Weight_Return_%": round((equal_weight_growth.iloc[-1] - 1) * 100, 2),
        "Cash_Return_%": 0.0,
        "Sharpe_Ratio": round(sharpe, 3),
        "Max_Drawdown_%": round(drawdown.min() * 100, 2),
        "Annualized_Volatility_%": round(daily_volatility * np.sqrt(252) * 100, 2),
        "Mean_Daily_Turnover_%": round(
            weight_changes.abs().sum(axis=1).mean() * 100, 2
        ),
    }
    return metrics, growth


def evaluate_portfolio(
    predictions: pd.Series,
    feature_data: pd.DataFrame,
    prices: pd.DataFrame,
    min_edge: float,
    evaluation_end: pd.Timestamp,
) -> tuple[dict, pd.Series, pd.Series, pd.Series]:
    start_date = predictions.index.get_level_values("Date").min()
    forward_returns = prices[TICKERS].pct_change(fill_method=None).shift(-1)
    spy_returns = prices["SPY"].pct_change(fill_method=None).shift(-1)
    daily_dates = prices.index[
        (prices.index >= start_date) & (prices.index < evaluation_end)
    ]
    daily_dates = daily_dates[forward_returns.reindex(daily_dates).notna().all(axis=1)]
    forward_returns = forward_returns.reindex(daily_dates)
    spy_returns = spy_returns.reindex(daily_dates)
    weights = build_weekly_weights(predictions, feature_data, daily_dates, min_edge)
    weights = weights.shift(EXECUTION_LAG_DAYS).fillna(0.0)

    weight_changes = weights.diff()
    if not weight_changes.empty:
        weight_changes.iloc[0] = weights.iloc[0]
    turnover = weight_changes.abs().sum(axis=1)
    trading_costs = turnover * (COST_PER_SIDE + SLIPPAGE_PER_SIDE)
    borrow_costs = weights.clip(upper=0).abs().sum(axis=1) * (
        SHORT_BORROW_RATE_ANNUAL / 252
    )
    strategy_returns = (weights * forward_returns).sum(axis=1) - trading_costs - borrow_costs
    benchmark_growth = (1 + spy_returns).cumprod()
    equal_weight_growth = (1 + forward_returns).cumprod().mean(axis=1)
    metrics, strategy_growth = _return_metrics(
        strategy_returns, weights, benchmark_growth, equal_weight_growth
    )
    return metrics, strategy_growth, benchmark_growth, equal_weight_growth


def evaluate_single_asset(
    predictions: pd.Series,
    prices: pd.DataFrame,
    ticker: str,
    min_edge: float,
    evaluation_end: pd.Timestamp,
) -> tuple[dict, pd.Series, pd.Series, pd.Series]:
    start_date = predictions.index.min()
    forward_returns = prices[ticker].pct_change(fill_method=None).shift(-1)
    daily_dates = prices.index[
        (prices.index >= start_date)
        & (prices.index < evaluation_end)
        & forward_returns.notna()
    ]
    weekly_positions = pd.Series(0.0, index=predictions.index)
    weekly_positions.loc[predictions > min_edge] = 1.0
    weekly_positions.loc[predictions < -min_edge] = -1.0
    positions = (
        weekly_positions.reindex(daily_dates)
        .ffill()
        .shift(EXECUTION_LAG_DAYS)
        .fillna(0.0)
    )

    position_changes = positions.diff()
    if not position_changes.empty:
        position_changes.iloc[0] = positions.iloc[0]
    actions = pd.Series("Hold", index=daily_dates)
    actions.loc[position_changes > 0] = "Buy"
    actions.loc[position_changes < 0] = "Sell"
    costs = position_changes.abs() * (COST_PER_SIDE + SLIPPAGE_PER_SIDE)
    borrow_costs = positions.clip(upper=0).abs() * SHORT_BORROW_RATE_ANNUAL / 252
    asset_returns = forward_returns.reindex(daily_dates)
    strategy_returns = positions * asset_returns - costs - borrow_costs
    growth = (1 + strategy_returns).cumprod()
    benchmark_growth = (1 + asset_returns).cumprod()
    drawdown = growth / growth.cummax() - 1
    daily_volatility = strategy_returns.std(ddof=1)
    sharpe = (
        np.sqrt(252) * strategy_returns.mean() / daily_volatility
        if daily_volatility > 0
        else 0.0
    )
    metrics = {
        "Strategy_Return_%": round((growth.iloc[-1] - 1) * 100, 2),
        "Benchmark_Return_%": round((benchmark_growth.iloc[-1] - 1) * 100, 2),
        "Sharpe_Ratio": round(sharpe, 3),
        "Max_Drawdown_%": round(drawdown.min() * 100, 2),
        "Annualized_Volatility_%": round(daily_volatility * np.sqrt(252) * 100, 2),
        "Mean_Daily_Turnover_%": round(position_changes.abs().mean() * 100, 2),
    }
    return metrics, growth, benchmark_growth, actions


def prediction_metrics(actual: pd.Series, predictions: pd.Series) -> dict:
    joined = pd.concat(
        [actual.rename("Actual"), predictions.rename("Prediction")], axis=1
    ).dropna()
    rank_ic = joined.groupby(level="Date").apply(
        lambda group: group["Actual"].corr(group["Prediction"], method="spearman")
        if len(group) > 1
        else np.nan
    )
    return {
        "RMSE_5d_Excess_%": round(
            np.sqrt(mean_squared_error(joined["Actual"], joined["Prediction"])) * 100, 3
        ),
        "Mean_Rank_IC": round(rank_ic.mean(), 4),
    }


def make_walk_forward_folds(
    development_dates: pd.DatetimeIndex, n_splits: int = WALK_FORWARD_SPLITS
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    minimum_train_size = max(52, len(development_dates) // 2)
    fold_size = (len(development_dates) - minimum_train_size) // n_splits
    if fold_size < 10:
        raise ValueError("Not enough weekly history for walk-forward validation")

    folds = []
    for fold in range(n_splits):
        start = minimum_train_size + fold * fold_size
        end = (
            len(development_dates)
            if fold == n_splits - 1
            else minimum_train_size + (fold + 1) * fold_size
        )
        folds.append((development_dates[:start], development_dates[start:end]))
    return folds


def create_models() -> dict:
    return {
        "Random_Forest": RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=10,
            max_features=0.8,
            random_state=42,
            n_jobs=-1,
        ),
        "Extra_Trees": ExtraTreesRegressor(
            n_estimators=250,
            min_samples_leaf=10,
            max_features=0.8,
            random_state=42,
            n_jobs=-1,
        ),
        "Hist_Gradient_Boosting": HistGradientBoostingRegressor(
            max_iter=150, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=1.0,
            random_state=42,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_lambda=2,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        ),
        "Ridge_Regression": make_pipeline(
            StandardScaler(), Ridge(alpha=10.0)
        ),
    }


def select_threshold_walk_forward(
    model_name: str,
    weekly_data: pd.DataFrame,
    feature_data: pd.DataFrame,
    prices: pd.DataFrame,
    development_dates: pd.DatetimeIndex,
    weekly_dates: pd.DatetimeIndex,
) -> tuple[float, dict]:
    fold_metrics = {threshold: [] for threshold in MIN_EDGE_CANDIDATES}
    date_values = weekly_data.index.get_level_values("Date")

    for fold_number, (train_dates, validation_dates) in enumerate(
        make_walk_forward_folds(development_dates), start=1
    ):
        train_data = weekly_data.loc[date_values.isin(train_dates)]
        validation_data = weekly_data.loc[date_values.isin(validation_dates)]
        model = create_models()[model_name]
        model.fit(train_data[FEATURES], train_data[TARGET])
        validation_predictions = pd.Series(
            model.predict(validation_data[FEATURES]), index=validation_data.index
        )

        last_validation_date = validation_dates[-1]
        last_position = weekly_dates.get_loc(last_validation_date)
        evaluation_end = (
            weekly_dates[last_position + 1]
            if last_position + 1 < len(weekly_dates)
            else prices.index[-1]
        )
        for threshold in MIN_EDGE_CANDIDATES:
            metrics, _, _, _ = evaluate_portfolio(
                validation_predictions,
                feature_data,
                prices,
                threshold,
                evaluation_end,
            )
            fold_metrics[threshold].append(metrics)
        print(
            f"  {model_name}: walk-forward fold {fold_number}/{WALK_FORWARD_SPLITS}"
        )

    summaries = {
        threshold: {
            "CV_Mean_Sharpe": float(
                np.mean([metrics["Sharpe_Ratio"] for metrics in values])
            ),
            "CV_Mean_Return_%": float(
                np.mean([metrics["Strategy_Return_%"] for metrics in values])
            ),
        }
        for threshold, values in fold_metrics.items()
    }
    best_threshold = max(
        MIN_EDGE_CANDIDATES,
        key=lambda threshold: (
            summaries[threshold]["CV_Mean_Sharpe"],
            threshold,
        ),
    )
    return best_threshold, summaries[best_threshold]


def simulate_budget_portfolio(
    predictions: pd.Series,
    prices: pd.DataFrame,
    min_edge: float,
    evaluation_end: pd.Timestamp,
    include_end: bool = False,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    signal_dates = predictions.index.get_level_values("Date").unique().sort_values()
    if signal_dates.empty:
        raise ValueError("No predictions available for portfolio simulation")

    date_mask = prices.index >= signal_dates[0]
    date_mask &= prices.index <= evaluation_end if include_end else prices.index < evaluation_end
    daily_dates = prices.index[date_mask]
    if daily_dates.empty:
        raise ValueError("No price dates available for portfolio simulation")

    prediction_frame = predictions.rename("Score").to_frame().reset_index()
    scheduled_orders = {}
    for signal_date, rows in prediction_frame.groupby("Date"):
        signal_position = prices.index.get_indexer([signal_date])[0]
        if signal_position < 0 or signal_position + 1 >= len(prices.index):
            continue
        execution_date = prices.index[signal_position + 1]
        if execution_date in daily_dates:
            scheduled_orders[execution_date] = rows.set_index("Ticker")["Score"]

    cash = STARTING_CAPITAL
    shares = {ticker: 0.0 for ticker in TICKERS}
    trade_log = []
    account_rows = []
    position_rows = []
    transaction_cost_rate = COST_PER_SIDE + SLIPPAGE_PER_SIDE

    for date in daily_dates:
        close_prices = prices.loc[date, TICKERS]
        if date in scheduled_orders:
            scores = scheduled_orders[date].reindex(TICKERS)
            candidates = scores[scores > min_edge].sort_values(ascending=False)
            selected = candidates.head(MAX_HOLDINGS).index.tolist()
            if len(selected) < MIN_HOLDINGS:
                selected = []

            equity_before_trading = cash + sum(
                shares[ticker] * close_prices[ticker] for ticker in TICKERS
            )
            target_weight = (
                min(MAX_POSITION_WEIGHT, 1 / len(selected)) if selected else 0.0
            )
            target_equity = equity_before_trading
            for _ in range(10):
                target_values = {
                    ticker: (
                        target_equity * target_weight if ticker in selected else 0.0
                    )
                    for ticker in TICKERS
                }
                expected_turnover = sum(
                    abs(
                        target_values[ticker] - shares[ticker] * close_prices[ticker]
                    )
                    for ticker in TICKERS
                )
                target_equity = max(
                    equity_before_trading
                    - expected_turnover * transaction_cost_rate,
                    0.0,
                )
            target_shares = {
                ticker: (
                    target_equity * target_weight / close_prices[ticker]
                    if ticker in selected
                    else 0.0
                )
                for ticker in TICKERS
            }

            for ticker in TICKERS:
                quantity = max(shares[ticker] - target_shares[ticker], 0.0)
                if quantity <= 1e-10:
                    continue
                notional = quantity * close_prices[ticker]
                cost = notional * transaction_cost_rate
                cash += notional - cost
                shares[ticker] -= quantity
                trade_log.append(
                    {
                        "Date": date,
                        "Ticker": ticker,
                        "Side": "Sell",
                        "Units": quantity,
                        "Price": close_prices[ticker],
                        "Notional_$": notional,
                        "Transaction_Cost_$": cost,
                    }
                )

            for ticker in selected:
                quantity_needed = max(target_shares[ticker] - shares[ticker], 0.0)
                unit_cost = close_prices[ticker] * (1 + transaction_cost_rate)
                affordable_quantity = cash / unit_cost
                quantity = min(quantity_needed, affordable_quantity)
                if quantity <= 1e-10:
                    continue
                notional = quantity * close_prices[ticker]
                cost = notional * transaction_cost_rate
                cash -= notional + cost
                shares[ticker] += quantity
                trade_log.append(
                    {
                        "Date": date,
                        "Ticker": ticker,
                        "Side": "Buy",
                        "Units": quantity,
                        "Price": close_prices[ticker],
                        "Notional_$": notional,
                        "Transaction_Cost_$": cost,
                    }
                )

        invested_value = sum(
            shares[ticker] * close_prices[ticker] for ticker in TICKERS
        )
        equity = cash + invested_value
        account_rows.append(
            {
                "Date": date,
                "Account_Value_$": equity,
                "Cash_$": cash,
                "Invested_Value_$": invested_value,
                "Holdings": sum(shares[ticker] > 1e-10 for ticker in TICKERS),
            }
        )
        for ticker in TICKERS:
            if shares[ticker] > 1e-10:
                position_rows.append(
                    {
                        "Date": date,
                        "Ticker": ticker,
                        "Units": shares[ticker],
                        "Market_Value_$": shares[ticker] * close_prices[ticker],
                        "Weight_%": shares[ticker] * close_prices[ticker] / equity * 100,
                    }
                )

    account = pd.DataFrame(account_rows).set_index("Date")
    benchmark_prices = prices.loc[daily_dates]
    account["SPY_Buy_Hold_$"] = (
        benchmark_prices["SPY"] / benchmark_prices["SPY"].iloc[0] * STARTING_CAPITAL
    )
    equal_weight_growth = (
        benchmark_prices[TICKERS].div(benchmark_prices[TICKERS].iloc[0]).mean(axis=1)
        * STARTING_CAPITAL
    )
    account["Equal_Weight_Buy_Hold_$"] = equal_weight_growth
    account["Cash_Benchmark_$"] = STARTING_CAPITAL

    equity_returns = account["Account_Value_$"].pct_change().fillna(0.0)
    drawdown = account["Account_Value_$"] / account["Account_Value_$"].cummax() - 1
    daily_volatility = equity_returns.std(ddof=1)
    sharpe = (
        np.sqrt(252) * equity_returns.mean() / daily_volatility
        if daily_volatility > 0
        else 0.0
    )
    trade_frame = pd.DataFrame(
        trade_log,
        columns=[
            "Date",
            "Ticker",
            "Side",
            "Units",
            "Price",
            "Notional_$",
            "Transaction_Cost_$",
        ],
    )
    final_equity = account["Account_Value_$"].iloc[-1]
    metrics = {
        "Starting_Capital_$": STARTING_CAPITAL,
        "Ending_Value_$": round(final_equity, 2),
        "Profit_$": round(final_equity - STARTING_CAPITAL, 2),
        "Return_%": round((final_equity / STARTING_CAPITAL - 1) * 100, 2),
        "SPY_Ending_Value_$": round(account["SPY_Buy_Hold_$"].iloc[-1], 2),
        "Equal_Weight_Ending_Value_$": round(
            account["Equal_Weight_Buy_Hold_$"].iloc[-1], 2
        ),
        "Max_Drawdown_%": round(drawdown.min() * 100, 2),
        "Sharpe_Ratio": round(sharpe, 3),
        "Total_Costs_$": round(trade_frame["Transaction_Cost_$"].sum(), 2),
        "Number_of_Trades": len(trade_frame),
        "Average_Holdings": round(account["Holdings"].mean(), 2),
        "Max_Holdings": int(account["Holdings"].max()),
        "Final_Cash_$": round(account["Cash_$"].iloc[-1], 2),
    }
    positions = pd.DataFrame(
        position_rows, columns=["Date", "Ticker", "Units", "Market_Value_$", "Weight_%"]
    )
    return metrics, account, trade_frame, positions


def select_budget_portfolio_model(
    weekly_data: pd.DataFrame,
    prices: pd.DataFrame,
    development_dates: pd.DatetimeIndex,
    weekly_dates: pd.DatetimeIndex,
) -> tuple[str, float, dict]:
    date_values = weekly_data.index.get_level_values("Date")
    results = {}

    for name in create_models():
        fold_results = {edge: [] for edge in BUDGET_EDGE_CANDIDATES}
        for fold_number, (train_dates, validation_dates) in enumerate(
            make_walk_forward_folds(development_dates), start=1
        ):
            train_data = weekly_data.loc[date_values.isin(train_dates)]
            validation_data = weekly_data.loc[date_values.isin(validation_dates)]
            model = create_models()[name]
            model.fit(train_data[FEATURES], train_data[TARGET])
            predictions = pd.Series(
                model.predict(validation_data[FEATURES]), index=validation_data.index
            )
            last_position = weekly_dates.get_loc(validation_dates[-1])
            evaluation_end = (
                weekly_dates[last_position + 1]
                if last_position + 1 < len(weekly_dates)
                else prices.index[-1]
            )
            for edge in BUDGET_EDGE_CANDIDATES:
                metrics, _, _, _ = simulate_budget_portfolio(
                    predictions, prices, edge, evaluation_end
                )
                fold_results[edge].append(metrics)
            print(f"  {name}: budget walk-forward fold {fold_number}/{WALK_FORWARD_SPLITS}")

        edge_summaries = {
            edge: {
                "CV_Mean_Sharpe": float(
                    np.mean([metrics["Sharpe_Ratio"] for metrics in values])
                ),
                "CV_Mean_Return_%": float(
                    np.mean([metrics["Return_%"] for metrics in values])
                ),
            }
            for edge, values in fold_results.items()
        }
        selected_edge = max(
            BUDGET_EDGE_CANDIDATES,
            key=lambda edge: (
                edge_summaries[edge]["CV_Mean_Sharpe"],
                edge_summaries[edge]["CV_Mean_Return_%"],
                edge,
            ),
        )
        results[name] = {
            "Min_Edge": selected_edge,
            **edge_summaries[selected_edge],
        }
        print(
            f"  {name}: edge {selected_edge:.2%}, "
            f"CV Sharpe {results[name]['CV_Mean_Sharpe']:.3f}"
        )

    selected_model = max(
        results,
        key=lambda name: (
            results[name]["CV_Mean_Sharpe"],
            results[name]["CV_Mean_Return_%"],
        ),
    )
    return selected_model, results[selected_model]["Min_Edge"], results


def plot_portfolio_performance(
    name: str,
    strategy_growth: pd.Series,
    spy_growth: pd.Series,
    equal_weight_growth: pd.Series,
    plot_dir: Path,
) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(spy_growth.index, spy_growth, label="SPY Buy & Hold", color="gray")
    plt.plot(
        equal_weight_growth.index,
        equal_weight_growth,
        label="Equal-Weight Buy & Hold",
        color="darkorange",
    )
    plt.plot(strategy_growth.index, strategy_growth, label=f"{name} Strategy (Net)", linewidth=2)
    plt.axhline(1.0, color="black", linestyle="--", alpha=0.5, label="Cash")
    plt.title(f"Walk-Forward Strategy Performance: {name}")
    plt.xlabel("Signal date (close)")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_dir / f"{name.lower()}_performance.png")
    plt.close()


def plot_budget_portfolio(
    model_name: str,
    account: pd.DataFrame,
    trades: pd.DataFrame,
    positions: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, (axis, holdings_axis) = plt.subplots(
        2,
        1,
        figsize=(14, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.4]},
    )
    axis.plot(
        account.index,
        account["Account_Value_$"],
        label=f"{model_name} (budgeted)",
        linewidth=2,
        color="royalblue",
    )
    axis.plot(
        account.index,
        account["SPY_Buy_Hold_$"],
        label="SPY Buy & Hold",
        color="gray",
        alpha=0.8,
    )
    axis.plot(
        account.index,
        account["Equal_Weight_Buy_Hold_$"],
        label="Equal-Weight Buy & Hold",
        color="darkorange",
    )
    axis.axhline(
        STARTING_CAPITAL,
        color="black",
        linestyle="--",
        alpha=0.5,
        label="Cash",
    )
    if not trades.empty:
        for side, color, marker in (
            ("Buy", "green", "^"),
            ("Sell", "red", "v"),
        ):
            trade_dates = trades.loc[trades["Side"] == side, "Date"].drop_duplicates()
            axis.scatter(
                trade_dates,
                account.loc[trade_dates, "Account_Value_$"],
                label=side,
                color=color,
                marker=marker,
                s=40,
                zorder=3,
            )

    holding_matrix = (
        positions.pivot_table(
            index="Date",
            columns="Ticker",
            values="Market_Value_$",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(index=account.index, columns=TICKERS, fill_value=0.0)
        .gt(0)
    )
    date_numbers = mdates.date2num(account.index.to_pydatetime())
    if len(date_numbers) > 1:
        midpoints = (date_numbers[:-1] + date_numbers[1:]) / 2
        date_edges = np.concatenate(
            (
                [date_numbers[0] - (midpoints[0] - date_numbers[0])],
                midpoints,
                [date_numbers[-1] + (date_numbers[-1] - midpoints[-1])],
            )
        )
    else:
        date_edges = np.array([date_numbers[0] - 0.5, date_numbers[0] + 0.5])
    holdings_axis.pcolormesh(
        date_edges,
        np.arange(len(TICKERS) + 1),
        holding_matrix.to_numpy(dtype=int).T,
        cmap=ListedColormap(["#eeeeee", "seagreen"]),
        vmin=0,
        vmax=1,
        shading="flat",
    )
    holdings_axis.set_yticks(np.arange(len(TICKERS)) + 0.5)
    holdings_axis.set_yticklabels(TICKERS)
    holdings_axis.set_ylabel("Stocks held")
    holdings_axis.set_xlabel("Date")
    holdings_axis.xaxis.set_major_locator(mdates.AutoDateLocator())
    holdings_axis.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(holdings_axis.xaxis.get_major_locator())
    )
    axis.set_title(
        f"Budgeted portfolio: {model_name} | "
        f"ending value ${account['Account_Value_$'].iloc[-1]:,.2f}"
    )
    axis.set_ylabel("Portfolio value (USD)")
    axis.legend(loc="upper left")
    axis.grid(True, alpha=0.3)
    figure.subplots_adjust(hspace=0.08)
    figure.savefig(output_path)
    plt.close(figure)


def plot_asset_performance(
    ticker: str,
    model_name: str,
    strategy_growth: pd.Series,
    benchmark_growth: pd.Series,
    actions: pd.Series,
    plot_dir: Path,
) -> None:
    ticker_plot_dir = plot_dir / ticker
    ticker_plot_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(
        benchmark_growth.index,
        benchmark_growth,
        label=f"{ticker} Buy & Hold",
        color="gray",
        alpha=0.7,
    )
    plt.plot(
        strategy_growth.index,
        strategy_growth,
        label=f"{model_name} Strategy (Net)",
        linewidth=2,
    )
    action_styles = {
        "Buy": ("green", "^"),
        "Sell": ("red", "v"),
        "Hold": ("darkorange", "o"),
    }
    for action, (color, marker) in action_styles.items():
        action_mask = actions == action
        plt.scatter(
            strategy_growth.index[action_mask],
            strategy_growth.loc[action_mask],
            label=action,
            color=color,
            marker=marker,
            s=24 if action != "Hold" else 12,
            alpha=0.8 if action != "Hold" else 0.35,
            zorder=3,
        )
    plt.title(f"Weekly ML Strategy: {model_name} - {ticker}")
    plt.xlabel("Signal date (close)")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(ticker_plot_dir / f"{model_name.lower()}_performance.png")
    plt.close()


def _select_dates(data: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    return data.loc[data.index.get_level_values("Date").isin(dates)]


def run_training() -> pd.DataFrame:
    loaded_prices = {
        ticker: load_close(ticker, START_DATE, END_DATE)
        for ticker in [*TICKERS, "SPY", "QQQ"]
    }
    prices = pd.concat(loaded_prices, axis=1).dropna()
    prices.columns = prices.columns.get_level_values(0)
    feature_data = build_feature_table(prices)
    labeled_data = feature_data.dropna(subset=[TARGET])
    labeled_dates = (
        labeled_data.index.get_level_values("Date").unique().sort_values()
    )
    weekly_dates = labeled_dates[::REBALANCE_EVERY_DAYS]
    weekly_data = _select_dates(labeled_data, weekly_dates)
    if len(weekly_dates) < 100:
        raise ValueError(f"Not enough weekly observations: {len(weekly_dates)}")

    split_index = int(len(weekly_dates) * TRAIN_FRACTION)
    development_dates = weekly_dates[:split_index]
    holdout_dates = weekly_dates[split_index:]
    split_date = holdout_dates[0]
    train_data = _select_dates(weekly_data, development_dates)
    test_data = _select_dates(weekly_data, holdout_dates)
    evaluation_end = prices.index[-1]
    results = []
    output_dir = Path(__file__).resolve().parent
    plot_dir = output_dir / "images" / "ml_models"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for name in create_models():
        print(f"\nSelecting entry threshold for {name}...")
        threshold, cv_metrics = select_threshold_walk_forward(
            name,
            weekly_data.loc[weekly_data.index.get_level_values("Date").isin(development_dates)],
            feature_data,
            prices,
            development_dates,
            weekly_dates,
        )
        model = create_models()[name]
        model.fit(train_data[FEATURES], train_data[TARGET])
        predictions = pd.Series(model.predict(test_data[FEATURES]), index=test_data.index)

        portfolio_metrics, strategy_growth, spy_growth, equal_growth = evaluate_portfolio(
            predictions, feature_data, prices, threshold, evaluation_end
        )
        prediction_stats = prediction_metrics(test_data[TARGET], predictions)
        common = {
            "Train_Start": development_dates.min().date(),
            "Test_Start": split_date.date(),
            "Test_End": prices.index[-1].date(),
            "Horizon_Days": HORIZON_DAYS,
            "Rebalance_Days": REBALANCE_EVERY_DAYS,
            "Min_Edge_%": round(threshold * 100, 3),
            **cv_metrics,
        }
        results.append(
            {
                "Model": name,
                "Scope": "Portfolio",
                "Ticker": "ALL",
                "Stocks": len(TICKERS),
                **common,
                **prediction_stats,
                **portfolio_metrics,
            }
        )
        plot_portfolio_performance(
            name, strategy_growth, spy_growth, equal_growth, plot_dir
        )

        for ticker in TICKERS:
            ticker_predictions = predictions.xs(ticker, level="Ticker")
            asset_metrics, asset_growth, asset_benchmark, actions = evaluate_single_asset(
                ticker_predictions, prices, ticker, threshold, evaluation_end
            )
            ticker_actual = test_data.xs(ticker, level="Ticker")[TARGET]
            ticker_rmse = np.sqrt(
                mean_squared_error(ticker_actual, ticker_predictions)
            ) * 100
            results.append(
                {
                    "Model": name,
                    "Scope": "Stock",
                    "Ticker": ticker,
                    "Stocks": 1,
                    **common,
                    "RMSE_5d_Excess_%": round(ticker_rmse, 3),
                    "Mean_Rank_IC": np.nan,
                    **asset_metrics,
                }
            )
            plot_asset_performance(
                ticker,
                name,
                asset_growth,
                asset_benchmark,
                actions,
                plot_dir,
            )

        print(
            f"{name}: selected {threshold:.2%} edge; "
            f"walk-forward Sharpe {cv_metrics['CV_Mean_Sharpe']:.3f}"
        )

    baseline_scores = {
        "MOMENTUM_20D": weekly_data["Return_20d"].groupby(level="Date").transform(
            lambda values: values - values.mean()
        ),
        "MEAN_REVERSION_20D": -weekly_data["Return_20d"].groupby(
            level="Date"
        ).transform(lambda values: values - values.mean()),
    }
    random_scores = pd.Series(
        np.random.default_rng(42).normal(size=len(weekly_data)),
        index=weekly_data.index,
    )
    baseline_scores["RANDOM"] = random_scores.groupby(level="Date").transform(
        lambda values: values - values.mean()
    )
    baseline_scores["CASH"] = pd.Series(0.0, index=weekly_data.index)

    for name, scores in baseline_scores.items():
        predictions = scores.loc[scores.index.get_level_values("Date").isin(holdout_dates)]
        metrics, strategy_growth, spy_growth, equal_growth = evaluate_portfolio(
            predictions, feature_data, prices, 0.0, evaluation_end
        )
        results.append(
            {
                "Model": name,
                "Scope": "Baseline",
                "Ticker": "ALL",
                "Stocks": len(TICKERS),
                "Train_Start": development_dates.min().date(),
                "Test_Start": split_date.date(),
                "Test_End": prices.index[-1].date(),
                "Horizon_Days": HORIZON_DAYS,
                "Rebalance_Days": REBALANCE_EVERY_DAYS,
                "Min_Edge_%": 0.0,
                **metrics,
            }
        )
        plot_portfolio_performance(
            name, strategy_growth, spy_growth, equal_growth, plot_dir
        )

    print("\nSelecting model for the capital-limited portfolio...")
    budget_model_name, budget_min_edge, budget_cv_results = (
        select_budget_portfolio_model(
            weekly_data, prices, development_dates, weekly_dates
        )
    )
    budget_model = create_models()[budget_model_name]
    budget_model.fit(train_data[FEATURES], train_data[TARGET])
    budget_predictions = pd.Series(
        budget_model.predict(test_data[FEATURES]), index=test_data.index
    )
    budget_metrics, account, trades, positions = simulate_budget_portfolio(
        budget_predictions,
        prices,
        budget_min_edge,
        evaluation_end,
        include_end=True,
    )
    budget_cv_metrics = budget_cv_results[budget_model_name]
    results.append(
        {
            "Model": budget_model_name,
            "Scope": "Budget_Portfolio",
            "Ticker": "ALL",
            "Stocks": len(TICKERS),
            "Train_Start": development_dates.min().date(),
            "Test_Start": split_date.date(),
            "Test_End": prices.index[-1].date(),
            "Horizon_Days": HORIZON_DAYS,
            "Rebalance_Days": REBALANCE_EVERY_DAYS,
            "Min_Edge_%": round(budget_min_edge * 100, 3),
            "Budget_CV_Mean_Sharpe": round(
                budget_cv_metrics["CV_Mean_Sharpe"], 3
            ),
            "Budget_CV_Mean_Return_%": round(
                budget_cv_metrics["CV_Mean_Return_%"], 2
            ),
            **budget_metrics,
        }
    )

    scores = pd.DataFrame(results)
    csv_dir = output_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_filename = csv_dir / "model_comparison_results.csv"
    account_filename = csv_dir / "budget_portfolio_equity_curve.csv"
    trades_filename = csv_dir / "budget_portfolio_trades.csv"
    positions_filename = csv_dir / "budget_portfolio_positions.csv"
    summary_filename = csv_dir / "budget_portfolio_summary.csv"
    scores.to_csv(csv_filename, index=False)
    account.to_csv(account_filename, index_label="Date")
    trades.to_csv(trades_filename, index=False)
    positions.to_csv(positions_filename, index=False)
    pd.DataFrame(
        [
            {
                "Model": budget_model_name,
                "Min_Edge_%": round(budget_min_edge * 100, 3),
                "Budget_CV_Mean_Sharpe": budget_cv_metrics["CV_Mean_Sharpe"],
                "Budget_CV_Mean_Return_%": budget_cv_metrics["CV_Mean_Return_%"],
                **budget_metrics,
            }
        ]
    ).to_csv(summary_filename, index=False)
    budget_plot_filename = plot_dir / "selected_budget_portfolio.png"
    plot_budget_portfolio(
        budget_model_name, account, trades, positions, budget_plot_filename
    )

    print("\n--- Walk-forward selected models and final holdout ---")
    print(
        scores.loc[
            scores["Scope"].isin(["Portfolio", "Baseline"]),
            [
                "Model",
                "Scope",
                "Min_Edge_%",
                "CV_Mean_Sharpe",
                "Strategy_Return_%",
                "SPY_Return_%",
                "Equal_Weight_Return_%",
                "Sharpe_Ratio",
            ],
        ].to_string(index=False)
    )
    print(
        f"\nSelected budget portfolio: {budget_model_name}, "
        f"edge {budget_min_edge:.2%}, "
        f"CV Sharpe {budget_cv_metrics['CV_Mean_Sharpe']:.3f}"
    )
    print(
        f"Holdout: ${STARTING_CAPITAL:,.2f} -> "
        f"${budget_metrics['Ending_Value_$']:,.2f} "
        f"(profit ${budget_metrics['Profit_$']:,.2f}, "
        f"{budget_metrics['Return_%']:.2f}%, "
        f"{budget_metrics['Number_of_Trades']} trades)"
    )
    print(f"Results saved to '{csv_filename}'")
    print(f"Budget account curve saved to '{account_filename}'")
    print(f"Budget portfolio chart saved to '{budget_plot_filename}'")
    return scores


if __name__ == "__main__":
    run_training()
