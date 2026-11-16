import unittest

import numpy as np
import pandas as pd

from ML_Models import (
    TICKERS,
    build_feature_table,
    build_market_neutral_weights,
    build_weekly_weights,
    make_walk_forward_folds,
    simulate_budget_portfolio,
)


class FeatureTableTests(unittest.TestCase):
    def test_target_is_five_day_excess_return(self):
        dates = pd.bdate_range("2020-01-01", periods=180)
        returns = np.full(len(dates), 0.001)
        returns[100] = 0.02
        prices = pd.DataFrame(
            {
                "AAA": 100 * np.cumprod(1 + returns),
                "BBB": 100 * np.cumprod(1 - returns / 2),
                "SPY": 100 * np.cumprod(1 + returns / 3),
                "QQQ": 100 * np.cumprod(1 + returns / 2),
            },
            index=dates,
        )

        dataset = build_feature_table(prices, tickers=["AAA", "BBB"])
        row = dataset.loc[(dates[95], "AAA")]
        future_a = prices.loc[dates[101], "AAA"] / prices.loc[dates[96], "AAA"] - 1
        future_b = prices.loc[dates[101], "BBB"] / prices.loc[dates[96], "BBB"] - 1

        next_day_row = dataset.loc[(dates[99], "AAA")]
        self.assertAlmostEqual(next_day_row["Forward_Return_1d"], 0.02)
        self.assertAlmostEqual(row["Forward_Excess_Return_5d"], (future_a - future_b) / 2)


class PortfolioWeightTests(unittest.TestCase):
    def test_market_neutral_weights_and_one_sided_signals(self):
        dates = pd.bdate_range("2024-01-01", periods=2)
        signals = pd.DataFrame(
            [[1, 1, -1, -1], [1, 1, 0, 0]],
            index=dates,
            columns=["A", "B", "C", "D"],
        )
        volatility = pd.DataFrame(
            [[0.1, 0.2, 0.1, 0.2], [0.1, 0.2, 0.1, 0.2]],
            index=dates,
            columns=signals.columns,
        )

        weights = build_market_neutral_weights(signals, volatility)

        self.assertAlmostEqual(weights.loc[dates[0]].sum(), 0.0)
        self.assertAlmostEqual(weights.loc[dates[0]].abs().sum(), 1.0)
        self.assertTrue((weights.loc[dates[1]] == 0).all())

    def test_weekly_weights_hold_between_rebalances_and_enforce_edge(self):
        dates = pd.bdate_range("2024-01-01", periods=2)
        index = pd.MultiIndex.from_product(
            [[dates[0]], TICKERS], names=["Date", "Ticker"]
        )
        scores = pd.Series(
            [0.02, 0.015, 0.001, 0.0005, -0.0005, -0.001, -0.015, -0.02, 0, 0],
            index=index,
        )
        feature_data = pd.DataFrame(
            {"Volatility_20d": 0.2}, index=index
        )

        weights = build_weekly_weights(
            scores, feature_data, pd.DatetimeIndex(dates), min_edge=0.005
        )

        self.assertAlmostEqual(weights.loc[dates[0]].sum(), 0.0)
        self.assertAlmostEqual(weights.loc[dates[0]].abs().sum(), 1.0)
        np.testing.assert_array_equal(weights.loc[dates[0]], weights.loc[dates[1]])


class WalkForwardTests(unittest.TestCase):
    def test_folds_expand_training_only_into_later_validation_dates(self):
        dates = pd.bdate_range("2020-01-01", periods=220, freq="5B")

        folds = make_walk_forward_folds(dates)

        self.assertEqual(len(folds), 3)
        for train_dates, validation_dates in folds:
            self.assertLess(train_dates.max(), validation_dates.min())
        self.assertEqual(folds[0][1].min(), folds[1][0][len(folds[0][0])])


class BudgetPortfolioTests(unittest.TestCase):
    def test_budget_account_respects_cash_and_position_limits(self):
        dates = pd.bdate_range("2024-01-01", periods=12)
        price_data = {
            ticker: 100 * (1.001 ** np.arange(len(dates))) for ticker in TICKERS
        }
        price_data["SPY"] = 100 * (1.001 ** np.arange(len(dates)))
        price_data["QQQ"] = 100 * (1.001 ** np.arange(len(dates)))
        prices = pd.DataFrame(price_data, index=dates)
        prediction_index = pd.MultiIndex.from_product(
            [[dates[0]], TICKERS], names=["Date", "Ticker"]
        )
        predictions = pd.Series(
            [0.02, 0.015, 0.01, *([0.0] * (len(TICKERS) - 3))],
            index=prediction_index,
        )

        metrics, account, trades, positions = simulate_budget_portfolio(
            predictions, prices, min_edge=0.005, evaluation_end=dates[-1],
            include_end=True,
        )

        self.assertTrue((account["Cash_$"] >= 0).all())
        self.assertLess(account.loc[dates[1], "Cash_$"], 10_000)
        self.assertEqual(account.loc[dates[1], "Holdings"], 3)
        rebalance_dates = trades["Date"].drop_duplicates()
        self.assertLessEqual(
            positions.loc[positions["Date"].isin(rebalance_dates), "Weight_%"].max(),
            20.000001,
        )
        self.assertGreater(metrics["Total_Costs_$"], 0)
        self.assertAlmostEqual(
            account["Cash_$"].iloc[-1], metrics["Final_Cash_$"], places=2
        )
        self.assertTrue((trades["Date"] == dates[1]).all())


if __name__ == "__main__":
    unittest.main()
