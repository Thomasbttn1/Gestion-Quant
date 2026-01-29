"""
Replication-inspired framework + Extension: Rolling VAR(1) factor forecasting
Based on "Time-Varying Factor Allocation" (Vincenz & Zeissler, 2022)

Pipeline:
1) Load data
2) Build factor returns
3) Build macro predictors
4) Bayesian univariate predictive regression (macro -> factor)
5) EXTENSION: Rolling VAR(1) on factor returns (factor -> factor)
6) Allocation (utility / TE-like) + performance + Holm
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize


# =========================
# 1) DATA LOADER
# =========================

class DataLoader:
    """Load and preprocess data from Excel file with custom sheet layouts."""

    def __init__(self, filepath="DataGestionQuant.xlsx"):
        self.filepath = filepath
        self.xlsx = pd.ExcelFile(filepath)

    def _std_df(self, data_dict):
        df = pd.DataFrame(data_dict)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        return df

    def load_currencies(self):
        df = pd.read_excel(self.xlsx, sheet_name="Currencies", header=None)
        dates = pd.to_datetime(df.iloc[5:, 0])
        data_dict = {"Date": dates.values}

        col_idx = 1
        while col_idx < len(df.columns):
            ticker = df.iloc[3, col_idx]
            if pd.notna(ticker) and "Curncy" in str(ticker):
                bid_col = df.iloc[5:, col_idx].values
                ask_col = df.iloc[5:, col_idx + 1].values if col_idx + 1 < len(df.columns) else bid_col
                ticker_name = str(ticker).replace(" Curncy", "").replace("USD", "")
                data_dict[f"{ticker_name}_bid"] = pd.to_numeric(bid_col, errors="coerce")
                data_dict[f"{ticker_name}_ask"] = pd.to_numeric(ask_col, errors="coerce")
                col_idx += 2
            else:
                col_idx += 1

        return self._std_df(data_dict)

    def load_currencies_forward(self):
        df = pd.read_excel(self.xlsx, sheet_name="Currencies Forward", header=None)
        dates = pd.to_datetime(df.iloc[7:, 3])
        data_dict = {"Date": dates.values}

        for col_idx in range(4, len(df.columns)):
            ticker = df.iloc[5, col_idx]
            if pd.notna(ticker) and "BGN Curncy" in str(ticker):
                ticker_name = str(ticker).replace(" BGN Curncy", "")
                data_dict[ticker_name] = pd.to_numeric(df.iloc[7:, col_idx].values, errors="coerce")

        return self._std_df(data_dict)

    def load_rates_curves(self):
        df = pd.read_excel(self.xlsx, sheet_name="Rates Curves", header=None)
        dates = pd.to_datetime(df.iloc[5:, 0])
        data_dict = {"Date": dates.values}

        col_idx = 1
        while col_idx < len(df.columns):
            ticker = df.iloc[3, col_idx]
            if pd.notna(ticker) and "Index" in str(ticker):
                ticker_name = str(ticker).replace(" Index", "")
                data_dict[ticker_name] = pd.to_numeric(df.iloc[5:, col_idx].values, errors="coerce")
            col_idx += 1

        return self._std_df(data_dict)

    def load_commodities(self):
        df = pd.read_excel(self.xlsx, sheet_name="Commodities", header=None)
        dates = pd.to_datetime(df.iloc[5:, 0])
        data_dict = {"Date": dates.values}

        col_idx = 1
        while col_idx < len(df.columns):
            ticker = df.iloc[3, col_idx]
            if pd.notna(ticker) and "Comdty" in str(ticker):
                ticker_name = str(ticker).replace(" Comdty", "")
                data_dict[ticker_name] = pd.to_numeric(df.iloc[5:, col_idx].values, errors="coerce")
                col_idx += 3
            else:
                col_idx += 1

        return self._std_df(data_dict)

    def load_stock_indices(self):
        df = pd.read_excel(self.xlsx, sheet_name="Stock Indices", header=None)
        dates = pd.to_datetime(df.iloc[5:, 0])
        data_dict = {"Date": dates.values}

        col_idx = 1
        while col_idx < len(df.columns):
            ticker = df.iloc[3, col_idx]
            if pd.notna(ticker) and "MX" in str(ticker):
                ticker_name = str(ticker).replace(" Index", "")
                data_dict[f"{ticker_name}_price"] = pd.to_numeric(df.iloc[5:, col_idx].values, errors="coerce")
                if col_idx + 1 < len(df.columns):
                    data_dict[f"{ticker_name}_tri"] = pd.to_numeric(df.iloc[5:, col_idx + 1].values, errors="coerce")
                if col_idx + 2 < len(df.columns):
                    data_dict[f"{ticker_name}_mcap"] = pd.to_numeric(df.iloc[5:, col_idx + 2].values, errors="coerce")
                col_idx += 3
            else:
                col_idx += 1

        return self._std_df(data_dict)

    def load_macro_predictions(self):
        df = pd.read_excel(self.xlsx, sheet_name="Macro Predictions", header=None)
        dates = pd.to_datetime(df.iloc[5:, 0])
        columns = {
            1: "USGG3M", 2: "USGG10YR", 3: "TEDSP", 4: "USYC2Y10",
            5: "VIX", 6: "VXEEM", 7: "NAPMPMI", 8: "INJCJC",
            9: "CONCCONF", 10: "CPI_YOY", 11: "HY_SPREAD",
            12: "LBUSTRUU", 13: "SPX", 14: "MXEF", 15: "MXWD"
        }
        data_dict = {"Date": dates.values}
        for col_idx, col_name in columns.items():
            if col_idx < len(df.columns):
                data_dict[col_name] = pd.to_numeric(df.iloc[5:, col_idx].values, errors="coerce")
        return self._std_df(data_dict)

    def load_macro2(self):
        df = pd.read_excel(self.xlsx, sheet_name="Macro2", header=None)
        dates = pd.to_datetime(df.iloc[5:, 0])
        columns = {1: "CFNAI", 2: "EPUCGLCP", 3: "M2WD", 4: "SKEW", 5: "WBBGWORL"}
        data_dict = {"Date": dates.values}
        for col_idx, col_name in columns.items():
            if col_idx < len(df.columns):
                data_dict[col_name] = pd.to_numeric(df.iloc[5:, col_idx].values, errors="coerce")
        return self._std_df(data_dict)

    def load_risk_free_rates(self):
        df = pd.read_excel(self.xlsx, sheet_name="Risk Free Rates", header=None)
        dates = pd.to_datetime(df.iloc[5:, 0])
        data_dict = {"Date": dates.values}

        col_idx = 1
        while col_idx < len(df.columns):
            ticker = df.iloc[3, col_idx]
            if pd.notna(ticker) and "Index" in str(ticker):
                ticker_name = str(ticker).replace(" Index", "")
                data_dict[ticker_name] = pd.to_numeric(df.iloc[5:, col_idx].values, errors="coerce")
            col_idx += 1

        return self._std_df(data_dict)

    def load_inflation(self):
        df = pd.read_excel(self.xlsx, sheet_name="Inflation index", header=None)
        dates = pd.to_datetime(df.iloc[5:, 0])
        data_dict = {"Date": dates.values}

        col_idx = 1
        while col_idx < len(df.columns):
            ticker = df.iloc[3, col_idx]
            if pd.notna(ticker) and "Index" in str(ticker):
                ticker_name = str(ticker).replace(" Index", "").replace("CPIYOY", "")
                data_dict[ticker_name] = pd.to_numeric(df.iloc[5:, col_idx].values, errors="coerce")
            col_idx += 1

        return self._std_df(data_dict)


# =========================
# 2) FACTOR BUILDER
# =========================

class FactorBuilder:
    """Build factor portfolios (same spirit as your current version)."""

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    @staticmethod
    def compute_returns(prices):
        return prices.pct_change()

    @staticmethod
    def rank_and_sort(signal: pd.DataFrame, n_groups=6) -> pd.DataFrame:
        def rank_row(row):
            valid = row.dropna()
            if len(valid) < n_groups:
                return pd.Series(index=row.index, dtype=float)
            ranks = valid.rank(method="average")
            n = len(valid)
            cutoff = n / n_groups
            w = pd.Series(index=row.index, dtype=float)
            for idx in valid.index:
                r = ranks[idx]
                if r <= cutoff:
                    w[idx] = -1.0 / (n / n_groups)
                elif r > n - cutoff:
                    w[idx] = 1.0 / (n / n_groups)
                else:
                    w[idx] = 0.0
            return w

        return signal.apply(rank_row, axis=1)

    @staticmethod
    def volatility_scale(returns: pd.Series, target_vol=0.10, lookback=36) -> pd.Series:
        r = returns.replace([np.inf, -np.inf], np.nan)
        vol = r.rolling(window=lookback, min_periods=12).std() * np.sqrt(12)
        vol = vol.clip(lower=0.01)
        scale = (target_vol / vol.shift(1)).clip(lower=0.1, upper=3.0)
        out = (r * scale).replace([np.inf, -np.inf], np.nan)
        return out

    def build_fx_factors(self):
        print("Building FX factors...")
        currencies = self.data_loader.load_currencies()
        risk_free = self.data_loader.load_risk_free_rates()

        bid_cols = [c for c in currencies.columns if c.endswith("_bid")]
        spot = currencies[bid_cols].rename(columns=lambda x: x.replace("_bid", ""))

        min_valid = 120
        spot = spot.loc[:, spot.notna().sum() > min_valid]
        spot_ret = self.compute_returns(spot).clip(-0.3, 0.3)

        rf_mapping = {
            "GBP": "BP0001M", "EUR": "EU0001M", "JPY": "JY0001M", "CHF": "SF0001M",
            "CAD": "CD0001M", "AUD": "RBACOR", "NZD": "NZOCRS", "SEK": "STIB1D", "NOK": "NIBOR01"
        }
        us_rate = risk_free.get("US0001M", pd.Series(index=risk_free.index))
        if us_rate.empty:
            us_rate = pd.Series(0.03, index=spot.index)

        fx_excess = pd.DataFrame(index=spot.index)
        carry_signal = pd.DataFrame(index=spot.index)

        for ccy in spot.columns:
            spot_r = spot_ret[ccy]
            if ccy in rf_mapping and rf_mapping[ccy] in risk_free.columns:
                foreign = risk_free[rf_mapping[ccy]].reindex(spot.index)
                us_al = us_rate.reindex(spot.index)
                rd = ((foreign - us_al) / 100 / 12).fillna(0)
                fx_excess[ccy] = (spot_r + rd).clip(-0.3, 0.3)
                carry_signal[ccy] = foreign - us_al
            else:
                fx_excess[ccy] = spot_r
                carry_signal[ccy] = np.nan

        if carry_signal.notna().sum().sum() < 1000:
            carry_signal = fx_excess.rolling(3).mean()

        fx_market = fx_excess.mean(axis=1)

        mom_signal = fx_excess.shift(1).rolling(11).apply(lambda x: (1 + x).prod() - 1, raw=False)
        mom_w = self.rank_and_sort(mom_signal)
        fx_mom = (mom_w.shift(1) * fx_excess).sum(axis=1)

        carry_w = self.rank_and_sort(carry_signal)
        fx_carry = (carry_w.shift(1) * fx_excess).sum(axis=1)

        fx_val_signal = -(spot / spot.rolling(60).mean() - 1).replace([np.inf, -np.inf], np.nan)
        val_w = self.rank_and_sort(fx_val_signal)
        fx_val = (val_w.shift(1) * fx_excess).sum(axis=1)

        return pd.DataFrame({
            "FX_Market": fx_market,
            "FX_Carry": fx_carry,
            "FX_Momentum": fx_mom,
            "FX_Value": fx_val
        })

    def build_commodity_factors(self):
        print("Building Commodity factors...")
        commodities = self.data_loader.load_commodities()

        front_cols = [c for c in commodities.columns if c.endswith("1")]
        second_cols = [c for c in commodities.columns if c.endswith("2")]
        front = commodities[front_cols].rename(columns=lambda x: x[:-1])
        second = commodities[second_cols].rename(columns=lambda x: x[:-1])

        common = list(set(front.columns) & set(second.columns))
        front, second = front[common], second[common]

        min_valid = 120
        valid = front.columns[front.notna().sum() > min_valid]
        front, second = front[valid], second[valid]

        ret = self.compute_returns(front).clip(-0.5, 0.5)
        mkt = ret.mean(axis=1)

        mom_signal = ret.shift(1).rolling(11).apply(lambda x: (1 + x).prod() - 1, raw=False)
        mom_w = self.rank_and_sort(mom_signal)
        mom = (mom_w.shift(1) * ret).sum(axis=1)

        roll_yield = (second / front - 1).clip(-1, 1)
        carry_signal = -roll_yield
        carry_w = self.rank_and_sort(carry_signal)
        carry = (carry_w.shift(1) * ret).sum(axis=1)

        fivey = (front / front.shift(60) - 1).replace([np.inf, -np.inf], np.nan)
        val_signal = -fivey
        val_w = self.rank_and_sort(val_signal)
        val = (val_w.shift(1) * ret).sum(axis=1)

        basis_mom_signal = roll_yield.diff(12)
        bm_w = self.rank_and_sort(basis_mom_signal)
        bm = (bm_w.shift(1) * ret).sum(axis=1)

        return pd.DataFrame({
            "Commo_Market": mkt,
            "Commo_Carry": carry,
            "Commo_Momentum": mom,
            "Commo_Value": val,
            "Commo_BasisMom": bm
        })

    def build_fixed_income_factors(self):
        print("Building Fixed Income factors...")
        rates = self.data_loader.load_rates_curves()

        yield_cols = [c for c in rates.columns if ("10" in c or "10YR" in c.upper())]
        yields = rates[yield_cols] if yield_cols else rates.iloc[:, :10]

        min_valid = 120
        yields = yields.loc[:, yields.notna().sum() > min_valid]

        duration = 7
        dy = yields.diff() / 100
        coupon = yields.shift(1) / 100 / 12
        bond_ret = (coupon - duration * dy).clip(-0.2, 0.2)

        mkt = bond_ret.mean(axis=1)

        mom_signal = bond_ret.shift(1).rolling(11).apply(lambda x: (1 + x).prod() - 1, raw=False)
        mom_w = self.rank_and_sort(mom_signal)
        mom = (mom_w.shift(1) * bond_ret).sum(axis=1)

        yield_5y_cols = [c for c in rates.columns if ("5" in c and "10" not in c)]
        if yield_5y_cols:
            yields_5y = rates[yield_5y_cols]
            carry_signal = pd.DataFrame(index=yields.index)
            for c10 in yields.columns:
                country = c10[:2]
                match_5 = [c for c in yields_5y.columns if country in c]
                if match_5:
                    carry_signal[c10] = yields[c10] - yields_5y[match_5[0]]
            if carry_signal.empty:
                carry_signal = yields
        else:
            carry_signal = yields

        carry_w = self.rank_and_sort(carry_signal)
        carry = (carry_w.shift(1) * bond_ret).sum(axis=1)

        val_signal = (yields / yields.rolling(60).mean() - 1).replace([np.inf, -np.inf], np.nan)
        val_w = self.rank_and_sort(val_signal)
        val = (val_w.shift(1) * bond_ret).sum(axis=1)

        return pd.DataFrame({
            "FI_Market": mkt,
            "FI_Carry": carry,
            "FI_Momentum": mom,
            "FI_Value": val
        })

    def build_equity_factors(self):
        print("Building Equity factors...")
        stocks = self.data_loader.load_stock_indices()

        tri_cols = [c for c in stocks.columns if c.endswith("_tri")]
        price_cols = [c for c in stocks.columns if c.endswith("_price")]
        mcap_cols = [c for c in stocks.columns if c.endswith("_mcap")]

        if tri_cols:
            prices = stocks[tri_cols].rename(columns=lambda x: x.replace("_tri", ""))
        else:
            prices = stocks[price_cols].rename(columns=lambda x: x.replace("_price", ""))

        min_valid = 120
        prices = prices.loc[:, prices.notna().sum() > min_valid]

        ret = self.compute_returns(prices).clip(-0.5, 0.5)
        mkt = ret.mean(axis=1)

        mom_signal = ret.shift(1).rolling(11).apply(lambda x: (1 + x).prod() - 1, raw=False)
        mom_w = self.rank_and_sort(mom_signal)
        mom = (mom_w.shift(1) * ret).sum(axis=1)

        mcaps = stocks[[c for c in mcap_cols if c.replace("_mcap", "") in prices.columns]].rename(
            columns=lambda x: x.replace("_mcap", "")
        )
        if len(mcaps.columns) > 0:
            mcaps = mcaps[prices.columns.intersection(mcaps.columns)]
            size_signal = (-np.log(mcaps.replace(0, np.nan))).replace([np.inf, -np.inf], np.nan)
            size_w = self.rank_and_sort(size_signal)
            size = (size_w.shift(1) * ret[size_signal.columns]).sum(axis=1)
        else:
            size = pd.Series(0.0, index=ret.index)

        value_signal = (prices.rolling(60).mean() / prices - 1).replace([np.inf, -np.inf], np.nan)
        value_w = self.rank_and_sort(value_signal)
        val = (value_w.shift(1) * ret).sum(axis=1)

        return pd.DataFrame({
            "Eq_Market": mkt,
            "Eq_Momentum": mom,
            "Eq_Size": size,
            "Eq_Value": val
        })

    def build_all_factors(self):
        fx = self.build_fx_factors()
        com = self.build_commodity_factors()
        fi = self.build_fixed_income_factors()
        eq = self.build_equity_factors()

        all_fac = pd.concat([fx, com, fi, eq], axis=1)

        print("Applying volatility scaling...")
        scaled = pd.DataFrame(index=all_fac.index)
        for c in all_fac.columns:
            scaled[c] = self.volatility_scale(all_fac[c])

        return scaled


# =========================
# 3) PREDICTORS
# =========================

class PredictorBuilder:
    """Build and standardize macro and TS predictors (same spirit)."""

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    @staticmethod
    def standardize(x: pd.Series, lookback=120) -> pd.Series:
        mu = x.expanding(min_periods=lookback).mean()
        sd = x.expanding(min_periods=lookback).std()
        return (x - mu) / sd

    def build_all_predictors(self, factors: pd.DataFrame) -> pd.DataFrame:
        print("Building predictor variables...")
        macro = self.data_loader.load_macro_predictions()
        macro2 = self.data_loader.load_macro2()

        predictors = pd.DataFrame(index=factors.index)
        predictors["CFNAI"] = self.standardize(macro2.get("CFNAI", pd.Series(index=factors.index)))
        predictors["Inflation"] = self.standardize(macro.get("CPI_YOY", pd.Series(index=factors.index)))
        predictors["ShortRate"] = self.standardize(macro.get("USGG3M", pd.Series(index=factors.index)))
        predictors["YieldCurve"] = self.standardize(macro.get("USYC2Y10", pd.Series(index=factors.index)))
        predictors["VIX"] = self.standardize(macro.get("VIX", pd.Series(index=factors.index)))

        ted = macro.get("TEDSP", pd.Series(index=factors.index))
        if ted.notna().sum() > 60:
            predictors["TED"] = self.standardize(ted)
        else:
            hy = macro.get("HY_SPREAD", pd.Series(index=factors.index))
            predictors["TED"] = self.standardize(hy) if hy.notna().sum() > 60 else self.standardize(
                macro.get("USGG3M", pd.Series(index=factors.index)).diff(3)
            )

        predictors["EPU"] = self.standardize(macro2.get("EPUCGLCP", pd.Series(index=factors.index)))
        predictors["BudgetBal"] = self.standardize(macro2.get("WBBGWORL", pd.Series(index=factors.index)), lookback=36)
        predictors["SKEW"] = self.standardize(macro2.get("SKEW", pd.Series(index=factors.index)))

        m2 = macro2.get("M2WD", pd.Series(index=factors.index))
        predictors["M2Growth"] = self.standardize(m2.pct_change(12))

        # TS signals built from factors directly
        cum12 = factors.rolling(12).apply(lambda x: (1 + x).prod() - 1)
        predictors["TS_Mom"] = cum12.mean(axis=1)
        predictors["TS_Vol"] = -(factors.rolling(12).std() * np.sqrt(12)).mean(axis=1)
        # conservative publication lag: shift macro signals by 1 month
        macro_cols = ["CFNAI", "Inflation", "ShortRate", "YieldCurve", "VIX", "TED", "EPU", "BudgetBal", "SKEW",
                      "M2Growth"]
        predictors[macro_cols] = predictors[macro_cols].shift(1)

        return predictors.reindex(factors.index)


# =========================
# 4) BAYESIAN PREDICTOR (UNIVARIATE)
# =========================

class BayesianPredictor:
    """Simple skeptical-prior predictive regression for each factor separately."""

    def __init__(self, prior_r2=0.01):
        self.prior_r2 = prior_r2
        self.predictor_shrinkage = {  # pragmatic shrinkages (your style)
            "CFNAI": 0.08, "Inflation": 0.08, "ShortRate": 0.08, "YieldCurve": 0.08,
            "VIX": 0.08, "TED": 0.08, "EPU": 0.08, "BudgetBal": 0.08,
            "SKEW": 0.08, "M2Growth": 0.08, "TS_Mom": 0.08, "TS_Vol": 0.08,
        }
        self.current_predictor = None

    def fit_predict(self, y: pd.Series, x: pd.Series, min_obs=60) -> pd.Series:
        T = len(y)
        pred = pd.Series(index=y.index, dtype=float)

        for t in range(min_obs, T):
            y_t = y.iloc[:t].dropna()
            x_t = x.iloc[:t].dropna()
            idx = y_t.index.intersection(x_t.index)
            if len(idx) < min_obs:
                continue

            yy = y_t.loc[idx].values
            xx = x_t.loc[idx].values

            xdm = xx - xx.mean()
            ydm = yy - yy.mean()
            var_x = np.var(xdm)
            if var_x < 1e-10:
                continue

            beta_ols = np.sum(xdm * ydm) / np.sum(xdm ** 2)
            resid = yy - (yy.mean() + beta_ols * xdm)
            sigma2 = np.var(resid) if np.var(resid) > 1e-12 else 1e-12

            var_y = np.var(yy)
            prior_var_beta = self.prior_r2 * var_y / var_x if var_y > 1e-12 else 1e-6

            ols_prec = np.sum(xdm ** 2) / sigma2
            prior_prec = 1 / prior_var_beta
            post_prec = ols_prec + prior_prec
            post_var = 1 / post_prec
            post_mean = post_var * (ols_prec * beta_ols)

            beta = post_mean
            alpha = yy.mean()

            if pd.notna(x.iloc[t]):
                shrink = self.predictor_shrinkage.get(self.current_predictor, 0.3)
                beta = beta * shrink
                pred.iloc[t] = alpha + beta * (x.iloc[t] - xx.mean())

        return pred

    def compute_all_predictions(self, factors: pd.DataFrame, predictors: pd.DataFrame, min_obs=60):
        print("Computing Bayesian predictions...")
        all_pred = {}

        for p in predictors.columns:
            print(f"  Processing predictor: {p}")
            self.current_predictor = p
            pred_df = pd.DataFrame(index=factors.index)
            for fac in factors.columns:
                pred_df[fac] = self.fit_predict(factors[fac], predictors[p], min_obs=min_obs)
            all_pred[p] = pred_df

        return all_pred


# =========================
# 5) EXTENSION: ROLLING VAR(1)
# =========================

class VARPredictor:
    """
    Rolling VAR(1) on factor returns.
    Produces a DataFrame of predicted next-month factor returns, aligned like your other 'predictions'.

    Options:
    - ridge_alpha > 0: ridge shrinkage on coefficients (stabilizes when factors are noisy/collinear)
    """

    def __init__(self, lags=1, lookback=120, min_obs=60, ridge_alpha=0.0):
        assert lags == 1, "This implementation is VAR(1) for simplicity (lags=1)."
        self.lags = lags
        self.lookback = lookback
        self.min_obs = min_obs
        self.ridge_alpha = ridge_alpha

    def _fit_var1(self, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Y shape: (T, n)
        Model: y_t = c + A y_{t-1} + e_t
        Returns: c (n,), A (n,n)
        """
        T, n = Y.shape
        y = Y[1:, :]          # (T-1, n)
        x = Y[:-1, :]         # (T-1, n)

        X = np.hstack([np.ones((T-1, 1)), x])  # (T-1, 1+n)

        # ridge on coefficients (excluding intercept) if ridge_alpha>0
        if self.ridge_alpha > 0:
            # solve (X'X + alpha*diag([0,1,1,...])) B = X'y
            XtX = X.T @ X
            reg = np.diag([0.0] + [1.0] * n) * self.ridge_alpha
            B = np.linalg.solve(XtX + reg, X.T @ y)  # (1+n, n)
        else:
            B = np.linalg.lstsq(X, y, rcond=None)[0]  # (1+n, n)

        c = B[0, :]           # (n,)
        A = B[1:, :].T        # careful: B rows are regressors; want A mapping y_{t-1} -> y_t
        # Here B[1:,:] is (n, n) with columns = equations; so A should be (n,n) with row i eqn.
        # A = B[1:,:].T gives (n,n) correct: yhat = c + x @ B[1:,:]  => yhat = c + A @ x^T if A=B[1:,:].T
        return c, A

    def predict(self, factors: pd.DataFrame) -> pd.DataFrame:
        """
        Returns DataFrame with same index/columns as factors:
        pred.iloc[t] is the forecast for factors at t (using info up to t-1),
        consistent with your backtest that uses predictions.iloc[t-1].
        """
        fac = factors.copy()
        pred = pd.DataFrame(index=fac.index, columns=fac.columns, dtype=float)

        # Use expanding start with a rolling window (lookback) for stability
        for t in range(1, len(fac)):
            start = max(0, t - self.lookback)
            window = fac.iloc[start:t].dropna()
            if len(window) < self.min_obs:
                continue

            Y = window.values.astype(float)
            # if any columns are all-NaN inside window after dropna row-wise, handle via column drop
            # simpler: require no NaNs row-wise (dropna did that)
            if Y.shape[0] < self.min_obs:
                continue

            c, A = self._fit_var1(Y)
            y_last = fac.iloc[t-1].values.astype(float)

            # if current row has NaNs, replace with 0 for forecast input (pragmatic)
            y_last = np.nan_to_num(y_last, nan=0.0)

            yhat = c + (A @ y_last)
            pred.iloc[t] = yhat

        return pred


# =========================
# 6) ALLOCATOR
# =========================

class BlackLittermanAllocator:
    """Utility/TE-like optimizer used as 'allocation engine' (same style as your current one)."""

    def __init__(self, risk_aversion=5.0, view_confidence=0.50, transaction_cost=0.001):
        self.risk_aversion = risk_aversion
        self.view_confidence = view_confidence
        self.transaction_cost = transaction_cost
        self.ir_confidence_scale = {
            "CFNAI": 0.30, "Inflation": 0.13, "ShortRate": 0.16, "YieldCurve": 0.13,
            "VIX": 0.10, "TED": 0.08, "EPU": 0.16, "BudgetBal": 0.40,
            "SKEW": 0.20, "M2Growth": 0.20, "TS_Mom": 0.12, "TS_Vol": 0.08,
            "VAR": 0.20,          # <-- scale for VAR extension (tunable)
            "ENS_BayesVAR": 0.20  # <-- scale for ensemble
        }
        self.current_predictor = None

    @staticmethod
    def compute_equilibrium_weights(n_assets):
        return np.ones(n_assets) / n_assets

    def optimize_weights(self, mu, cov, w_bench):
        n = len(mu)

        if mu is None or cov is None:
            return w_bench

        mu = np.nan_to_num(np.asarray(mu), nan=0.0)
        cov = np.nan_to_num(np.asarray(cov), nan=0.0)

        scale = self.ir_confidence_scale.get(self.current_predictor, 1.0)
        mu = mu * self.view_confidence * scale

        # PSD fix
        eigvals = np.linalg.eigvals(cov)
        min_eig = np.min(eigvals.real)
        if min_eig < 0:
            cov = cov + (-min_eig + 1e-3) * np.eye(n)

        lam = self.risk_aversion

        def objective(w):
            a = w - w_bench
            exp_alpha = a @ mu
            te_var = a.T @ cov @ a
            util = exp_alpha - (lam / 2) * te_var
            return -util

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 0.30) for _ in range(n)]
        x0 = w_bench.copy()

        try:
            res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints,
                           options={"maxiter": 200})
            return res.x if res.success else w_bench
        except Exception:
            return w_bench

    def run_backtest(self, factors: pd.DataFrame, predictions: pd.DataFrame, lookback=60):
        print(f"Running backtest for {self.current_predictor} ...")
        T = len(factors)
        n = factors.shape[1]
        w_bench = self.compute_equilibrium_weights(n)

        strat = pd.Series(index=factors.index, dtype=float)
        bench = pd.Series(index=factors.index, dtype=float)
        prev_w = w_bench.copy()

        for t in range(lookback, T):
            hist = factors.iloc[t - lookback:t]
            cov = hist.cov().values

            # use prediction at t-1 to set weights for month t
            mu = predictions.iloc[t - 1].values if t - 1 >= 0 else np.zeros(n)

            w = self.optimize_weights(mu, cov, w_bench)

            r_t = factors.iloc[t].values
            gross = w @ r_t

            turnover = np.sum(np.abs(w - prev_w))
            tc = turnover * self.transaction_cost

            strat.iloc[t] = gross - tc
            bench.iloc[t] = w_bench @ r_t

            prev_w = w.copy()

        return strat, bench


# =========================
# 7) PERFORMANCE
# =========================

class PerformanceAnalyzer:
    @staticmethod
    def annualized_return(r):
        r = r.dropna()
        if len(r) < 12:
            return np.nan
        total = (1 + r).prod()
        yrs = len(r) / 12
        if yrs <= 0 or total <= 0:
            return np.nan
        return total ** (1 / yrs) - 1

    @staticmethod
    def annualized_volatility(r):
        r = r.dropna()
        if len(r) < 12:
            return np.nan
        return r.std() * np.sqrt(12)

    @staticmethod
    def sharpe_ratio(r, rf=0.0):
        ex = r - rf / 12
        ar = PerformanceAnalyzer.annualized_return(ex)
        av = PerformanceAnalyzer.annualized_volatility(ex)
        if pd.isna(av) or av == 0:
            return np.nan
        return ar / av

    @staticmethod
    def information_ratio(strat, bench):
        a = (strat - bench).dropna()
        if len(a) < 12:
            return np.nan
        mean = a.mean() * 12
        te = a.std() * np.sqrt(12)
        if te == 0 or pd.isna(te):
            return np.nan
        return mean / te

    @staticmethod
    def max_drawdown(r):
        r = r.dropna()
        if len(r) < 2:
            return np.nan
        cr = (1 + r).cumprod()
        peak = cr.cummax()
        dd = cr / peak - 1
        return dd.min()

    @staticmethod
    def t_statistic(r):
        r = r.dropna()
        if len(r) < 2:
            return 0.0
        m = r.mean()
        s = r.std()
        n = len(r)
        if s == 0 or pd.isna(s):
            return 0.0
        return m / (s / np.sqrt(n))

    @staticmethod
    def compute_all_metrics(strat, bench, name="Strategy"):
        return {
            "Name": name,
            "Ann. Return (%)": PerformanceAnalyzer.annualized_return(strat) * 100,
            "Ann. Volatility (%)": PerformanceAnalyzer.annualized_volatility(strat) * 100,
            "Sharpe Ratio": PerformanceAnalyzer.sharpe_ratio(strat),
            "Information Ratio": PerformanceAnalyzer.information_ratio(strat, bench),
            "Max Drawdown (%)": PerformanceAnalyzer.max_drawdown(strat) * 100,
            "t-stat": PerformanceAnalyzer.t_statistic(strat - bench),
        }

    @staticmethod
    def holm_correction(p_values):
        n = len(p_values)
        idx = np.argsort(p_values)
        sp = np.array(p_values)[idx]
        adj = [min(p * (n - i), 1.0) for i, p in enumerate(sp)]
        out = np.zeros(n)
        for i, j in enumerate(idx):
            out[j] = adj[i]
        return out


# =========================
# 8) RUN
# =========================

def run_replication_with_var_extension():
    print("=" * 60)
    print("REPLICATION + EXTENSION: Rolling VAR(1)")
    print("=" * 60)
    print()

    print("Step 1: Loading data...")
    loader = DataLoader("DataGestionQuant.xlsx")

    print("\nStep 2: Building factors...")
    factor_builder = FactorBuilder(loader)
    factors = factor_builder.build_all_factors()
    print(f"  Factors built: {list(factors.columns)}")
    print(f"  Date range: {factors.index.min()} to {factors.index.max()}")
    print(f"  Observations: {len(factors)}")

    print("\nStep 3: Building predictors...")
    predictor_builder = PredictorBuilder(loader)
    predictors = predictor_builder.build_all_predictors(factors)
    print(f"  Predictors built: {list(predictors.columns)}")

    print("\nStep 4: Computing Bayesian predictions (macro -> factor)...")
    bayes = BayesianPredictor(prior_r2=0.01)
    bayes_predictions = bayes.compute_all_predictions(factors, predictors, min_obs=60)

    print("\nStep 5 (EXT): Computing Rolling VAR(1) predictions (factor -> factor)...")
    var_model = VARPredictor(lags=1, lookback=120, min_obs=60, ridge_alpha=1e-3)
    var_predictions = var_model.predict(factors)

    # Optional: ensemble Bayes+VAR (simple average of standardized signals)
    # We take one representative Bayesian predictor (e.g. CFNAI) or average all predictors.
    # Here: average across Bayesian predictions over all predictors, then average with VAR.
    print("Building ensemble predictions (Bayes avg + VAR)...")
    bayes_avg = pd.DataFrame(0.0, index=factors.index, columns=factors.columns)
    k = 0
    for p, dfp in bayes_predictions.items():
        bayes_avg = bayes_avg.add(dfp.fillna(0.0), fill_value=0.0)
        k += 1
    if k > 0:
        bayes_avg = bayes_avg / k

    ens_predictions = 0.5 * bayes_avg.fillna(0.0) + 0.5 * var_predictions.fillna(0.0)

    print("\nStep 6: Running backtests (Bayes predictors + VAR + Ensemble)...")
    allocator = BlackLittermanAllocator(risk_aversion=5.0, view_confidence=0.50, transaction_cost=0.001)

    results = {}

    # Benchmark from any run
    bench_returns = None

    # Bayes runs
    for pred_name, pred_df in bayes_predictions.items():
        allocator.current_predictor = pred_name
        strat, bench = allocator.run_backtest(factors, pred_df, lookback=60)
        results[f"BL.{pred_name}"] = {"strategy": strat, "benchmark": bench}
        if bench_returns is None:
            bench_returns = bench

    # VAR run
    allocator.current_predictor = "VAR"
    strat_var, bench_var = allocator.run_backtest(factors, var_predictions, lookback=60)
    results["BL.VAR"] = {"strategy": strat_var, "benchmark": bench_var}

    # Ensemble run
    allocator.current_predictor = "ENS_BayesVAR"
    strat_ens, bench_ens = allocator.run_backtest(factors, ens_predictions, lookback=60)
    results["BL.ENS_BayesVAR"] = {"strategy": strat_ens, "benchmark": bench_ens}

    print("\nStep 7: Performance summary...")
    summary = []
    # EW benchmark metrics
    bm = PerformanceAnalyzer.compute_all_metrics(bench_returns, bench_returns, "EW Benchmark")
    summary.append(bm)

    for name, res in results.items():
        m = PerformanceAnalyzer.compute_all_metrics(res["strategy"], res["benchmark"], name)
        summary.append(m)

    results_df = pd.DataFrame(summary).set_index("Name")
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(results_df.round(2).to_string())
    print()

    # Significance & Holm on active t-stats
    print("=" * 80)
    print("STATISTICAL SIGNIFICANCE + HOLM")
    print("=" * 80)

    strat_names = [n for n in results_df.index if n != "EW Benchmark"]
    t_stats = results_df.loc[strat_names, "t-stat"].values
    df_dof = len(bench_returns.dropna()) - 2
    pvals = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=df_dof))
    adj = PerformanceAnalyzer.holm_correction(pvals)

    survivors = []
    for i, n in enumerate(strat_names):
        if adj[i] < 0.05:
            survivors.append((n, adj[i], results_df.loc[n, "Information Ratio"], results_df.loc[n, "t-stat"]))

    print(f"Survivors (adjusted p < 0.05): {len(survivors)}")
    for n, ap, ir, ts in survivors:
        print(f"  - {n}: adj p={ap:.4f}, IR={ir:.2f}, t={ts:.2f}")

    # Save outputs
    results_df.to_csv("replication_results_with_VAR.csv")
    factors.to_csv("factor_returns.csv")
    predictors.to_csv("predictors.csv")
    var_predictions.to_csv("var_predictions.csv")
    ens_predictions.to_csv("ensemble_predictions.csv")

    print("\nSaved:")
    print(" - replication_results_with_VAR.csv")
    print(" - factor_returns.csv")
    print(" - predictors.csv")
    print(" - var_predictions.csv")
    print(" - ensemble_predictions.csv")

    return results_df, factors, predictors, results, var_predictions, ens_predictions


if __name__ == "__main__":
    run_replication_with_var_extension()
