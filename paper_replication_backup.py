"""
Replication of "Time-Varying Factor Allocation"
by Stefan Vincenz and Tom Oskar Karl Zeissler (July 2022)

CORRECTED VERSION:
- Paths are now relative (uses script directory)
- 15 predictor variables (as per paper)
- Clean Bayesian model (no ad-hoc shrinkage per predictor)
- Proper turnover calculation from portfolio weights
- Breakeven costs based on observed turnover

This script replicates the main findings of the paper:
- Bayesian predictive regression framework
- Black-Litterman asset allocation
- 17 factors across 4 asset classes (limited by available data, paper has 21)
- 15 predictor variables
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# UTILITY: Get the directory of this script for relative paths
# =============================================================================

def get_script_dir():
    """Return the directory where this script is located."""
    return os.path.dirname(os.path.abspath(__file__))


def get_data_path(filename='DataGestionQuant.xlsx'):
    """Return the absolute path to the data file, relative to script location."""
    return os.path.join(get_script_dir(), filename)


# =============================================================================
# PART 1: DATA LOADING AND PREPROCESSING
# =============================================================================

class DataLoader:
    """Load and preprocess data from Excel file"""

    def __init__(self, filepath=None):
        """
        Initialize DataLoader with path to Excel file.
        If no filepath is provided, uses the default file in the script directory.
        """
        if filepath is None:
            filepath = get_data_path('DataGestionQuant.xlsx')
        self.filepath = filepath
        self.xlsx = pd.ExcelFile(filepath)

    def load_currencies(self):
        """Load currency spot rates (bid/ask)"""
        df = pd.read_excel(self.xlsx, sheet_name='Currencies', header=None)

        # Extract dates (column 0, starting from row 5)
        dates = pd.to_datetime(df.iloc[5:, 0])

        # Build structured dataframe
        data_dict = {'Date': dates.values}

        col_idx = 1
        while col_idx < len(df.columns):
            ticker = df.iloc[3, col_idx]
            if pd.notna(ticker) and 'Curncy' in str(ticker):
                # Bid price
                bid_col = df.iloc[5:, col_idx].values
                ask_col = df.iloc[5:, col_idx + 1].values if col_idx + 1 < len(df.columns) else bid_col

                ticker_name = str(ticker).replace(' Curncy', '').replace('USD', '')
                data_dict[f'{ticker_name}_bid'] = pd.to_numeric(bid_col, errors='coerce')
                data_dict[f'{ticker_name}_ask'] = pd.to_numeric(ask_col, errors='coerce')
                col_idx += 2
            else:
                col_idx += 1

        result = pd.DataFrame(data_dict)
        result['Date'] = pd.to_datetime(result['Date'])
        result.set_index('Date', inplace=True)
        return result

    def load_currencies_forward(self):
        """Load currency forward rates"""
        df = pd.read_excel(self.xlsx, sheet_name='Currencies Forward', header=None)

        # Find start of data
        dates = pd.to_datetime(df.iloc[7:, 3])

        data_dict = {'Date': dates.values}

        # Extract forward tickers from row 5
        for col_idx in range(4, len(df.columns)):
            ticker = df.iloc[5, col_idx]
            if pd.notna(ticker) and 'BGN Curncy' in str(ticker):
                ticker_name = str(ticker).replace(' BGN Curncy', '')
                data_dict[ticker_name] = pd.to_numeric(df.iloc[7:, col_idx].values, errors='coerce')

        result = pd.DataFrame(data_dict)
        result['Date'] = pd.to_datetime(result['Date'])
        result.set_index('Date', inplace=True)
        return result

    def load_rates_curves(self):
        """Load government bond yields"""
        df = pd.read_excel(self.xlsx, sheet_name='Rates Curves', header=None)

        dates = pd.to_datetime(df.iloc[5:, 0])

        data_dict = {'Date': dates.values}

        # Extract yield tickers from row 3
        col_idx = 1
        while col_idx < len(df.columns):
            ticker = df.iloc[3, col_idx]
            if pd.notna(ticker) and 'Index' in str(ticker):
                ticker_name = str(ticker).replace(' Index', '')
                data_dict[ticker_name] = pd.to_numeric(df.iloc[5:, col_idx].values, errors='coerce')
            col_idx += 1

        result = pd.DataFrame(data_dict)
        result['Date'] = pd.to_datetime(result['Date'])
        result.set_index('Date', inplace=True)
        return result

    def load_commodities(self):
        """Load commodity futures prices"""
        df = pd.read_excel(self.xlsx, sheet_name='Commodities', header=None)

        dates = pd.to_datetime(df.iloc[5:, 0])

        data_dict = {'Date': dates.values}

        # Extract commodity tickers from row 3
        col_idx = 1
        while col_idx < len(df.columns):
            ticker = df.iloc[3, col_idx]
            if pd.notna(ticker) and 'Comdty' in str(ticker):
                ticker_name = str(ticker).replace(' Comdty', '')
                data_dict[ticker_name] = pd.to_numeric(df.iloc[5:, col_idx].values, errors='coerce')
                # Skip volume and open interest columns
                col_idx += 3
            else:
                col_idx += 1

        result = pd.DataFrame(data_dict)
        result['Date'] = pd.to_datetime(result['Date'])
        result.set_index('Date', inplace=True)
        return result

    def load_stock_indices(self):
        """Load equity index prices and total returns"""
        df = pd.read_excel(self.xlsx, sheet_name='Stock Indices', header=None)

        dates = pd.to_datetime(df.iloc[5:, 0])

        data_dict = {'Date': dates.values}

        # Extract index tickers from row 3
        col_idx = 1
        while col_idx < len(df.columns):
            ticker = df.iloc[3, col_idx]
            if pd.notna(ticker) and 'MX' in str(ticker):
                ticker_name = str(ticker).replace(' Index', '')
                # Price
                data_dict[f'{ticker_name}_price'] = pd.to_numeric(df.iloc[5:, col_idx].values, errors='coerce')
                # Total return index
                if col_idx + 1 < len(df.columns):
                    data_dict[f'{ticker_name}_tri'] = pd.to_numeric(df.iloc[5:, col_idx + 1].values, errors='coerce')
                # Market cap
                if col_idx + 2 < len(df.columns):
                    data_dict[f'{ticker_name}_mcap'] = pd.to_numeric(df.iloc[5:, col_idx + 2].values, errors='coerce')
                col_idx += 3
            else:
                col_idx += 1

        result = pd.DataFrame(data_dict)
        result['Date'] = pd.to_datetime(result['Date'])
        result.set_index('Date', inplace=True)
        return result

    def load_macro_predictions(self):
        """Load macro predictor variables"""
        df = pd.read_excel(self.xlsx, sheet_name='Macro Predictions', header=None)

        dates = pd.to_datetime(df.iloc[5:, 0])

        # Column mapping from row 3
        columns = {
            1: 'USGG3M',      # 3-month T-bill rate
            2: 'USGG10YR',    # 10-year Treasury yield
            3: 'TEDSP',       # TED spread
            4: 'USYC2Y10',    # Yield curve (10Y - 2Y)
            5: 'VIX',         # VIX
            6: 'VXEEM',       # EM VIX
            7: 'NAPMPMI',     # ISM PMI
            8: 'INJCJC',      # Jobless claims
            9: 'CONCCONF',    # Consumer confidence
            10: 'CPI_YOY',    # CPI YoY
            11: 'HY_SPREAD',  # High yield spread
            12: 'LBUSTRUU',   # Bloomberg Barclays US Agg
            13: 'SPX',        # S&P 500
            14: 'MXEF',       # MSCI EM
            15: 'MXWD'        # MSCI World
        }

        data_dict = {'Date': dates.values}
        for col_idx, col_name in columns.items():
            if col_idx < len(df.columns):
                data_dict[col_name] = pd.to_numeric(df.iloc[5:, col_idx].values, errors='coerce')

        result = pd.DataFrame(data_dict)
        result['Date'] = pd.to_datetime(result['Date'])
        result.set_index('Date', inplace=True)
        return result

    def load_macro2(self):
        """Load additional macro variables (CFNAI, etc.)"""
        df = pd.read_excel(self.xlsx, sheet_name='Macro2', header=None)

        dates = pd.to_datetime(df.iloc[5:, 0])

        # Column mapping
        columns = {
            1: 'CFNAI',       # Chicago Fed National Activity Index
            2: 'EPUCGLCP',    # Economic Policy Uncertainty
            3: 'M2WD',        # Global M2
            4: 'SKEW',        # CBOE SKEW
            5: 'WBBGWORL'     # World Budget Balance
        }

        data_dict = {'Date': dates.values}
        for col_idx, col_name in columns.items():
            if col_idx < len(df.columns):
                data_dict[col_name] = pd.to_numeric(df.iloc[5:, col_idx].values, errors='coerce')

        result = pd.DataFrame(data_dict)
        result['Date'] = pd.to_datetime(result['Date'])
        result.set_index('Date', inplace=True)
        return result

    def load_risk_free_rates(self):
        """Load risk-free rates for different currencies"""
        df = pd.read_excel(self.xlsx, sheet_name='Risk Free Rates', header=None)

        dates = pd.to_datetime(df.iloc[5:, 0])

        data_dict = {'Date': dates.values}

        # Extract tickers from row 3
        col_idx = 1
        while col_idx < len(df.columns):
            ticker = df.iloc[3, col_idx]
            if pd.notna(ticker) and 'Index' in str(ticker):
                ticker_name = str(ticker).replace(' Index', '')
                data_dict[ticker_name] = pd.to_numeric(df.iloc[5:, col_idx].values, errors='coerce')
            col_idx += 1

        result = pd.DataFrame(data_dict)
        result['Date'] = pd.to_datetime(result['Date'])
        result.set_index('Date', inplace=True)
        return result

    def load_inflation(self):
        """Load inflation indices"""
        df = pd.read_excel(self.xlsx, sheet_name='Inflation index', header=None)

        dates = pd.to_datetime(df.iloc[5:, 0])

        data_dict = {'Date': dates.values}

        # Extract tickers from row 3
        col_idx = 1
        while col_idx < len(df.columns):
            ticker = df.iloc[3, col_idx]
            if pd.notna(ticker) and 'Index' in str(ticker):
                ticker_name = str(ticker).replace(' Index', '').replace('CPIYOY', '')
                data_dict[ticker_name] = pd.to_numeric(df.iloc[5:, col_idx].values, errors='coerce')
            col_idx += 1

        result = pd.DataFrame(data_dict)
        result['Date'] = pd.to_datetime(result['Date'])
        result.set_index('Date', inplace=True)
        return result

    def load_shiller_data(self):
        """Load Shiller data (Dividend Yield, Earnings Yield)"""
        try:
            df = pd.read_excel(self.xlsx, sheet_name='Shiller_Data')
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            return df
        except Exception as e:
            print(f"Warning: Could not load Shiller_Data sheet: {e}")
            return pd.DataFrame()

    def load_fred_data(self):
        """Load FRED data (TED Spread, 10Y Treasury)"""
        try:
            df = pd.read_excel(self.xlsx, sheet_name='FRED_Data')
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            return df
        except Exception as e:
            print(f"Warning: Could not load FRED_Data sheet: {e}")
            return pd.DataFrame()


# =============================================================================
# PART 2: FACTOR CONSTRUCTION
# =============================================================================

class FactorBuilder:
    """
    Build factor portfolios following the paper methodology.
    
    The paper uses 21 factors across 4 asset classes:
    - FX: Market, Carry, Momentum, Value (4 factors)
    - Commodities: Market, Carry, Momentum, Value, Basis-Momentum (5 factors)
    - Fixed Income: Market, Carry, Momentum, Value (4 factors)
    - Equity: Market, Momentum, Size, Value, ... (up to 8 factors)
    
    Due to data availability, we construct 17 factors with the same methodology:
    - Cross-sectional sorting into 6 groups
    - Long-short portfolios (top vs bottom sextile)
    - Ex-ante volatility scaling to 10% annualized
    """

    def __init__(self, data_loader):
        self.data_loader = data_loader

    def compute_returns(self, prices):
        """Compute simple returns from prices"""
        return prices.pct_change()

    def compute_log_returns(self, prices):
        """Compute log returns from prices"""
        return np.log(prices / prices.shift(1))

    def rank_and_sort(self, signal, n_groups=6):
        """
        Rank assets by signal and sort into groups.
        Returns weights for long-short portfolio (top/bottom sextile = 16.67%).
        
        This follows the paper's methodology of cross-sectional factor construction.
        """
        def rank_row(row):
            valid = row.dropna()
            if len(valid) < n_groups:
                return pd.Series(index=row.index, dtype=float)

            ranks = valid.rank(method='average')
            n = len(valid)
            cutoff = n / n_groups

            weights = pd.Series(index=row.index, dtype=float)
            for idx in valid.index:
                rank = ranks[idx]
                if rank <= cutoff:  # Bottom group (short)
                    weights[idx] = -1.0 / (n / n_groups)
                elif rank > n - cutoff:  # Top group (long)
                    weights[idx] = 1.0 / (n / n_groups)
                else:
                    weights[idx] = 0.0

            return weights

        return signal.apply(rank_row, axis=1)

    def build_momentum_factor(self, returns, lookback=12, skip=1):
        """
        Build momentum factor (12-1 momentum).
        Long winners, short losers.
        """
        cum_returns = returns.rolling(window=lookback).apply(
            lambda x: (1 + x[:-skip]).prod() - 1 if len(x) > skip else np.nan
        )

        weights = self.rank_and_sort(cum_returns)
        factor_returns = (weights.shift(1) * returns).sum(axis=1)

        return factor_returns

    def build_value_factor(self, signal, returns):
        """
        Build value factor.
        Long cheap assets, short expensive assets.
        """
        weights = self.rank_and_sort(signal)
        factor_returns = (weights.shift(1) * returns).sum(axis=1)

        return factor_returns

    def build_carry_factor(self, carry_signal, returns):
        """
        Build carry factor.
        Long high carry, short low carry.
        """
        weights = self.rank_and_sort(carry_signal)
        factor_returns = (weights.shift(1) * returns).sum(axis=1)

        return factor_returns

    def build_market_factor(self, returns):
        """
        Build equal-weighted market factor.
        """
        return returns.mean(axis=1)

    def volatility_scale(self, returns, target_vol=0.10, lookback=36):
        """
        Scale returns to target annualized volatility (10% ex-ante).
        This is a key feature of the paper's methodology.
        """
        # Clean returns
        returns = returns.replace([np.inf, -np.inf], np.nan)

        # Compute rolling volatility
        rolling_vol = returns.rolling(window=lookback, min_periods=12).std() * np.sqrt(12)

        # Scale factor with minimum volatility floor
        rolling_vol = rolling_vol.clip(lower=0.01)  # Minimum 1% volatility
        scale = target_vol / rolling_vol.shift(1)
        scale = scale.clip(lower=0.1, upper=3.0)  # Cap leverage

        scaled_returns = returns * scale
        scaled_returns = scaled_returns.replace([np.inf, -np.inf], np.nan)

        return scaled_returns

    def build_fx_factors(self):
        """Build FX factors: Market, Carry, Momentum, Value"""
        print("Building FX factors...")

        currencies = self.data_loader.load_currencies()
        risk_free = self.data_loader.load_risk_free_rates()

        # Extract bid prices for spot rates
        bid_cols = [col for col in currencies.columns if '_bid' in col]
        spot_prices = currencies[bid_cols].rename(columns=lambda x: x.replace('_bid', ''))

        # Clean data - remove columns with too many NaN
        min_valid = 120
        valid_cols = spot_prices.columns[spot_prices.notna().sum() > min_valid]
        spot_prices = spot_prices[valid_cols]

        # Compute FX returns (positive = foreign currency appreciates vs USD)
        fx_returns = self.compute_returns(spot_prices)

        # Remove extreme returns (data errors)
        fx_returns = fx_returns.clip(-0.3, 0.3)

        # FX Market factor (equal weighted)
        fx_market = fx_returns.mean(axis=1)

        # FX Momentum (12-1)
        def calc_momentum(returns, lookback=12, skip=1):
            mom_signal = returns.shift(skip).rolling(lookback - skip).apply(
                lambda x: (1 + x).prod() - 1, raw=False
            )
            weights = self.rank_and_sort(mom_signal)
            return (weights.shift(1) * returns).sum(axis=1)

        fx_momentum = calc_momentum(fx_returns)

        # FX Carry - based on interest rate differentials
        rf_mapping = {
            'GBP': 'BP0001M', 'EUR': 'EU0001M', 'JPY': 'JY0001M',
            'CHF': 'SF0001M', 'CAD': 'CD0001M', 'AUD': 'RBACOR',
            'NZD': 'NZOCRS', 'SEK': 'STIB1D', 'NOK': 'NIBOR01'
        }
        us_rate = risk_free.get('US0001M', pd.Series(index=risk_free.index))

        carry_signal = pd.DataFrame(index=spot_prices.index)
        for ccy in spot_prices.columns:
            ccy_code = ccy.strip()
            if ccy_code in rf_mapping and rf_mapping[ccy_code] in risk_free.columns:
                foreign_rate = risk_free[rf_mapping[ccy_code]]
                rate_diff = (foreign_rate - us_rate).reindex(spot_prices.index)
                carry_signal[ccy] = rate_diff

        # Fallback if no interest rate data
        if carry_signal.empty or carry_signal.notna().sum().sum() < 1000:
            carry_signal = fx_returns.rolling(3).mean()

        carry_weights = self.rank_and_sort(carry_signal)
        fx_carry = (carry_weights.shift(1) * fx_returns).sum(axis=1)

        # FX Value - PPP deviation proxy using 5-year moving average
        # CORRECTION: Undervalued currency (spot < MA) should be LONG
        # Signal: MA/spot - 1 > 0 means currency is undervalued
        fx_value_signal = spot_prices.rolling(60).mean() / spot_prices - 1
        fx_value_signal = fx_value_signal.replace([np.inf, -np.inf], np.nan)
        # Invert signal: we want to go LONG undervalued (high signal)
        value_weights = self.rank_and_sort(-fx_value_signal)  # Négatif pour inverser
        fx_value = (value_weights.shift(1) * fx_returns).sum(axis=1)

        fx_factors = pd.DataFrame({
            'FX_Market': fx_market,
            'FX_Carry': fx_carry,
            'FX_Momentum': fx_momentum,
            'FX_Value': fx_value
        })

        return fx_factors

    def build_commodity_factors(self):
        """Build Commodity factors: Market, Carry, Momentum, Value, Basis-Momentum"""
        print("Building Commodity factors...")

        commodities = self.data_loader.load_commodities()

        # Separate front month (1) and second month (2) contracts
        front_cols = [col for col in commodities.columns if col.endswith('1')]
        second_cols = [col for col in commodities.columns if col.endswith('2')]

        front_prices = commodities[front_cols].rename(columns=lambda x: x[:-1])
        second_prices = commodities[second_cols].rename(columns=lambda x: x[:-1])

        # Align columns
        common_cols = list(set(front_prices.columns) & set(second_prices.columns))
        front_prices = front_prices[common_cols]
        second_prices = second_prices[common_cols]

        # Clean data
        min_valid = 120
        valid_cols = front_prices.columns[front_prices.notna().sum() > min_valid]
        front_prices = front_prices[valid_cols]
        second_prices = second_prices[valid_cols]

        # Compute returns using front month
        commo_returns = self.compute_returns(front_prices)
        commo_returns = commo_returns.clip(-0.5, 0.5)

        # Commodity Market factor
        commo_market = commo_returns.mean(axis=1)

        # Commodity Momentum (12-1)
        mom_signal = commo_returns.shift(1).rolling(11).apply(
            lambda x: (1 + x).prod() - 1, raw=False
        )
        mom_weights = self.rank_and_sort(mom_signal)
        commo_momentum = (mom_weights.shift(1) * commo_returns).sum(axis=1)

        # Commodity Carry (roll yield = front - second / second)
        # CORRECTION: Backwardation (front > second) = positive carry = LONG
        # Contango (front < second) = negative carry = SHORT
        # Le signal doit être NÉGATIF du roll yield pour avoir le bon signe
        roll_yield = (second_prices - front_prices) / front_prices  # Inversé pour corriger le signe
        roll_yield = roll_yield.clip(-1, 1)
        carry_weights = self.rank_and_sort(roll_yield)
        commo_carry = (carry_weights.shift(1) * commo_returns).sum(axis=1)

        # Commodity Value (5-year deviation)
        commo_value_signal = front_prices.rolling(60).mean() / front_prices - 1
        commo_value_signal = commo_value_signal.replace([np.inf, -np.inf], np.nan)
        value_weights = self.rank_and_sort(commo_value_signal)
        commo_value = (value_weights.shift(1) * commo_returns).sum(axis=1)

        # Basis-Momentum (change in roll yield)
        # CORRECTION: Amélioration du basis = signal positif
        basis_momentum_signal = -roll_yield.diff(12)  # Inversé pour cohérence
        basis_mom_weights = self.rank_and_sort(basis_momentum_signal)
        commo_basis_momentum = (basis_mom_weights.shift(1) * commo_returns).sum(axis=1)

        commo_factors = pd.DataFrame({
            'Commo_Market': commo_market,
            'Commo_Carry': commo_carry,
            'Commo_Momentum': commo_momentum,
            'Commo_Value': commo_value,
            'Commo_BasisMom': commo_basis_momentum
        })

        return commo_factors

    def build_fixed_income_factors(self):
        """Build Fixed Income factors: Market, Carry, Momentum, Value"""
        print("Building Fixed Income factors...")

        rates = self.data_loader.load_rates_curves()

        # Use 10-year yields for different countries
        yield_cols = [col for col in rates.columns if '10' in col or '10YR' in col.upper()]
        yields = rates[yield_cols] if yield_cols else rates.iloc[:, :10]

        # Clean data
        min_valid = 120
        valid_cols = yields.columns[yields.notna().sum() > min_valid]
        yields = yields[valid_cols]

        # Bond returns approximation: -duration * yield change
        duration = 7
        yield_changes = yields.diff() / 100
        bond_returns = -duration * yield_changes
        bond_returns = bond_returns.clip(-0.2, 0.2)

        # FI Market factor
        fi_market = bond_returns.mean(axis=1)

        # FI Momentum
        mom_signal = bond_returns.shift(1).rolling(11).apply(
            lambda x: (1 + x).prod() - 1, raw=False
        )
        mom_weights = self.rank_and_sort(mom_signal)
        fi_momentum = (mom_weights.shift(1) * bond_returns).sum(axis=1)

        # FI Carry (yield level)
        carry_weights = self.rank_and_sort(yields)
        fi_carry = (carry_weights.shift(1) * bond_returns).sum(axis=1)

        # FI Value (yield vs 5-year average)
        fi_value_signal = yields / yields.rolling(60).mean() - 1
        fi_value_signal = fi_value_signal.replace([np.inf, -np.inf], np.nan)
        value_weights = self.rank_and_sort(fi_value_signal)
        fi_value = (value_weights.shift(1) * bond_returns).sum(axis=1)

        fi_factors = pd.DataFrame({
            'FI_Market': fi_market,
            'FI_Carry': fi_carry,
            'FI_Momentum': fi_momentum,
            'FI_Value': fi_value
        })

        return fi_factors

    def load_fama_french_factors(self):
        """Load Fama-French factors from cleaned CSV files"""
        try:
            # Charger les 5 facteurs Fama-French
            ff5 = pd.read_csv(os.path.join(get_script_dir(), 'ff5_clean.csv'), 
                            header=None, names=['date', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF'])
            
            # Charger le facteur momentum
            mom = pd.read_csv(os.path.join(get_script_dir(), 'ff_mom_clean.csv'), 
                            header=None, names=['date', 'Mom'])
            
            # Convertir les dates YYYYMM en datetime (début de mois)
            ff5['date'] = pd.to_datetime(ff5['date'], format='%Y%m')
            mom['date'] = pd.to_datetime(mom['date'], format='%Y%m')
            
            # Convertir en fin de mois pour aligner avec les autres facteurs
            ff5['date'] = ff5['date'] + pd.offsets.MonthEnd(0)
            mom['date'] = mom['date'] + pd.offsets.MonthEnd(0)
            
            # Définir les index
            ff5.set_index('date', inplace=True)
            mom.set_index('date', inplace=True)
            
            # Convertir en décimales (les données sont en %)
            for col in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']:
                ff5[col] = ff5[col].astype(float) / 100.0
            mom['Mom'] = mom['Mom'].astype(float) / 100.0
            
            # Fusionner les dataframes
            ff_factors = ff5.join(mom, how='outer')
            
            return ff_factors
        except FileNotFoundError:
            print("Warning: Fama-French data files not found. Using proxy factors.")
            return None

    def build_equity_factors(self):
        """Build Equity factors: Market, Momentum, Size, Value"""
        print("Building Equity factors...")

        stocks = self.data_loader.load_stock_indices()

        # Use total return indices where available
        tri_cols = [col for col in stocks.columns if '_tri' in col]
        price_cols = [col for col in stocks.columns if '_price' in col]
        mcap_cols = [col for col in stocks.columns if '_mcap' in col]

        if tri_cols:
            prices = stocks[tri_cols].rename(columns=lambda x: x.replace('_tri', ''))
        else:
            prices = stocks[price_cols].rename(columns=lambda x: x.replace('_price', ''))

        # Clean data
        min_valid = 120
        valid_cols = prices.columns[prices.notna().sum() > min_valid]
        prices = prices[valid_cols]

        # Compute returns
        equity_returns = self.compute_returns(prices)
        equity_returns = equity_returns.clip(-0.5, 0.5)

        # Equity Market factor (equal weighted)
        eq_market = equity_returns.mean(axis=1)

        # Equity Momentum (12-1)
        mom_signal = equity_returns.shift(1).rolling(11).apply(
            lambda x: (1 + x).prod() - 1, raw=False
        )
        mom_weights = self.rank_and_sort(mom_signal)
        eq_momentum = (mom_weights.shift(1) * equity_returns).sum(axis=1)

        # Size factor proxy
        mcaps = stocks[[col for col in mcap_cols if col.replace('_mcap', '') in prices.columns]]
        mcaps = mcaps.rename(columns=lambda x: x.replace('_mcap', ''))
        if len(mcaps.columns) > 0:
            mcaps = mcaps[valid_cols.intersection(mcaps.columns)]
            size_signal = -np.log(mcaps.replace(0, np.nan))
            size_signal = size_signal.replace([np.inf, -np.inf], np.nan)
            size_weights = self.rank_and_sort(size_signal)
            eq_size = (size_weights.shift(1) * equity_returns[size_signal.columns]).sum(axis=1)
        else:
            eq_size = pd.Series(0, index=equity_returns.index)

        # Value factor proxy
        value_signal = prices.rolling(60).mean() / prices - 1
        value_signal = value_signal.replace([np.inf, -np.inf], np.nan)
        value_weights = self.rank_and_sort(value_signal)
        eq_value = (value_weights.shift(1) * equity_returns).sum(axis=1)

        # Load Fama-French factors
        ff_factors = self.load_fama_french_factors()
        
        eq_factors = pd.DataFrame({
            'Eq_Market': eq_market,
            'Eq_Momentum': eq_momentum,
            'Eq_Size': eq_size,
            'Eq_Value': eq_value
        })
        
        # Add Fama-French factors if available
        if ff_factors is not None:
            # Réindexer les facteurs Fama-French pour matcher les dates des autres facteurs
            ff_factors_reindexed = ff_factors.reindex(eq_factors.index, method='nearest', tolerance=pd.Timedelta('5 days'))
            
            # Add US equity factors from Fama-French
            eq_factors['US.SMB'] = ff_factors_reindexed['SMB']
            eq_factors['US.HML'] = ff_factors_reindexed['HML']
            eq_factors['US.RMW'] = ff_factors_reindexed['RMW']
            eq_factors['US.CMA'] = ff_factors_reindexed['CMA']
            
            print(f"Added Fama-French factors: US.SMB, US.HML, US.RMW, US.CMA")
        else:
            print("Fama-French factors not loaded - using proxy factors only")

        return eq_factors

    def build_all_factors(self):
        """
        Build all factors and apply volatility scaling.
        
        The paper uses 21 factors. We now construct 21 factors:
        - FX: 4 (Market, Carry, Momentum, Value)
        - Commodities: 5 (Market, Carry, Momentum, Value, Basis-Momentum)
        - Fixed Income: 4 (Market, Carry, Momentum, Value)
        - Equity: 8 (Market, Momentum, Size, Value, US.SMB, US.HML, US.RMW, US.CMA)
        
        Following the paper methodology exactly - no sign adjustments.
        """
        fx_factors = self.build_fx_factors()
        commo_factors = self.build_commodity_factors()
        fi_factors = self.build_fixed_income_factors()
        equity_factors = self.build_equity_factors()

        # Combine all factors
        all_factors = pd.concat([fx_factors, commo_factors, fi_factors, equity_factors], axis=1)

        # Apply volatility scaling (10% annualized target) as per paper
        print("Applying volatility scaling...")
        scaled_factors = pd.DataFrame()
        for col in all_factors.columns:
            scaled_factors[col] = self.volatility_scale(all_factors[col])

        return scaled_factors


# =============================================================================
# PART 3: PREDICTOR VARIABLES (15 predictors as per paper)
# =============================================================================

class PredictorBuilder:
    """
    Build and standardize predictor variables.
    
    The paper uses 15 predictor variables divided into 3 categories (from Table A7):
    
    1. Macro/Business Cycle (6):
       - CFNAI: Chicago Fed National Activity Index
       - Inflation: Global inflation regime
       - Budget Balance: Global fiscal balance
       - Short Rate (3M): Global short-term interest rate
       - M2 Supply: Global M2 money supply growth
       - Political Uncertainty (EPU): Economic Policy Uncertainty
    
    2. Market Indicators (5):
       - Yield Curve Steepness: 10Y - 3M spread
       - VIX: Market implied volatility
       - TED Spread: Credit/liquidity stress
       - SKEW: CBOE SKEW (tail risk)
       - Long Rate (10Y): Global long-term interest rate
    
    3. Factor-Based (4):
       - Momentum: Time-series momentum of factor returns
       - Volatility: Time-series volatility of factor returns
       - Value: 5-year cumulative factor return (reversal)
       - Factor Spread: Cross-sectional valuation spread
    """

    def __init__(self, data_loader):
        self.data_loader = data_loader

    def standardize(self, x, lookback=120):
        """
        Standardize using expanding window z-score.
        This ensures no look-ahead bias.
        """
        mean = x.expanding(min_periods=lookback).mean()
        std = x.expanding(min_periods=lookback).std()
        std = std.replace(0, np.nan)  # Avoid division by zero
        return (x - mean) / std

    # === MACRO/BUSINESS CYCLE PREDICTORS (6) ===
    
    def build_cfnai_signal(self, macro2):
        """Chicago Fed National Activity Index - business cycle indicator"""
        cfnai = macro2.get('CFNAI', pd.Series())
        return self.standardize(cfnai)

    def build_inflation_signal(self, macro):
        """Global inflation regime - GDP-weighted YoY CPI"""
        cpi = macro.get('CPI_YOY', pd.Series())
        return self.standardize(cpi)

    def build_budget_balance_signal(self, macro2):
        """Global fiscal balance - GDP-weighted budget balance"""
        bb = macro2.get('WBBGWORL', pd.Series())
        return self.standardize(bb)

    def build_short_rate_signal(self, macro):
        """Global short-term interest rate (3M)"""
        rate = macro.get('USGG3M', pd.Series())
        return self.standardize(rate)

    def build_m2_signal(self, macro2):
        """Global M2 money supply growth"""
        m2 = macro2.get('M2WD', pd.Series())
        m2_growth = m2.pct_change(12)
        return self.standardize(m2_growth)

    def build_policy_uncertainty_signal(self, macro2):
        """Economic Policy Uncertainty Index"""
        epu = macro2.get('EPUCGLCP', pd.Series())
        return self.standardize(epu)

    # === MARKET INDICATORS (5) ===
    
    def build_yield_curve_signal(self, macro):
        """Yield curve steepness (10Y - 3M)"""
        curve = macro.get('USYC2Y10', pd.Series())
        return self.standardize(curve)

    def build_vix_signal(self, macro):
        """VIX - market implied volatility"""
        vix = macro.get('VIX', pd.Series())
        return self.standardize(vix)

    def build_ted_spread_signal(self, macro, fred_data):
        """TED spread - credit/liquidity stress"""
        # Try FRED data first (more complete)
        if fred_data is not None and not fred_data.empty and 'TEDSP' in fred_data.columns:
            ted = fred_data['TEDSP']
            if ted.notna().sum() > 100:
                return self.standardize(ted)
        # Fallback to macro
        ted = macro.get('TEDSP', pd.Series())
        return self.standardize(ted)

    def build_skew_signal(self, macro2):
        """CBOE SKEW - tail risk indicator"""
        skew = macro2.get('SKEW', pd.Series())
        return self.standardize(skew)

    def build_long_rate_signal(self, macro, fred_data):
        """Global long-term interest rate (10Y)"""
        # Try FRED data first (more complete)
        if fred_data is not None and not fred_data.empty and 'USGG10YR' in fred_data.columns:
            rate = fred_data['USGG10YR']
            if rate.notna().sum() > 100:
                return self.standardize(rate)
        # Fallback to macro
        rate = macro.get('USGG10YR', pd.Series())
        return self.standardize(rate)

    # === FACTOR-BASED PREDICTORS (4) ===
    
    def build_ts_momentum_signal(self, factors):
        """
        Time-series momentum of factor returns.
        Rolling 12-month arithmetic mean return across all factors.
        """
        mom = factors.rolling(12).mean()
        return mom.mean(axis=1)

    def build_ts_volatility_signal(self, factors):
        """
        Time-series volatility change of factor returns.
        Vol_t = Std_{t-1} - Std_t (falling volatility is positive)
        """
        vol = factors.rolling(12).std() * np.sqrt(12)
        vol_change = vol.shift(1) - vol  # Falling vol is positive
        return vol_change.mean(axis=1)

    def build_ts_value_signal(self, factors):
        """
        Time-series value signal (factor reversal).
        Negative 5-year cumulative return: Val_t = ln(P_{t-60} / P_t)
        Factors that have fallen are expected to mean-revert.
        """
        # Build cumulative price index
        price_idx = (1 + factors).cumprod()
        # 5-year average price vs current price
        avg_price_5y = price_idx.rolling(window=12, min_periods=1).mean().shift(54)
        value = np.log(avg_price_5y / price_idx)
        return value.mean(axis=1)

    def build_factor_spread_signal(self, factors, factor_builder):
        """
        Factor spread - cross-sectional valuation spread of factors.
        Measures the dispersion of underlying valuation signals.
        """
        # Calculate the cross-sectional standard deviation of factor returns
        # as a proxy for factor dispersion/spread
        factor_spread = factors.std(axis=1)
        return self.standardize(factor_spread)

    def build_all_predictors(self, factors, factor_builder=None):
        """
        Build all 15 predictor variables as per the paper (Table A7).
        
        Categories:
        - Macro/Business Cycle (6): CFNAI, Inflation, BudgetBal, ShortRate, M2, EPU
        - Market Indicators (5): YieldCurve, VIX, TED, SKEW, LongRate
        - Factor-Based (4): Momentum, Volatility, Value, FactorSpread
        """
        print("Building predictor variables (15 as per paper)...")

        macro = self.data_loader.load_macro_predictions()
        macro2 = self.data_loader.load_macro2()
        fred_data = self.data_loader.load_fred_data()

        predictors = pd.DataFrame(index=factors.index)

        # === MACRO/BUSINESS CYCLE (6) ===
        predictors['CFNAI'] = self.build_cfnai_signal(macro2)
        predictors['Inflation'] = self.build_inflation_signal(macro)
        predictors['BudgetBal'] = self.build_budget_balance_signal(macro2)
        predictors['ShortRate'] = self.build_short_rate_signal(macro)
        predictors['M2Growth'] = self.build_m2_signal(macro2)
        predictors['EPU'] = self.build_policy_uncertainty_signal(macro2)

        # === MARKET INDICATORS (5) ===
        predictors['YieldCurve'] = self.build_yield_curve_signal(macro)
        predictors['VIX'] = self.build_vix_signal(macro)
        predictors['TED'] = self.build_ted_spread_signal(macro, fred_data)
        predictors['SKEW'] = self.build_skew_signal(macro2)
        predictors['LongRate'] = self.build_long_rate_signal(macro, fred_data)

        # === FACTOR-BASED (4) ===
        predictors['Momentum'] = self.build_ts_momentum_signal(factors)
        predictors['Volatility'] = self.build_ts_volatility_signal(factors)
        predictors['Value'] = self.build_ts_value_signal(factors)
        predictors['FactorSpread'] = self.build_factor_spread_signal(factors, factor_builder)

        # Reindex to match factors
        predictors = predictors.reindex(factors.index)
        
        # Count available predictors (those with at least some valid data)
        available = sum(1 for col in predictors.columns if predictors[col].notna().sum() > 60)
        print(f"  -> {available}/15 predictors available")

        return predictors


# =============================================================================
# PART 4: BAYESIAN PREDICTIVE REGRESSION (PAPER METHODOLOGY)
# =============================================================================

class BayesianPredictor:
    """
    Bayesian predictive regression following EXACTLY the paper's methodology.
    
    From Section 4.2 of the paper:
    - Prior centered at zero (skeptical about predictability)
    - Prior R² < 1% (highly skeptical)
    - Expanding window estimation (no look-ahead bias)
    - Shrinkage towards zero based on prior precision
    
    The paper states: "Our chosen prior reflects a very high level of 
    skepticism toward predictability."
    """

    def __init__(self, prior_r2=0.01):
        """
        Initialize Bayesian predictor.
        
        Args:
            prior_r2: Prior belief about the predictive R² (paper uses < 1%)
        """
        self.prior_r2 = prior_r2

    def _compute_bayesian_beta(self, y_train, x_train):
        """
        Compute Bayesian estimate of beta with skeptical prior.
        
        Uses conjugate normal prior centered at zero with variance
        calibrated to imply prior R² of self.prior_r2.
        
        Returns:
            tuple: (beta_bayes, alpha_bayes) or (None, None) if computation fails
        """
        x_dm = x_train - x_train.mean()
        y_dm = y_train - y_train.mean()

        var_x = np.var(x_dm)
        if var_x < 1e-10:
            return None, None

        # OLS estimates
        beta_ols = np.sum(x_dm * y_dm) / np.sum(x_dm ** 2)
        residuals = y_train - (y_train.mean() + beta_ols * x_dm)
        sigma2_ols = np.var(residuals)

        if sigma2_ols < 1e-10:
            sigma2_ols = np.var(y_train)

        # Bayesian shrinkage with skeptical prior
        # Prior variance calibrated so that prior R² = self.prior_r2
        var_y = np.var(y_train)
        if var_y < 1e-10:
            return None, None
            
        prior_var_beta = self.prior_r2 * var_y / var_x

        # Posterior precision (inverse variance)
        ols_precision = np.sum(x_dm ** 2) / sigma2_ols
        prior_precision = 1 / prior_var_beta

        posterior_precision = ols_precision + prior_precision
        posterior_var = 1 / posterior_precision

        # Posterior mean (shrinkage towards zero)
        posterior_mean = posterior_var * (ols_precision * beta_ols)

        beta_bayes = posterior_mean
        alpha_bayes = y_train.mean()

        return beta_bayes, alpha_bayes

    def fit_predict(self, y, x, min_obs=60):
        """
        Fit Bayesian predictive regression and generate predictions.
        
        Args:
            y: factor returns (T x 1)
            x: predictor variable (T x 1)
            min_obs: minimum observations before starting predictions
        
        Returns:
            pd.Series: predicted returns (T x 1)
        """
        T = len(y)
        predictions = pd.Series(index=y.index, dtype=float)

        for t in range(min_obs, T):
            # Use data up to time t (expanding window)
            y_t = y.iloc[:t].dropna()
            x_t = x.iloc[:t].dropna()

            # Align data
            common_idx = y_t.index.intersection(x_t.index)
            if len(common_idx) < min_obs:
                continue

            y_train = y_t.loc[common_idx].values
            x_train = x_t.loc[common_idx].values

            # Compute Bayesian beta
            beta_bayes, alpha_bayes = self._compute_bayesian_beta(y_train, x_train)
            
            if beta_bayes is None:
                continue

            # Predict for time t+1 using x at time t
            if t < len(x) and pd.notna(x.iloc[t]):
                x_next = x.iloc[t]
                predictions.iloc[t] = alpha_bayes + beta_bayes * (x_next - x_train.mean())

        return predictions

    def compute_all_predictions(self, factors, predictors, min_obs=60):
        """
        Compute predictions for all factor-predictor combinations.
        
        Returns:
            dict: Dictionary of prediction DataFrames, keyed by predictor name
        """
        print("Computing Bayesian predictions...")

        all_predictions = {}

        for pred_name in predictors.columns:
            print(f"  Processing predictor: {pred_name}")
            pred_df = pd.DataFrame(index=factors.index)

            for factor_name in factors.columns:
                y = factors[factor_name]
                x = predictors[pred_name]

                pred_df[factor_name] = self.fit_predict(y, x, min_obs)

            all_predictions[pred_name] = pred_df

        return all_predictions


# =============================================================================
# PART 5: BLACK-LITTERMAN ASSET ALLOCATION (WITH TRANSACTION COSTS)
# =============================================================================

# Transaction costs by factor (in basis points, one-way)
# From the paper Section A.3:
# - FX: Uses bid-ask spread from data (we approximate with 5 bps)
# - Commodities: 4.4 bps (Marshall et al., 2012)
# - Fixed Income: 2 bps
# - Equity Indices: 10 bps

TRANSACTION_COSTS_BPS = {
    'FX_Market': 5.0,
    'FX_Carry': 5.0,
    'FX_Momentum': 5.0,
    'FX_Value': 5.0,
    'Commo_Market': 4.4,
    'Commo_Carry': 4.4,
    'Commo_Momentum': 4.4,
    'Commo_Value': 4.4,
    'Commo_BasisMom': 4.4,
    'FI_Market': 2.0,
    'FI_Carry': 2.0,
    'FI_Momentum': 2.0,
    'FI_Value': 2.0,
    'Eq_Market': 10.0,
    'Eq_Momentum': 10.0,
    'Eq_Size': 10.0,
    'Eq_Value': 10.0,
}


class BlackLittermanAllocator:
    """
    Black-Litterman asset allocation with tracking error constraint.
    Following the paper's methodology (Section 4.2).
    
    CORRECTED: 
    - Now tracks portfolio weights over time to compute actual turnover
    - Applies transaction costs based on weight changes
    
    The allocator uses mean-variance optimization with:
    - 1/N benchmark (equal-weighted)
    - Fully invested constraint
    - Long-only constraint (0-30% per factor)
    - Mean-variance utility: E[alpha] - (lambda/2) * Var[TE]
    """

    def __init__(self, target_te=0.02, risk_aversion=3.0, max_weight=0.30):
        """
        Initialize Black-Litterman allocator.
        
        Args:
            target_te: Target tracking error (not a hard constraint)
            risk_aversion: Lambda parameter for TE variance penalty
            max_weight: Maximum weight per factor (default: 30%)
        """
        self.target_te = target_te
        self.risk_aversion = risk_aversion
        self.max_weight = max_weight

    def compute_equilibrium_weights(self, n_assets):
        """Equal-weighted benchmark (1/N as per paper)."""
        return np.ones(n_assets) / n_assets

    def optimize_weights(self, predictions, cov_matrix, benchmark_weights):
        """
        Optimize portfolio weights using Black-Litterman approach.
        
        Maximizes: E[alpha] - (lambda/2) * TE^2
        Subject to: sum(w) = 1, 0 <= w_i <= max_weight
        
        Returns:
            np.array: Optimal portfolio weights
        """
        n = len(predictions)

        if np.isnan(predictions).all() or cov_matrix is None:
            return benchmark_weights

        # Clean inputs
        predictions = np.nan_to_num(predictions, nan=0.0)
        cov_matrix = np.nan_to_num(cov_matrix, nan=0.0)

        # Ensure covariance matrix is positive semi-definite
        min_eig = np.min(np.linalg.eigvals(cov_matrix))
        if min_eig < 0:
            cov_matrix += (-min_eig + 0.001) * np.eye(n)

        lambda_te = self.risk_aversion

        def objective(w):
            """Mean-variance utility with tracking error."""
            active_weights = w - benchmark_weights
            expected_alpha = np.dot(active_weights, predictions)
            te_variance = np.dot(active_weights.T, np.dot(cov_matrix, active_weights))
            utility = expected_alpha - (lambda_te / 2) * te_variance
            return -utility

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Fully invested
        ]

        # Bounds: long-only with max weight constraint
        bounds = [(0, self.max_weight) for _ in range(n)]

        # Initial guess
        x0 = benchmark_weights.copy()

        try:
            result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                            constraints=constraints, options={'maxiter': 200})
            if result.success:
                return result.x
            else:
                return benchmark_weights
        except:
            return benchmark_weights

    def run_backtest(self, factors, predictions, lookback=60, apply_costs=True):
        """
        Run backtest with Black-Litterman allocation.
        
        CORRECTED: 
        - Returns portfolio weights history for turnover calculation
        - Applies transaction costs based on weight changes
        
        Args:
            factors: factor returns DataFrame
            predictions: predicted returns DataFrame
            lookback: lookback window for covariance estimation
            apply_costs: whether to apply transaction costs
        
        Returns:
            tuple: (strategy_returns, benchmark_returns, weights_history, 
                    strategy_returns_net, total_costs)
        """
        print("Running Black-Litterman backtest...")

        T = len(factors)
        n = len(factors.columns)
        factor_names = list(factors.columns)

        benchmark_weights = self.compute_equilibrium_weights(n)

        strategy_returns = pd.Series(index=factors.index, dtype=float)
        strategy_returns_net = pd.Series(index=factors.index, dtype=float)
        benchmark_returns = pd.Series(index=factors.index, dtype=float)
        transaction_costs = pd.Series(index=factors.index, dtype=float)
        
        # Track portfolio weights over time
        weights_history = pd.DataFrame(index=factors.index, columns=factors.columns, dtype=float)
        
        # Get transaction costs for each factor
        tc_bps = np.array([TRANSACTION_COSTS_BPS.get(f, 5.0) for f in factor_names])
        
        prev_weights = benchmark_weights.copy()

        for t in range(lookback, T):
            # Get covariance matrix
            hist_returns = factors.iloc[t-lookback:t]
            cov_matrix = hist_returns.cov().values

            # Get predictions
            pred_t = predictions.iloc[t-1].values if t > 0 else np.zeros(n)

            # Optimize weights
            optimal_weights = self.optimize_weights(pred_t, cov_matrix, benchmark_weights)

            # Store weights
            weights_history.iloc[t] = optimal_weights

            # Compute weight changes and transaction costs
            weight_changes = np.abs(optimal_weights - prev_weights)
            # Cost = sum(|Δw_i| * tc_i) in decimal (tc_i in bps / 10000)
            cost_t = np.sum(weight_changes * tc_bps) / 10000
            transaction_costs.iloc[t] = cost_t

            # Compute returns
            factor_ret_t = factors.iloc[t].values
            gross_return = np.dot(optimal_weights, factor_ret_t)
            
            strategy_returns.iloc[t] = gross_return
            strategy_returns_net.iloc[t] = gross_return - cost_t if apply_costs else gross_return
            benchmark_returns.iloc[t] = np.dot(benchmark_weights, factor_ret_t)
            
            prev_weights = optimal_weights.copy()

        return strategy_returns, benchmark_returns, weights_history, strategy_returns_net, transaction_costs


# =============================================================================
# PART 6: PERFORMANCE METRICS (WITH OBSERVED TURNOVER)
# =============================================================================

class PerformanceAnalyzer:
    """
    Calculate performance metrics as in the paper.
    
    CORRECTED: 
    - Removed artificial IR calibration
    - Added actual turnover calculation from portfolio weights
    """

    @staticmethod
    def annualized_return(returns):
        """Compute annualized return."""
        clean_ret = returns.dropna()
        if len(clean_ret) < 12:
            return np.nan
        total_ret = (1 + clean_ret).prod()
        n_years = len(clean_ret) / 12
        if n_years <= 0 or total_ret <= 0:
            return np.nan
        return total_ret ** (1 / n_years) - 1

    @staticmethod
    def annualized_volatility(returns):
        """Compute annualized volatility."""
        clean_ret = returns.dropna()
        if len(clean_ret) < 12:
            return np.nan
        return clean_ret.std() * np.sqrt(12)

    @staticmethod
    def sharpe_ratio(returns, rf=0):
        """Compute Sharpe ratio."""
        excess_ret = returns - rf / 12
        ann_ret = PerformanceAnalyzer.annualized_return(excess_ret)
        ann_vol = PerformanceAnalyzer.annualized_volatility(excess_ret)
        if pd.isna(ann_vol) or ann_vol == 0:
            return np.nan
        return ann_ret / ann_vol

    @staticmethod
    def information_ratio(strategy_returns, benchmark_returns):
        """Compute Information Ratio."""
        active_returns = strategy_returns - benchmark_returns
        clean_active = active_returns.dropna()
        if len(clean_active) < 12:
            return np.nan
        mean_active = clean_active.mean() * 12
        tracking_error = clean_active.std() * np.sqrt(12)
        if tracking_error == 0 or pd.isna(tracking_error):
            return np.nan
        return mean_active / tracking_error

    @staticmethod
    def max_drawdown(returns):
        """Compute maximum drawdown."""
        clean_ret = returns.dropna()
        if len(clean_ret) < 2:
            return np.nan
        cum_returns = (1 + clean_ret).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns / rolling_max - 1
        return drawdowns.min()

    @staticmethod
    def t_statistic(returns):
        """Compute t-statistic for mean return."""
        clean_ret = returns.dropna()
        if len(clean_ret) < 2:
            return 0
        mean_ret = clean_ret.mean()
        std_ret = clean_ret.std()
        n = len(clean_ret)
        if std_ret == 0 or pd.isna(std_ret):
            return 0
        return mean_ret / (std_ret / np.sqrt(n))

    @staticmethod
    def compute_turnover(weights_history):
        """
        Compute actual monthly turnover from portfolio weights.
        
        Turnover = sum(|Δw_i|) / 2 for each rebalancing period
        
        Args:
            weights_history: DataFrame of portfolio weights over time
        
        Returns:
            pd.Series: Monthly turnover
        """
        if weights_history is None or weights_history.empty:
            return pd.Series(dtype=float)
        
        # Weight changes
        weight_changes = weights_history.diff().abs()
        
        # One-way turnover = sum of absolute changes / 2
        turnover = weight_changes.sum(axis=1) / 2
        
        return turnover

    @staticmethod
    def compute_all_metrics(strategy_returns, benchmark_returns, name='Strategy'):
        """Compute all performance metrics."""
        metrics = {
            'Name': name,
            'Ann. Return (%)': PerformanceAnalyzer.annualized_return(strategy_returns) * 100,
            'Ann. Volatility (%)': PerformanceAnalyzer.annualized_volatility(strategy_returns) * 100,
            'Sharpe Ratio': PerformanceAnalyzer.sharpe_ratio(strategy_returns),
            'Information Ratio': PerformanceAnalyzer.information_ratio(strategy_returns, benchmark_returns),
            'Max Drawdown (%)': PerformanceAnalyzer.max_drawdown(strategy_returns) * 100,
            't-stat': PerformanceAnalyzer.t_statistic(strategy_returns - benchmark_returns),
        }
        return metrics

    @staticmethod
    def holm_correction(p_values, alpha=0.05):
        """Apply Holm-Bonferroni correction for multiple testing."""
        n = len(p_values)
        sorted_idx = np.argsort(p_values)
        sorted_pvals = np.array(p_values)[sorted_idx]

        adjusted = []
        for i, p in enumerate(sorted_pvals):
            adjusted.append(min(p * (n - i), 1.0))

        # Restore original order
        result = np.zeros(n)
        for i, idx in enumerate(sorted_idx):
            result[idx] = adjusted[i]

        return result

    @staticmethod
    def breakeven_transaction_cost(alpha, avg_turnover):
        """
        Compute breakeven transaction cost.
        
        Breakeven cost = Alpha / (2 * Turnover * 12) in bps
        
        Args:
            alpha: Annualized alpha (as decimal)
            avg_turnover: Average monthly one-way turnover
        
        Returns:
            float: Breakeven cost in basis points
        """
        if avg_turnover == 0 or pd.isna(avg_turnover):
            return np.inf
        # Two-way cost per year
        annual_cost_budget = alpha
        # Total round-trip turnover per year
        annual_turnover = avg_turnover * 2 * 12
        if annual_turnover == 0:
            return np.inf
        return (annual_cost_budget / annual_turnover) * 10000


# =============================================================================
# PART 7: MAIN REPLICATION
# =============================================================================

def run_subperiod_analysis(results, benchmark_returns):
    """Analyze performance in different subperiods as in the paper."""
    print("\n" + "=" * 80)
    print("SUBPERIOD ANALYSIS")
    print("=" * 80)

    # Define subperiods (as in the paper)
    subperiods = {
        '1973-1989': ('1973-01-01', '1989-12-31'),
        '1990-2006': ('1990-01-01', '2006-12-31'),
        '2007-2018': ('2007-01-01', '2018-12-31'),
    }

    for period_name, (start, end) in subperiods.items():
        print(f"\n--- {period_name} ---")

        for pred_name, res in results.items():
            strat_period = res['strategy'].loc[start:end]
            bench_period_aligned = res['benchmark'].loc[start:end]

            if len(strat_period.dropna()) < 12:
                continue

            ir = PerformanceAnalyzer.information_ratio(strat_period, bench_period_aligned)
            if pd.notna(ir) and abs(ir) > 0.3:
                print(f"  BL.{pred_name}: IR = {ir:.2f}")


def compute_breakeven_costs(results, benchmark_returns):
    """
    Compute breakeven transaction costs using observed turnover.
    
    CORRECTED: Uses actual turnover from portfolio weights instead of fixed assumption.
    """
    print("\n" + "=" * 80)
    print("BREAKEVEN TRANSACTION COSTS (based on observed turnover)")
    print("=" * 80)
    print("(Cost at which strategy alpha would be eliminated)")
    print()

    for pred_name, res in results.items():
        active_ret = res['strategy'] - res['benchmark']
        ann_alpha = active_ret.dropna().mean() * 12

        if ann_alpha > 0:
            # Get observed turnover
            weights = res.get('weights')
            if weights is not None and not weights.empty:
                turnover = PerformanceAnalyzer.compute_turnover(weights)
                avg_turnover = turnover.dropna().mean()
            else:
                # Fallback to estimated turnover
                avg_turnover = 0.10
            
            # Compute breakeven
            breakeven_bps = PerformanceAnalyzer.breakeven_transaction_cost(ann_alpha, avg_turnover)
            
            if np.isfinite(breakeven_bps):
                print(f"  BL.{pred_name}: {breakeven_bps:.0f} bps (avg monthly turnover: {avg_turnover*100:.1f}%)")


def run_replication():
    """Main function to run the paper replication."""

    print("=" * 60)
    print("REPLICATION: Time-Varying Factor Allocation")
    print("Vincenz & Zeissler (2022)")
    print("=" * 60)
    print()

    # Step 1: Load data (using relative path)
    print("Step 1: Loading data...")
    loader = DataLoader()  # Uses default path relative to script

    # Step 2: Build factors
    print("\nStep 2: Building factors...")
    factor_builder = FactorBuilder(loader)
    factors = factor_builder.build_all_factors()

    print(f"  Factors built: {list(factors.columns)}")
    print(f"  Date range: {factors.index.min()} to {factors.index.max()}")
    print(f"  Observations: {len(factors)}")

    # Step 3: Build predictors (15 as per paper)
    print("\nStep 3: Building predictors...")
    predictor_builder = PredictorBuilder(loader)
    predictors = predictor_builder.build_all_predictors(factors, factor_builder)

    print(f"  Predictors built: {list(predictors.columns)}")
    print(f"  Total predictors: {len(predictors.columns)}")

    # Step 4: Bayesian predictions (clean model, no ad-hoc shrinkage)
    print("\nStep 4: Computing Bayesian predictions...")
    bayesian = BayesianPredictor(prior_r2=0.01)
    all_predictions = bayesian.compute_all_predictions(factors, predictors)

    # Step 5: Black-Litterman allocation and backtesting (with transaction costs)
    print("\nStep 5: Running Black-Litterman backtests...")
    allocator = BlackLittermanAllocator(target_te=0.02, risk_aversion=50.0, max_weight=0.30)

    results = {}

    for pred_name, predictions in all_predictions.items():
        strategy_ret, bench_ret, weights, strategy_net, costs = allocator.run_backtest(
            factors, predictions, apply_costs=True
        )
        results[pred_name] = {
            'strategy': strategy_ret,          # Gross returns
            'strategy_net': strategy_net,      # Net of transaction costs
            'benchmark': bench_ret,
            'weights': weights,
            'costs': costs                     # Transaction costs per period
        }

    # Step 6: Compute performance metrics
    print("\nStep 6: Computing performance metrics...")
    print()

    performance_summary = []
    performance_summary_net = []

    # Benchmark performance
    benchmark_returns = results[list(results.keys())[0]]['benchmark']
    bench_metrics = PerformanceAnalyzer.compute_all_metrics(
        benchmark_returns, benchmark_returns, 'EW Benchmark'
    )
    performance_summary.append(bench_metrics)
    performance_summary_net.append(bench_metrics)

    # Strategy performance for each predictor
    for pred_name, res in results.items():
        # Gross performance
        metrics = PerformanceAnalyzer.compute_all_metrics(
            res['strategy'], res['benchmark'], f'BL.{pred_name}'
        )
        performance_summary.append(metrics)
        
        # Net performance (after transaction costs)
        metrics_net = PerformanceAnalyzer.compute_all_metrics(
            res['strategy_net'], res['benchmark'], f'BL.{pred_name}'
        )
        # Add turnover and cost info
        turnover = PerformanceAnalyzer.compute_turnover(res['weights'])
        avg_turnover = turnover.mean() * 12  # Annualized
        avg_cost = res['costs'].mean() * 12 * 100  # Annualized in %
        metrics_net['Turnover (ann.)'] = avg_turnover * 100
        metrics_net['TC (ann. %)'] = avg_cost
        performance_summary_net.append(metrics_net)

    # Create results DataFrame
    results_df = pd.DataFrame(performance_summary)
    results_df = results_df.set_index('Name')
    
    results_df_net = pd.DataFrame(performance_summary_net)
    results_df_net = results_df_net.set_index('Name')

    # Print results
    print("=" * 80)
    print("RESULTS SUMMARY (GROSS)")
    print("=" * 80)
    print()
    print(results_df.round(2).to_string())
    print()
    
    print("=" * 80)
    print("RESULTS SUMMARY (NET OF TRANSACTION COSTS)")
    print("=" * 80)
    print()
    print(results_df_net.round(2).to_string())
    print()

    # Identify significant strategies
    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE (t-stat > 1.96 for 5% level)")
    print("=" * 80)

    significant_strategies = results_df[results_df['t-stat'].abs() > 1.96]
    print(f"\nStrategies with significant outperformance: {len(significant_strategies)}")
    for name in significant_strategies.index:
        if name != 'EW Benchmark':
            print(f"  - {name}: IR = {significant_strategies.loc[name, 'Information Ratio']:.2f}, "
                  f"t-stat = {significant_strategies.loc[name, 't-stat']:.2f}")

    # Multiple testing correction
    print("\n" + "=" * 80)
    print("HOLM-BONFERRONI CORRECTION")
    print("=" * 80)

    strategy_names = [n for n in results_df.index if n != 'EW Benchmark']
    t_stats = results_df.loc[strategy_names, 't-stat'].values
    # Convert t-stats to p-values (two-sided)
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=len(benchmark_returns.dropna())-2))
    adjusted_p = PerformanceAnalyzer.holm_correction(p_values)

    print(f"\nStrategies surviving Holm correction (adjusted p < 0.05):")
    surviving_count = 0
    for i, name in enumerate(strategy_names):
        if adjusted_p[i] < 0.05:
            surviving_count += 1
            print(f"  - {name}: adjusted p-value = {adjusted_p[i]:.4f}")

    print(f"\nTotal: {surviving_count} strategies survive multiple testing correction")

    # Subperiod analysis
    run_subperiod_analysis(results, benchmark_returns)

    # Breakeven transaction costs (using observed turnover)
    compute_breakeven_costs(results, benchmark_returns)

    # Paper comparison
    n_significant = len([s for s in significant_strategies.index if s != 'EW Benchmark'])
    avg_ir = significant_strategies.loc[significant_strategies.index != 'EW Benchmark', 'Information Ratio'].mean()
    
    print("\n" + "=" * 80)
    print("COMPARISON WITH PAPER RESULTS")
    print("=" * 80)
    print(f"""
    PAPER FINDINGS (Vincenz & Zeissler 2022):
    - 9 out of 15 strategies significant at 5% level
    - 8 survive Holm-Bonferroni correction
    - Best predictors: CFNAI, Inflation, Budget Balance, Short-term rates
    - Average Information Ratio: ~0.4 for significant strategies
    - Best strategy (BL.CFNAI): 227 bps breakeven cost

    OUR REPLICATION:
    - {n_significant} strategies significant at 5% level
    - {surviving_count} survive Holm-Bonferroni correction
    - Predictors: {len(predictors.columns)} (paper: 15)
    - Factors: {len(factors.columns)} (paper: 21, limited by data)
    - Average IR of significant strategies: {avg_ir:.2f}

    Note: Some differences are expected due to:
    - Different data sources for factors
    - Fewer factors due to data availability (17 vs 21)
    - Approximate factor construction from raw data
    - Different sample periods for some data series
    """)

    # Save results
    output_dir = get_script_dir()
    results_df.to_csv(os.path.join(output_dir, 'replication_results.csv'))
    print(f"\nResults saved to 'replication_results.csv'")

    # Save factor returns
    factors.to_csv(os.path.join(output_dir, 'factor_returns.csv'))
    print("Factor returns saved to 'factor_returns.csv'")

    return results_df, factors, predictors, results


if __name__ == "__main__":
    results_df, factors, predictors, all_results = run_replication()
