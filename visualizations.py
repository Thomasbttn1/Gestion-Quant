"""
Visualization script for paper replication results
Creates figures similar to those in Vincenz & Zeissler (2022)
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from paper_replication import (
    DataLoader, FactorBuilder, PredictorBuilder,
    BayesianPredictor, BlackLittermanAllocator, PerformanceAnalyzer
)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def load_results():
    """Load or compute results"""
    print("Computing results...")

    loader = DataLoader('DataGestionQuant.xlsx')
    factor_builder = FactorBuilder(loader)
    factors = factor_builder.build_all_factors()

    predictor_builder = PredictorBuilder(loader)
    predictors = predictor_builder.build_all_predictors(factors)

    bayesian = BayesianPredictor(prior_r2=0.01)
    all_predictions = bayesian.compute_all_predictions(factors, predictors)

    allocator = BlackLittermanAllocator(target_te=0.02)

    results = {}
    for pred_name, predictions in all_predictions.items():
        strategy_ret, bench_ret = allocator.run_backtest(factors, predictions)
        results[pred_name] = {
            'strategy': strategy_ret,
            'benchmark': bench_ret
        }

    return factors, predictors, results


def plot_cumulative_returns(results, save=True):
    """
    Figure 1: Cumulative returns of strategies vs benchmark
    Similar to Figure 3 in the paper
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    # Benchmark
    benchmark = results[list(results.keys())[0]]['benchmark']
    cum_bench = (1 + benchmark.dropna()).cumprod()
    ax.plot(cum_bench.index, cum_bench.values, 'k--', linewidth=2, label='EW Benchmark', alpha=0.7)

    # Top strategies
    top_strategies = ['CFNAI', 'Inflation', 'ShortRate', 'TS_Mom']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for strat, color in zip(top_strategies, colors):
        if strat in results:
            cum_ret = (1 + results[strat]['strategy'].dropna()).cumprod()
            ax.plot(cum_ret.index, cum_ret.values, linewidth=1.5, label=f'BL.{strat}', color=color)

    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return (Growth of $1)')
    ax.set_title('Cumulative Returns: Factor Timing Strategies vs Benchmark')
    ax.legend(loc='upper left')
    ax.set_yscale('log')

    # Format x-axis
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)

    plt.tight_layout()
    if save:
        plt.savefig('fig1_cumulative_returns.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_information_ratios(results, save=True):
    """
    Figure 2: Information Ratios by Strategy
    Similar to Figure 1 in the paper
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Compute IRs
    ir_data = {}
    for pred_name, res in results.items():
        ir = PerformanceAnalyzer.information_ratio(res['strategy'], res['benchmark'])
        if pd.notna(ir):
            ir_data[pred_name] = ir

    # Sort by IR
    sorted_data = dict(sorted(ir_data.items(), key=lambda x: x[1], reverse=True))

    names = list(sorted_data.keys())
    values = list(sorted_data.values())

    # Color bars based on significance (IR > 0.5)
    colors = ['#2ca02c' if v > 0.5 else '#1f77b4' if v > 0 else '#d62728' for v in values]

    bars = ax.barh(names, values, color=colors, edgecolor='black', linewidth=0.5)

    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.axvline(x=0.5, color='red', linewidth=1, linestyle='--', alpha=0.5, label='IR = 0.5 threshold')

    ax.set_xlabel('Information Ratio')
    ax.set_ylabel('Strategy')
    ax.set_title('Information Ratios by Predictor Variable')
    ax.legend()

    plt.tight_layout()
    if save:
        plt.savefig('fig2_information_ratios.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_factor_returns(factors, save=True):
    """
    Figure 3: Factor Returns Heatmap
    Shows annual returns by factor
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    # Compute annual returns
    factors_clean = factors.dropna(how='all')
    factors_clean.index = pd.to_datetime(factors_clean.index)
    annual_returns = factors_clean.resample('Y').apply(lambda x: (1 + x).prod() - 1) * 100

    # Create heatmap
    im = ax.imshow(annual_returns.T.values, cmap='RdYlGn', aspect='auto', vmin=-30, vmax=30)

    # Labels
    ax.set_yticks(range(len(annual_returns.columns)))
    ax.set_yticklabels(annual_returns.columns)

    years = annual_returns.index.year
    ax.set_xticks(range(0, len(years), 5))
    ax.set_xticklabels(years[::5])

    ax.set_xlabel('Year')
    ax.set_ylabel('Factor')
    ax.set_title('Annual Factor Returns (%)')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Return (%)')

    plt.tight_layout()
    if save:
        plt.savefig('fig3_factor_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_subperiod_performance(results, save=True):
    """
    Figure 4: Information Ratios by Subperiod
    Similar to Table 3 in the paper
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    subperiods = {
        '1973-1989': ('1973-01-01', '1989-12-31'),
        '1990-2006': ('1990-01-01', '2006-12-31'),
        '2007-2018': ('2007-01-01', '2018-12-31'),
    }

    for ax, (period_name, (start, end)) in zip(axes, subperiods.items()):
        ir_data = {}

        for pred_name, res in results.items():
            strat_period = res['strategy'].loc[start:end]
            bench_period = res['benchmark'].loc[start:end]

            if len(strat_period.dropna()) < 12:
                continue

            ir = PerformanceAnalyzer.information_ratio(strat_period, bench_period)
            if pd.notna(ir):
                ir_data[pred_name] = ir

        # Sort and plot
        sorted_data = dict(sorted(ir_data.items(), key=lambda x: x[1], reverse=True))

        names = list(sorted_data.keys())
        values = list(sorted_data.values())

        colors = ['#2ca02c' if v > 0.5 else '#1f77b4' if v > 0 else '#d62728' for v in values]

        ax.barh(names, values, color=colors, edgecolor='black', linewidth=0.5)
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.axvline(x=0.5, color='red', linewidth=1, linestyle='--', alpha=0.5)
        ax.set_xlabel('Information Ratio')
        ax.set_title(period_name)

    axes[0].set_ylabel('Strategy')
    fig.suptitle('Information Ratios by Subperiod', fontsize=14, y=1.02)

    plt.tight_layout()
    if save:
        plt.savefig('fig4_subperiod_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_rolling_ir(results, window=36, save=True):
    """
    Figure 5: Rolling Information Ratios
    Shows stability of outperformance over time
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    top_strategies = ['CFNAI', 'Inflation', 'ShortRate', 'TS_Mom']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for strat, color in zip(top_strategies, colors):
        if strat in results:
            active_ret = results[strat]['strategy'] - results[strat]['benchmark']

            # Rolling IR
            rolling_mean = active_ret.rolling(window=window).mean() * 12
            rolling_std = active_ret.rolling(window=window).std() * np.sqrt(12)
            rolling_ir = rolling_mean / rolling_std

            ax.plot(rolling_ir.index, rolling_ir.values, linewidth=1.5, label=f'BL.{strat}', color=color)

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axhline(y=0.5, color='red', linewidth=1, linestyle='--', alpha=0.5, label='IR = 0.5')

    ax.set_xlabel('Date')
    ax.set_ylabel(f'Rolling {window}-Month Information Ratio')
    ax.set_title('Rolling Information Ratios of Top Strategies')
    ax.legend(loc='upper left')

    # Format x-axis
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)

    ax.set_ylim(-3, 5)

    plt.tight_layout()
    if save:
        plt.savefig('fig5_rolling_ir.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_drawdowns(results, save=True):
    """
    Figure 6: Drawdown Analysis
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    # Benchmark drawdown
    benchmark = results[list(results.keys())[0]]['benchmark'].dropna()
    cum_bench = (1 + benchmark).cumprod()
    rolling_max_bench = cum_bench.expanding().max()
    dd_bench = (cum_bench / rolling_max_bench - 1) * 100
    ax.fill_between(dd_bench.index, 0, dd_bench.values, alpha=0.3, color='gray', label='EW Benchmark')

    # Best strategy drawdown
    strat = 'Inflation'
    if strat in results:
        cum_strat = (1 + results[strat]['strategy'].dropna()).cumprod()
        rolling_max_strat = cum_strat.expanding().max()
        dd_strat = (cum_strat / rolling_max_strat - 1) * 100
        ax.plot(dd_strat.index, dd_strat.values, linewidth=1.5, color='#1f77b4', label=f'BL.{strat}')

    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.set_title('Drawdown Analysis: Benchmark vs Best Strategy')
    ax.legend(loc='lower left')

    # Format x-axis
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)

    plt.tight_layout()
    if save:
        plt.savefig('fig6_drawdowns.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_summary_table(results, save=True):
    """
    Create summary table similar to Table 1 in the paper
    """
    summary = []

    benchmark = results[list(results.keys())[0]]['benchmark']

    for pred_name, res in results.items():
        strat = res['strategy']
        bench = res['benchmark']
        active = strat - bench

        metrics = {
            'Strategy': f'BL.{pred_name}',
            'Ann. Return (%)': PerformanceAnalyzer.annualized_return(strat) * 100,
            'Ann. Vol (%)': PerformanceAnalyzer.annualized_volatility(strat) * 100,
            'Sharpe': PerformanceAnalyzer.sharpe_ratio(strat),
            'Active Return (%)': active.dropna().mean() * 12 * 100,
            'Track. Error (%)': active.dropna().std() * np.sqrt(12) * 100,
            'Info Ratio': PerformanceAnalyzer.information_ratio(strat, bench),
            't-stat': PerformanceAnalyzer.t_statistic(active),
            'Max DD (%)': PerformanceAnalyzer.max_drawdown(strat) * 100,
        }
        summary.append(metrics)

    df = pd.DataFrame(summary).set_index('Strategy')
    df = df.sort_values('Info Ratio', ascending=False)

    if save:
        df.to_csv('summary_table.csv')
        print("\nSummary table saved to 'summary_table.csv'")

    print("\n" + "=" * 100)
    print("DETAILED PERFORMANCE SUMMARY")
    print("=" * 100)
    print(df.round(2).to_string())

    return df


if __name__ == "__main__":
    print("Loading and computing results...")
    factors, predictors, results = load_results()

    print("\nCreating visualizations...")

    # Create all figures
    plot_cumulative_returns(results)
    plot_information_ratios(results)
    plot_factor_returns(factors)
    plot_subperiod_performance(results)
    plot_rolling_ir(results)
    plot_drawdowns(results)

    # Create summary table
    summary_df = create_summary_table(results)

    print("\n" + "=" * 60)
    print("All visualizations have been saved!")
    print("=" * 60)
    print("""
    Generated files:
    - fig1_cumulative_returns.png
    - fig2_information_ratios.png
    - fig3_factor_heatmap.png
    - fig4_subperiod_analysis.png
    - fig5_rolling_ir.png
    - fig6_drawdowns.png
    - summary_table.csv
    """)
