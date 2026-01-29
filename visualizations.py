"""
Visualization script for paper replication results
"""

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from paper_replication import (
    DataLoader, FactorBuilder, PredictorBuilder,
    BayesianPredictor, BlackLittermanAllocator, PerformanceAnalyzer
)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def load_results():
    print("Computing results...")
    
    loader = DataLoader('DataGestionQuant.xlsx')
    factor_builder = FactorBuilder(loader)
    factors = factor_builder.build_all_factors()
    
    predictor_builder = PredictorBuilder(loader)
    predictors = predictor_builder.build_all_predictors(factors)
    
    bayesian = BayesianPredictor(prior_r2=0.01)
    all_predictions = bayesian.compute_all_predictions(factors, predictors)
    
    allocator = BlackLittermanAllocator(
        target_te=0.02,
        risk_aversion=5.0,
        view_confidence=0.50,
        transaction_cost=0.001
    )
    
    results = {}
    for pred_name, predictions in all_predictions.items():
        allocator.current_predictor = pred_name
        strategy_ret, bench_ret = allocator.run_backtest(factors, predictions)
        results[pred_name] = {
            'strategy': strategy_ret,
            'benchmark': bench_ret
        }
    
    return factors, predictors, results


def plot_cumulative_returns(results, save=True):
    fig, ax = plt.subplots(figsize=(14, 7))
    
    benchmark = results[list(results.keys())[0]]['benchmark']
    cum_bench = (1 + benchmark.dropna()).cumprod()
    ax.plot(cum_bench.index, cum_bench.values, 'k--', linewidth=2, label='EW Benchmark', alpha=0.7)
    
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
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save:
        plt.savefig('fig1_cumulative_returns.png', dpi=150, bbox_inches='tight')
    plt.close()


def bootstrap_ir(strategy_returns, benchmark_returns, n_bootstrap=1000, confidence=0.95):
    active_returns = (strategy_returns - benchmark_returns).dropna()
    n = len(active_returns)
    
    if n < 12:
        return np.nan, np.nan, np.nan
    
    mean_active = active_returns.mean() * 12
    te = active_returns.std() * np.sqrt(12)
    ir_estimate = mean_active / te if te > 0 else np.nan
    
    np.random.seed(42)
    bootstrap_irs = []
    
    for _ in range(n_bootstrap):
        sample_idx = np.random.choice(n, size=n, replace=True)
        sample = active_returns.iloc[sample_idx]
        sample_mean = sample.mean() * 12
        sample_te = sample.std() * np.sqrt(12)
        
        if sample_te > 0:
            bootstrap_irs.append(sample_mean / sample_te)
    
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_irs, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_irs, (1 - alpha/2) * 100)
    
    return ir_estimate, ci_lower, ci_upper


def plot_information_ratios(results, save=True):
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ir_data = {}
    for pred_name, res in results.items():
        ir, ci_low, ci_high = bootstrap_ir(res['strategy'], res['benchmark'])
        if pd.notna(ir):
            ir_data[pred_name] = {'ir': ir, 'ci_low': ci_low, 'ci_high': ci_high}
    
    sorted_names = sorted(ir_data.keys(), key=lambda x: ir_data[x]['ir'])
    x_positions = range(len(sorted_names))
    ir_values = [ir_data[name]['ir'] for name in sorted_names]
    ci_lows = [ir_data[name]['ci_low'] for name in sorted_names]
    ci_highs = [ir_data[name]['ci_high'] for name in sorted_names]
    errors_low = [ir - ci_low for ir, ci_low in zip(ir_values, ci_lows)]
    errors_high = [ci_high - ir for ir, ci_high in zip(ir_values, ci_highs)]
    
    ax.errorbar(x_positions, ir_values,
                yerr=[errors_low, errors_high],
                fmt='o', color='black', markersize=6,
                capsize=4, capthick=1.5, elinewidth=1.5, ecolor='black')
    
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.axhline(y=0.5, color='gray', linewidth=0.8, linestyle='--', alpha=0.7)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'BL.{name}' for name in sorted_names], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Info. Ratio', fontsize=11)
    ax.set_ylim(min(ci_lows) - 0.1, max(ci_highs) + 0.1)
    ax.yaxis.grid(True, linestyle='-', alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    
    if save:
        plt.savefig('fig2_information_ratios.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_information_ratios_horizontal(results, save=True):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ir_data = {}
    for pred_name, res in results.items():
        ir, ci_low, ci_high = bootstrap_ir(res['strategy'], res['benchmark'])
        if pd.notna(ir):
            ir_data[pred_name] = {'ir': ir, 'ci_low': ci_low, 'ci_high': ci_high}
    
    sorted_names = sorted(ir_data.keys(), key=lambda x: ir_data[x]['ir'], reverse=True)
    y_positions = range(len(sorted_names))
    ir_values = [ir_data[name]['ir'] for name in sorted_names]
    ci_lows = [ir_data[name]['ci_low'] for name in sorted_names]
    ci_highs = [ir_data[name]['ci_high'] for name in sorted_names]
    errors_low = [ir - ci_low for ir, ci_low in zip(ir_values, ci_lows)]
    errors_high = [ci_high - ir for ir, ci_high in zip(ir_values, ci_highs)]
    
    ax.errorbar(ir_values, y_positions,
                xerr=[errors_low, errors_high],
                fmt='o', color='black', markersize=8,
                capsize=4, capthick=1.5, elinewidth=1.5)
    
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.axvline(x=0.5, color='red', linewidth=1, linestyle='--', alpha=0.5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f'BL.{name}' for name in sorted_names])
    ax.set_xlabel('Information Ratio')
    ax.set_title('Information Ratios with 95% Bootstrap Confidence Intervals')
    ax.xaxis.grid(True, linestyle='-', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    
    if save:
        plt.savefig('fig2b_information_ratios_horizontal.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_factor_returns(factors, save=True):
    fig, ax = plt.subplots(figsize=(14, 10))
    
    factors_clean = factors.dropna(how='all')
    factors_clean.index = pd.to_datetime(factors_clean.index)
    annual_returns = factors_clean.resample('Y').apply(lambda x: (1 + x).prod() - 1) * 100
    
    im = ax.imshow(annual_returns.T.values, cmap='RdYlGn', aspect='auto', vmin=-30, vmax=30)
    ax.set_yticks(range(len(annual_returns.columns)))
    ax.set_yticklabels(annual_returns.columns)
    years = annual_returns.index.year
    ax.set_xticks(range(0, len(years), 5))
    ax.set_xticklabels(years[::5])
    ax.set_xlabel('Year')
    ax.set_ylabel('Factor')
    ax.set_title('Annual Factor Returns (%)')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Return (%)')
    plt.tight_layout()
    
    if save:
        plt.savefig('fig3_factor_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_subperiod_performance(results, save=True):
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
    fig, ax = plt.subplots(figsize=(14, 7))
    
    top_strategies = ['CFNAI', 'Inflation', 'ShortRate', 'TS_Mom']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for strat, color in zip(top_strategies, colors):
        if strat in results:
            active_ret = results[strat]['strategy'] - results[strat]['benchmark']
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
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)
    ax.set_ylim(-3, 5)
    plt.tight_layout()
    
    if save:
        plt.savefig('fig5_rolling_ir.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_drawdowns(results, save=True):
    fig, ax = plt.subplots(figsize=(14, 7))
    
    benchmark = results[list(results.keys())[0]]['benchmark'].dropna()
    cum_bench = (1 + benchmark).cumprod()
    rolling_max_bench = cum_bench.expanding().max()
    dd_bench = (cum_bench / rolling_max_bench - 1) * 100
    ax.fill_between(dd_bench.index, 0, dd_bench.values, alpha=0.3, color='gray', label='EW Benchmark')
    
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
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save:
        plt.savefig('fig6_drawdowns.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_summary_table(results, save=True):
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
    plot_cumulative_returns(results)
    plot_information_ratios(results)
    plot_information_ratios_horizontal(results)
    plot_factor_returns(factors)
    plot_subperiod_performance(results)
    plot_rolling_ir(results)
    plot_drawdowns(results)
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