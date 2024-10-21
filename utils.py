import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.graph_objects as go


def load_signal(signal_file):
    signal = pd.read_csv(signal_file, index_col=0, parse_dates=True)
    return signal

def plot_results(long_pnl, short_pnl, total_pnl, correlation, title_prefix=""):
    try:
        # Set the style
        sns.set_style("darkgrid")
        sns.set_palette("deep")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
        
        # Time series plot
        cumulative_returns = pd.DataFrame({
            'Total PnL': total_pnl.sum(axis=1).cumsum(),
            'Long PnL': long_pnl.sum(axis=1).cumsum(),
            'Short PnL': short_pnl.sum(axis=1).cumsum()
        })
        
        sns.lineplot(data=cumulative_returns, ax=ax1)
        ax1.set_title(f'{title_prefix} Cumulative PnL Time Series', fontsize=16)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('PnL', fontsize=12)
        ax1.legend(fontsize=10)
        # y tick lables set as scientific notation
        ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax1.tick_params(axis='x', which='major', labelsize=10, rotation=45)
        ax1.text(0.5, 0.95, f'Long-Short Correlation: {correlation:.4f}', 
                transform=ax1.transAxes, ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.5))
        
        # Add annotations for final values (without arrows), use scientific notation
        for column in cumulative_returns.columns:
            final_value = cumulative_returns[column].iloc[-1]
            ax1.text(1.01, final_value, f'{column}: {final_value:.2e}', 
                    transform=ax1.get_yaxis_transform(),
                    ha='left', va='center',
                    fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.35))
        
        # Histogram plot
        sns.histplot(data=total_pnl.sum(axis=1), kde=True, ax=ax2, label='Long-Short', color='purple', stat='density')
        sns.histplot(data=long_pnl.sum(axis=1), kde=True, ax=ax2, label='Long', color='green', stat='density')
        sns.histplot(data=short_pnl.sum(axis=1), kde=True, ax=ax2, label='Short', color='red', stat='density')
        ax2.set_title('PnL Distribution', fontsize=16)
        ax2.set_xlabel('Daily PnL', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.tick_params(axis='both', which='major', labelsize=10)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    
    except Exception as e:
        st.error(f"An error occurred while plotting: {str(e)}")
        st.write("Displaying raw data instead:")
        st.write(cumulative_returns)

def calculate_max_drawdown(cumulative_returns):
    peak = cumulative_returns.sort_index().expanding(min_periods=1).max()
    df = pd.DataFrame({'peak': peak, 'cumulative_returns': cumulative_returns})
    df = df.query('peak > 0')   
    mdd = ((df['peak'] - df['cumulative_returns']) / df['peak']).max() * -100
    return mdd

def calculate_rolling_sharpe(pnl, window):
    rolling_mean = pnl.mean(axis=1).rolling(window=window).mean()
    rolling_std = pnl.mean(axis=1).rolling(window=window).std()
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
    return rolling_sharpe

def plot_single_asset_results(asset_results, asset):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
    
    # Cumulative PnL plot
    cumulative_returns = pd.DataFrame({
        asset: asset_results['target_pnl'].cumsum(),
        'Total': asset_results['total_pnl'].cumsum(),
        'Rest': asset_results['rest_pnl'].cumsum()
    })
    
    sns.lineplot(data=cumulative_returns, ax=ax1)
    ax1.set_title(f'Cumulative PnL: {asset_results["asset"]} vs Total Portfolio', fontsize=16)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Cumulative PnL', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    
    # Daily returns distribution plot
    daily_returns = pd.DataFrame({
        'Asset': asset_results['target_pnl'],
        'Total': asset_results['total_pnl'],
        'Rest': asset_results['rest_pnl']
    })
    
    sns.histplot(data=daily_returns, kde=True, ax=ax2, stat='density')
    ax2.set_title('Daily PnL Distribution', fontsize=16)
    ax2.set_xlabel('Daily PnL', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout()
    st.pyplot(fig)


def calhalfic(x):
    # Implement the calhalfic function if it's not defined elsewhere
    # This is a placeholder implementation, adjust as needed
    return np.log(2) / np.log(1 + x.iloc[0] / x.iloc[-1])

def create_calendar_plot(metrics, title, x_title):
    fig = go.Figure()
    
    # Add Mean PnL bars
    fig.add_trace(go.Bar(
        x=metrics.index,
        y=metrics['mean'],
        width=0.5,
        name='Mean PnL',
        marker_color='lightblue'
    ))
    
    # Add Sharpe ratio line
    fig.add_trace(go.Scatter(
        x=metrics.index,
        y=metrics['sh'],
        name='Sharpe',
        yaxis='y2',
        line=dict(color='orange', width=2)
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis=dict(
            title='Mean PnL',
            side='left',
            showgrid=False,
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=2
        ),
        yaxis2=dict(
            title='Sharpe',
            side='right',
            overlaying='y',
            showgrid=False,
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=2
        ),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.5)'),
        height=400
    )
    
    # Ensure both y-axes include zero and have the same zero point
    y_range = max(abs(min(metrics['mean'].min(), 0)), abs(max(metrics['mean'].max(), 0)))
    y2_range = max(abs(min(metrics['sh'].min(), 0)), abs(max(metrics['sh'].max(), 0)))
    
    fig.update_layout(
        yaxis=dict(range=[-y_range, y_range]),
        yaxis2=dict(range=[-y2_range, y2_range])
    )
    
    return fig