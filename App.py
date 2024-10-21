import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from Evaluator import Evaluator
from utils import load_signal, create_calendar_plot
from yre.data import DailyDB
from yin.common.numbers import Formatter

st.set_page_config(page_title="Signal Evaluator", layout="wide", page_icon="🧊")

# Add custom CSS for compact layout
st.markdown("""
<style>
    .reportview-container .main .block-container {
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
    }
    .stPlotlyChart {
        margin-bottom: 0.5rem;
    }
    .row-widget.stButton {
        text-align: center;
    }
    .dataframe {
        font-size: 0.5rem !important;
    }
    h1, h2, h3, h4, h5, h6 {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_database():
    return DailyDB('US').clip_date('2020-01-01')

@st.cache_data
def get_signal(file_path, version=0):
    return load_signal(file_path)

@st.cache_resource
def get_evaluator(_db, signal, version=0):
    return Evaluator(save_path='./results', db=_db, signal=signal)

@st.cache_data
def run_overview(_evaluator, version=0):
    return _evaluator.Overview()

@st.cache_data
def run_ls_decoder(_evaluator, window, version=0):
    return _evaluator.LS_Decoder(window=window)

@st.cache_data
def run_sector_decoder(_evaluator, selected_sector, version=0):
    return _evaluator.Sector_Decoder(selected_sector)

@st.cache_data
def run_asset_decoder(_evaluator, topk, btmk, version=0):
    return _evaluator.Asset_Decoder(topk, btmk)

@st.cache_data
def run_calendar_decoder(_evaluator, by, version=0):
    return _evaluator.Calendar_Decoder(by)

@st.cache_data
def run_shift_decoder(_evaluator, n, version=0):
    return _evaluator.Shift_Decoder(n)

@st.cache_data
def run_mktcap_decoder(_evaluator, version=0):
    return _evaluator.MktCap_Decoder()


@st.cache_data
def run_cap_estimation1(_evaluator, pct, version=0):
    return _evaluator.Estimate_Cap(pct=pct)
@st.cache_data
def run_cap_estimation2(_evaluator, max_multiplier, market_impact_threshold, version=0):
    return _evaluator.Estimate_Cap2(max_multiplier, market_impact_threshold)

def main():
    st.title("Signal Evaluator")

    # Initialize database and load signal
    db = get_database()
    signal = get_signal('signal.csv')

    # Initialize Evaluator
    evaluator = get_evaluator(db, signal)

    # st.header("Overview")

    # Estimate strategy capacity
    st.subheader("Estimated Strategy Capacity")
    col1, col2 = st.columns(2)
    with col1:
        max_multiplier = st.slider("Maximum capital multiplier", min_value=1.0, max_value=5000.0, value=4000.0, step=1.0)
    with col2:
        market_impact_threshold = st.slider("Market impact threshold", min_value=0.01, max_value=0.1, value=0.02, step=0.01)
    est_cap2, fig = run_cap_estimation2(evaluator, max_multiplier, market_impact_threshold)
    st.write(f"Capacity estimation: {est_cap2:.2f} Million")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Long-Short Decoder")
    window = st.slider("Rolling Window", min_value=1, max_value=100, value=30, step=1)
    metrics, fig, fig1 = run_ls_decoder(evaluator, window=window, version=10)
    col1, col2 = st.columns([3, 5])
    with col1:
        st.dataframe(metrics)
    with col2:
        st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig, use_container_width=True)


    st.header("Sector Decoder")
    sector_columns = ['gics_sector', 'gics_ind', 'bics_sector', 'bics_sector_sm', 'bics_samart_sector', 'lse_sector']
    selected_sector = st.selectbox("Choose Sector Classification", sector_columns)
    fig0, time_series, metrics = run_sector_decoder(evaluator, selected_sector)

    st.dataframe(metrics)
    st.plotly_chart(fig0, use_container_width=True)
    st.plotly_chart(time_series, use_container_width=True)


    st.header("Asset Decoder")
    col1, col2 = st.columns(2)
    with col1:
        topk = st.number_input("Top K Assets", min_value=1, max_value=20, value=10, step=1)   
    with col2:
        btmk = st.number_input("Bottom K Assets", min_value=1, max_value=20, value=10, step=1)
    fig, comparison = run_asset_decoder(evaluator, topk=topk, btmk=btmk)
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Metrics Comparison")
    st.dataframe(comparison)


    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("##### Day of Week Analysis")
        metrics = run_calendar_decoder(evaluator, by='dow')
        fig = create_calendar_plot(metrics[0], 'Mean PnL and Sharpe by Day of Week', 'Day of Week')
        st.plotly_chart(fig, use_container_width=True)
        # formatting matrics
        metrics[0]['mean'] = metrics[0]['mean'].map(Formatter.readable(2))
        metrics[0]['std'] = metrics[0]['std'].map(Formatter.readable(2))
        metrics[0]['sharpe'] = metrics[0]['sh'].map(Formatter.readable(2))
        metrics[0].drop(columns=['sh'], inplace=True)
        metrics[0]['ob'] = metrics[0]['ob'].astype(int)
        st.dataframe(metrics[0])

    metrics = run_calendar_decoder(evaluator, by='dom')
    with col2:
        st.write("##### Day of Month Analysis 1")
        fig = create_calendar_plot(metrics[0], 'Mean PnL and Sharpe by Day of Month', 'Day of Month')
        st.plotly_chart(fig, use_container_width=True)
        metrics[0]['mean'] = metrics[0]['mean'].map(Formatter.readable(2))
        metrics[0]['std'] = metrics[0]['std'].map(Formatter.readable(2))
        metrics[0]['sharpe'] = metrics[0]['sh'].map(Formatter.readable(2))
        metrics[0].drop(columns=['sh'], inplace=True)
        metrics[0]['ob'] = metrics[0]['ob'].astype(int)
        st.dataframe(metrics[0])

    with col3:
        st.write("##### Day of Month Analysis 2")
        fig = create_calendar_plot(metrics[1], 'Mean PnL and Sharpe by Day of Month', 'Day of Month') # type: ignore
        st.plotly_chart(fig, use_container_width=True)
        metrics[1]['mean'] = metrics[1]['mean'].map(Formatter.readable(2)) # type: ignore
        metrics[1]['std'] = metrics[1]['std'].map(Formatter.readable(2)) # type: ignore
        metrics[1]['sharpe'] = metrics[1]['sh'].map(Formatter.readable(2)) # type: ignore
        metrics[1].drop(columns=['sh'], inplace=True) # type: ignore
        metrics[1]['ob'] = metrics[1]['ob'].astype(int) # type: ignore
        st.dataframe(metrics[1]) # type: ignore
    st.info("""
    This analysis shows the mean PnL and Sharpe for each day of the week/month.
    - Day of Month Analysis 1: shows the mean PnL and Sharpe for each day of the month if we count the trading days from the start of the month.
    - Day of Month Analysis 2: shows the mean PnL and Sharpe for each day of the month if we count the trading days from the end of the month.
    """)

    st.header("Shift Decoder")
    fig_shift = run_shift_decoder(evaluator, n=10)
    st.plotly_chart(fig_shift, use_container_width=True)
    st.info("""
    This analysis shows the daily return and sharpe changes by shifting the signal.
    """)
            
    st.header("Market Cap Decoder")
    fig_mktcap = run_mktcap_decoder(evaluator)
    st.plotly_chart(fig_mktcap, use_container_width=True)
    st.info("""
    This analysis shows the mean PnL and Sharpe for each market cap group:
    - micro: <= 250 M
    - small: <= 2 B
    - mid: <= 10 B
    - large: <= 200 B
    - mega: > 200 B
    """)

if __name__ == "__main__":
    main()