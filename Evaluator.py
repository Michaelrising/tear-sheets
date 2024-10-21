import pandas as pd
import numpy as np
from typing import Union, Literal, Tuple, Dict, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from yre.data import DailyDB, IntraDB, RateDB
from yre.sim import Actual
from pathlib import Path
import plotly.colors as pc
import plotly.express as px
from yre.sim.helper import perf_by_dom, perf_by_dow
from utils import calhalfic
import bottleneck as bn
from yin.common import mat_util, to_array
from yin.common.numbers import Formatter

class Evaluator:
    def __init__(self, 
                 save_path: str,
                 db: Union[DailyDB, IntraDB, RateDB], 
                 signal: pd.DataFrame,
                 actual = None):
        self.db = db
        self.signal = signal.reindex(db['u'].index, axis=1) # type: ignore
        self.signal.index.name = 'date'
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

        # if actual is None:
        #     self.run_sim('dsim')
        # else:
        #     self.actual = actual
    
    def run_sim(self, method: str):
        if method.lower() == 'dsim':
            from yre.sim import DSim
            self.actual = DSim(self.signal,
                                self.db,
                                hedge_inst = self.db['hedge_inst'])
        else:
            pass

    def Overview(self):
        signals = self.signal.copy()
        lmv = signals.where(signals > 0)
        smv = -1 * signals.where(signals < 0)
        gmv = lmv + smv
        top_long_names = lmv.sum(axis=0).sort_values(ascending=False).head(10)
        top_short_names = smv.sum(axis=0).sort_values(ascending=False).head(10)
        top_long_names = top_long_names.rename('Capital')
        top_short_names = top_short_names.rename('Capital')

        return top_long_names, top_short_names, lmv.sum().sum(), smv.sum().sum()

    def LS_Decoder(self, signals=None, window: int=30) -> Tuple[pd.DataFrame, go.Figure, go.Figure]:
        if signals is None:
            signals = self.signal.copy()
        pnls = self._get_pnls(signals)
        metrics = self._get_metrics(pnls, signals)
        
        # Calculate daily PnLs
        long_pnl = pnls['long'].sum(axis=1) 
        short_pnl = pnls['short'].sum(axis=1) 
        total_pnl = pnls['total'].sum(axis=1)
        hedge_total = pnls['hedge_total'].sum(axis=1)
        lmv = signals.where(signals > 0).sum(axis=1)
        smv = -1 * signals.where(signals < 0).sum(axis=1)
        gmv = lmv + smv
        l_contrib = long_pnl / gmv
        s_contrib = short_pnl / gmv
        h_contrib = hedge_total / gmv

        # Create a DataFrame for contributions
        contributions = pd.DataFrame({
            'Long': l_contrib,
            'Short': s_contrib,
            'Hedge': h_contrib
        })

        # Create subplot figure
        fig = go.Figure()

        #Time Series with Rolling Window
        rolling_contributions = contributions.sort_index().rolling(window=window).mean()

        fig.add_trace(
            go.Scatter(x=rolling_contributions.index, y=rolling_contributions['Long'], 
                       name='Long (hedged)', fill='tozeroy', fillcolor='rgba(0,255,0,0.1)', line=dict(color='green'))
        )
        fig.add_trace(
            go.Scatter(x=rolling_contributions.index, y=rolling_contributions['Short'], 
                       name='Short (hedged)', fill='tozeroy', fillcolor='rgba(255,0,0,0.1)', line=dict(color='red'))
           
        )
        fig.add_trace(
            go.Scatter(x=rolling_contributions.index, y=rolling_contributions['Hedge'], 
                       name='Hedge', fill='tozeroy', fillcolor='rgba(0,0,255,0.1)', line=dict(color='blue'))
        )

        # Update layout
        fig.update_layout(
            height=400, 
            title_text='30-Day Rolling Average of Long/Short Contributions',
            showlegend=True,
            xaxis=dict(title='Date'),
            yaxis=dict(title='Contribution')
        )

        fig1 = go.Figure()
        fig1.add_trace(
            go.Bar(x=metrics.index, y=metrics['Ann Ret'], name='Annualized Return', marker_color='lightblue', yaxis='y2', width=0.3))
    
        fig1.add_trace(
            go.Scatter(x=metrics.index, y=metrics['Sharpe'], name='Sharpe', line=dict(color='orange'), yaxis='y', zorder=100)
        )
        fig1.update_layout(
            height=350, 
            # title='Sharpe and Mean Return',
            yaxis = dict(title='Sharpe', showgrid=False),
            yaxis2 = dict(title='Ann Ret', overlaying='y', side='right')
        )
        metrics_to_show = pd.DataFrame(index=metrics.index, columns = ['Daily Ret', 'Daily PnL', 'Ann Ret', 'Ann Vol', 'Sharpe'])
        metrics_to_show['Daily Ret'] = metrics['Daily Ret'].map(Formatter.bps(2))
        metrics_to_show['Daily PnL'] = metrics['Daily PnL'].map(Formatter.readable(1))
        metrics_to_show['Ann Ret'] = metrics['Ann Ret'].map(Formatter.percentage(2))
        metrics_to_show['Ann Vol'] = metrics['Ann Vol'].map(Formatter.percentage(2))
        metrics_to_show['Sharpe'] = metrics['Sharpe'].map(Formatter.readable(3))

        return metrics_to_show, fig, fig1
    
    def Sector_Decoder(self, sector_name: str, topk: int=10) -> Tuple[go.Figure, go.Figure, pd.DataFrame]:
        sector_metrics, sector_pnls, total_contribution, gmvs = self._decode_by_group(sector_name)
        topk_contribution = total_contribution[:topk]
        if topk < len(total_contribution):
            other = ('Others', sum([item[1] for item in total_contribution]) - sum([item[1] for item in total_contribution[:topk]]))
            topk_contribution.append(other)
        
        colors = px.colors.qualitative.Plotly[:len(topk_contribution)]

        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "bar"}, {"type": "pie"}]], 
                            column_widths=[0.6, 0.45], horizontal_spacing=0.05, subplot_titles=('Total PnL', 'GMV'))

        # Add bar chart for contribution
        fig.add_trace(go.Bar(
            x=[item[0] for item in topk_contribution],
            y=[item[1] for item in topk_contribution],
            name='Total PnL',
            marker_color=colors,
            width=0.5
        ), row=1, col=1)

        # Add pie chart for GMV
        gmv_values = [gmvs[item[0]] for item in topk_contribution[:-1]]  
        gmv_values.append(sum(gmvs.values()) - sum(gmv_values)) 
        fig.add_trace(go.Pie(
            labels=[item[0] for item in topk_contribution],
            values=gmv_values,
            textposition='inside',
            textinfo='percent+label',
            insidetextorientation='radial',
            marker_colors=colors,
            name='GMV'
        ), row=1, col=2)

        # Update layout
        fig.update_layout(
            title_text=f"Top {topk} {sector_name} PnL and GMV",
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="right",
                x=1.1
            ),
            height=400,
            bargap=0.15
        )


        fig.update_yaxes(title_text="Total PnL", row=1, col=1)

        fig.update_traces(showlegend=False, row=1, col=1)

        # Create time series plot
        time_series = go.Figure()
        for i, (sector, _) in enumerate(topk_contribution):
            if sector == 'Others':
                continue
            pnl = sector_pnls[sector].sum(axis=1)
            time_series.add_trace(go.Scatter(x=pnl.index, y=pnl.cumsum(), 
                                             mode='lines', name=sector, line=dict(color=colors[i])))
        time_series.update_layout(title_text=f"Top {topk} cumulative PnL by {sector_name}",
                                  xaxis_title="Date",
                                  yaxis_title="Cumulative PnL")
        sector_metrics = pd.concat(sector_metrics)
        sector_metrics = sector_metrics.sort_values(by='Sharpe', ascending=False)
        sector_metrics['Daily PnL'] = sector_metrics['Daily PnL'].map(Formatter.readable(1))
        sector_metrics['Daily Ret'] = sector_metrics['Daily Ret'].map(Formatter.bps(2))
        sector_metrics['Ann Ret'] = sector_metrics['Ann Ret'].map(Formatter.percentage(2))
        sector_metrics['Ann Vol'] = sector_metrics['Ann Vol'].map(Formatter.percentage(2))
        sector_metrics['Sharpe'] = sector_metrics['Sharpe'].map(Formatter.readable(3))
        sector_metrics.index.names = ['sector', 'side']
        sector_metrics = sector_metrics.query('side == "Total"').droplevel('side')
        return fig, time_series, sector_metrics

    def Asset_Decoder(self, topk: int=10, btmk: int=10):
        _, asset_pnls, total_contribution, _ = self._decode_by_group(pd.Series(self.signal.columns, index=self.signal.columns))
        topk_contribution = total_contribution[:topk]
        btmk_contribution = total_contribution[-btmk:]

        fig = go.Figure()

        # Generate color scales
        greens = pc.n_colors('rgb(255,255,255)', 'rgb(0,255,0)', topk, colortype='rgb')
        reds = pc.n_colors('rgb(255,255,255)', 'rgb(255,0,0)', btmk, colortype='rgb')

        # Plot top contributors
        for i, (asset, _) in enumerate(reversed(topk_contribution)):
            pnl = asset_pnls[asset].sum(axis=1)
            fig.add_trace(go.Scatter(
                x=pnl.index, 
                y=pnl.cumsum(),
                mode='lines', 
                name=asset,
                line=dict(color=greens[i]),  # type: ignore
                legendgroup='top'
            ))

        # Plot bottom contributors
        for i, (asset, _) in enumerate(btmk_contribution):
            pnl = asset_pnls[asset].sum(axis=1)
            fig.add_trace(go.Scatter(
                x=pnl.index, 
                y=pnl.cumsum(),
                mode='lines', 
                name=asset,
                line=dict(color=reds[i]),  # type: ignore
                legendgroup='bottom'
            ))

        fig.update_layout(
            title_text=f"Top {topk} and Bottom {btmk} Assets cumulative PnL",
            xaxis_title="Date",
            yaxis_title="Cumulative PnL",
            legend_title_text="Assets",
            legend=dict(
                groupclick="toggleitem",
                tracegroupgap=10,
                entrywidth=0.1,
                entrywidthmode='fraction',
                orientation='v'
            )
        )
        # if we romove the topk or btmk names, how the metrics will change
        topk_names = [item[0] for item in topk_contribution]
        btmk_names = [item[0] for item in btmk_contribution]
        pnls = self._get_pnls(self.signal)
        metrics = self._get_metrics(pnls, self.signal, pd.DataFrame(True, index=self.signal.index, columns=self.signal.columns))
        topk_mask = pd.DataFrame(True, index=self.signal.index, columns=self.signal.columns)
        topk_mask[topk_names] = False
        btmk_mask = pd.DataFrame(True, index=self.signal.index, columns=self.signal.columns)
        btmk_mask[btmk_names] = False
        rest_mask = topk_mask & btmk_mask

        metrics_without_top = self._get_metrics(pnls, self.signal, mask=topk_mask) 
        metrics_without_bottom = self._get_metrics(pnls, self.signal, mask=btmk_mask) 
        metrics_without_top_or_btm = self._get_metrics(pnls, self.signal, mask=rest_mask) 


        comparison = pd.DataFrame({
            'All Assets': metrics.loc['Total'],
            f'w/o Top {topk}': metrics_without_top.loc['Total'],
            f'w/o Btm {btmk}': metrics_without_bottom.loc['Total'],
            f'w/o Top {topk} and Btm {btmk}': metrics_without_top_or_btm.loc['Total']
        })
        comparison['% Change (w/o Top 10)'] = (comparison[f'w/o Top {topk}'] - comparison['All Assets']) / comparison['All Assets'] 
        comparison['% Change (w/o Btm 10)'] = (comparison[f'w/o Btm {btmk}'] - comparison['All Assets']) / comparison['All Assets']
        comparison['% Change (w/o Top 10 and Btm 10)'] = (comparison[f'w/o Top {topk} and Btm {btmk}'] - comparison['All Assets']) / comparison['All Assets']

        # formatting
        comparison.loc['Daily PnL', ['All Assets', 'w/o Top 10', 'w/o Btm 10', 'w/o Top 10 and Btm 10']] = comparison.loc['Daily PnL', ['All Assets', 'w/o Top 10', 'w/o Btm 10', 'w/o Top 10 and Btm 10']].map(Formatter.readable(2)) # type: ignore
        comparison.loc['Daily Ret', ['All Assets', 'w/o Top 10', 'w/o Btm 10', 'w/o Top 10 and Btm 10']] = comparison.loc['Daily Ret', ['All Assets', 'w/o Top 10', 'w/o Btm 10', 'w/o Top 10 and Btm 10']].map(Formatter.bps(2)) # type: ignore
        comparison.loc['Ann Ret', ['All Assets', 'w/o Top 10', 'w/o Btm 10', 'w/o Top 10 and Btm 10']] = comparison.loc['Ann Ret', ['All Assets', 'w/o Top 10', 'w/o Btm 10', 'w/o Top 10 and Btm 10']].map(Formatter.percentage(2)) # type: ignore
        comparison.loc['Ann Vol', ['All Assets', 'w/o Top 10', 'w/o Btm 10', 'w/o Top 10 and Btm 10']] = comparison.loc['Ann Vol', ['All Assets', 'w/o Top 10', 'w/o Btm 10', 'w/o Top 10 and Btm 10']].map(Formatter.percentage(2)) # type: ignore
        comparison.loc['Sharpe', ['All Assets', 'w/o Top 10', 'w/o Btm 10', 'w/o Top 10 and Btm 10']] = comparison.loc['Sharpe', ['All Assets', 'w/o Top 10', 'w/o Btm 10', 'w/o Top 10 and Btm 10']].map(Formatter.readable(3)) # type: ignore
        comparison['% Change (w/o Top 10)'] = comparison['% Change (w/o Top 10)'].map(Formatter.percentage(2))
        comparison['% Change (w/o Btm 10)'] = comparison['% Change (w/o Btm 10)'].map(Formatter.percentage(2))
        comparison['% Change (w/o Top 10 and Btm 10)'] = comparison['% Change (w/o Top 10 and Btm 10)'].map(Formatter.percentage(2))
        return fig, comparison
    
    def Calendar_Decoder(self, by: str='dow'):
        pnls = self._get_pnls(self.signal)
        pnl = pnls['total'].sum(axis=1)
        if by == 'dow':
            metrics = perf_by_dow(pnl=pnl)
            metrics = (metrics.droplevel(-1), )
        elif by == 'dom':
            metrics = perf_by_dom(pnl=pnl)
            metrics = (metrics[0].droplevel(-1), metrics[1].droplevel(-1))
        else:
            raise ValueError("Invalid 'by' parameter. Choose 'dow' or 'dom'.")

        return metrics
    
    def Shift_Decoder(self, n: int=10):
        _shifted_res = {}
        for i in range(0, n+1):  
            shifted_sigs = self.signal.shift(i).copy()
            pnls = self._get_pnls(shifted_sigs)
            metrics = self._get_metrics(pnls, shifted_sigs)
            _shifted_res[i] = metrics

        fig = go.Figure()
        _shifted_res = pd.concat(_shifted_res)
        _shifted_res = _shifted_res.xs('Total', level=-1)
        _shifted_res.index.name = 'shift_days'
        
        sharpe_half_life = n / np.log2(_shifted_res['Sharpe'].iloc[0] / _shifted_res['Sharpe'].iloc[-1])
        fig.add_trace(go.Bar(x=_shifted_res.index.get_level_values('shift_days'), 
                            y=_shifted_res['Daily Ret'], width=0.5,name='Mean Daily Return', marker_color='lightblue', yaxis='y'))
        fig.add_trace(go.Scatter(x=_shifted_res.index.get_level_values('shift_days'), 
                                 y=_shifted_res['Sharpe'], mode='lines', name='Sharpe', line=dict(color='orange'), yaxis='y2'))

        fig.update_layout(
            xaxis=dict(title='Shift Days'),
            yaxis=dict(title='Mean Daily Return'),
            yaxis2=dict(title='Sharpe', overlaying='y', side='right', showgrid=False), 
            legend=dict(x=0.85, y=0.99, bgcolor='rgba(255,255,255,0.5)')
        )

        return fig
    
    def MktCap_Decoder(self):
        mega_cap = 200 * 1e3
        large_cap = 10 * 1e3
        mid_cap = 2000
        small_cap = 250

        pnls = self._get_pnls(self.signal)
        mktcap: pd.DataFrame = self.db['mkt_cap_usd'] # type: ignore
        mktcap = mktcap[self.signal.columns]  # Filter for relevant stocks

        # Define the groups and bins
        groups = ['micro', 'small', 'mid', 'large', 'mega']
        bins = [0, small_cap, mid_cap, large_cap, mega_cap, np.inf]

        mktcap_groups = pd.DataFrame(index=mktcap.index, columns=mktcap.columns)
        for date in mktcap.index:
            mktcap_groups.loc[date] = pd.cut(mktcap.loc[date].fillna(0), bins=bins, labels=groups)

        total_pnl = pnls['total']
        grouped_pnls = {group: total_pnl.where(mktcap_groups == group) for group in groups}
        total_contribution = {group: pnl.sum().sum() for group, pnl in grouped_pnls.items()}

        fig = make_subplots(rows=1, cols=2, subplot_titles=('Total PnL by Market Cap', 'PnL Distribution by Market Cap'))

        # Bar chart
        fig.add_trace(
            go.Bar(
                x=groups,
                y=[total_contribution[key] for key in groups],
                width=0.35,
                name='Total PnL',
                opacity=0.6
            ),
            row=1, col=1
        )

        # Box plot
        for group in groups:
            pnl = grouped_pnls[group]
            fig.add_trace(
                go.Box(y=pnl.sum(axis=1).values.flatten(), name=group, boxpoints=False, jitter=0.3, pointpos=-1.8, width=0.35),# type: ignore
                row=1, col=2,
            )

        # Update layout
        fig.update_layout(
            height=400,
            width=1000,
            showlegend=False
        )

        # Update y-axis 
        fig.update_yaxes(title_text="Total PnL", row=1, col=1)
        fig.update_yaxes(title_text="Daily PnL", row=1, col=2)

        return fig
    
    def Estimate_Cap(self, pct: float=0.02, vol=None):
        trd_tvr = self.signal.copy().fillna(0).diff()
        trd_tvr.iloc[0] = self.signal.copy().fillna(0).iloc[0]
        mkt_tvr = self.db['tvr_usd']
        mkt_tvr = mkt_tvr.reindex_like(trd_tvr).values # type: ignore
        net_trd_tvr = trd_tvr.abs()#.groupby(trd_tvr.index.normalize()).sum() # type: ignore
        net_trd_tvr_abs = net_trd_tvr.abs().values
        adv_win_start = (np.arange(len(net_trd_tvr))[bn.nansum(net_trd_tvr_abs, axis=1) > mat_util.epsilon])[0]
        if vol is not None:
            net_trd_tvr_abs = net_trd_tvr_abs * vol.reindex_like(net_trd_tvr).values
        tvr = bn.nansum(net_trd_tvr_abs[adv_win_start:])
        # esp = np.maximum(pct, np.minimum(pct, net_trd_tvr_abs[adv_win_start:] / mkt_tvr[adv_win_start:]))
        est_cap = (bn.nansum(pct * mkt_tvr[adv_win_start:] * net_trd_tvr_abs[adv_win_start:]) / tvr) if tvr != 0 else np.nan
        est_cap_in_million = est_cap / 1e6
        return est_cap_in_million

    def Estimate_Cap2(self, max_multiplier: float = 10, market_impact_threshold: float = 0.05, long_cost: float = 0.0010, short_cost: float = 0.0015):
        base_signal = self.signal.copy()
        base_pnls = self._get_pnls(base_signal)
        base_metrics = self._get_metrics(base_pnls, base_signal)
        base_sharpe = base_metrics.loc['Total', 'Sharpe']

        multipliers = np.linspace(1, max_multiplier, 40)
        sharpes = []
        capacities = []

        mkt_tvr = self.db['tvr_usd']
        mkt_tvr = mkt_tvr.reindex_like(base_signal) # type: ignore

        for multiplier in multipliers:
            scaled_signal = base_signal * multiplier
            scaled_tvr = scaled_signal.diff().abs()
            
            # Apply market impact threshold
            max_allowed_tvr = mkt_tvr * market_impact_threshold
            capped_tvr = np.minimum(scaled_tvr, max_allowed_tvr)
            
            # Calculate the actual multiplier after capping
            _multiplier = np.where(scaled_tvr != 0, capped_tvr / scaled_tvr, 0)
            capped_signal = scaled_signal * _multiplier
            
            capped_pnls = self._get_pnls(capped_signal)
            capped_metrics = self._get_metrics(capped_pnls, capped_signal)
            capped_sharpe = capped_metrics.loc['Total', 'Sharpe']
            sharpes.append(capped_sharpe)
            
            # Calculate capacity
            capacity = bn.nansum(abs(capped_tvr), axis=1)[1:].mean()
            capacities.append(capacity)


        # Find the point where performance starts to decay significantly
        decay_point = np.argmax(sharpes)
        optimal_multiplier = multipliers[decay_point]
        est_cap = capacities[decay_point] 
        est_cap_in_million = est_cap / 1e6

        # Create a plot of Sharpe ratio vs. capacity
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=capacities[:len(sharpes)], y=sharpes, mode='lines+markers'))
        fig.add_vline(x=est_cap, line_dash="dash", line_color="red")
        fig.update_layout(
            title="Sharpe Ratio vs. Daily Tvr",
            xaxis_title="Daily Tvr (M USD)",
            yaxis_title="Sharpe Ratio",
            height=350,
            annotations=[
                dict(
                    x=est_cap,
                    y=max(sharpes),
                    xref="x",
                    yref="y",
                    text=f"Optimal: {est_cap_in_million:.2f}M",
                    showarrow=True,
                    arrowhead=7,
                    ax=0,
                    ay=-40
                )
            ]
        )

        return est_cap_in_million, fig

    def _decode_by_group(self, group: Union[str, pd.Series], signal=None):
        if isinstance(group, str):
            info: pd.DataFrame = self.db['u']  # type: ignore
            try:
                group_info = info[group]
                group_info = group_info.dropna()
                group_info[group_info == ''] = f'Not Classified by {group}'
            except KeyError:
                raise ValueError(f"Group {group} not found in database")
        else:
            group_info = group
        if signal is None:
            sigs = self.signal.copy()
        else:
            sigs = signal.copy()
        sigs = sigs.reindex(group_info.index, axis=1)
        # Calculate PnLs
        pnls = self._get_pnls(sigs)
        total_pnl = pnls['total']
        # get the metrics for each sector
        group_metrics = {}
        gmvs = {}
        for group in group_info.unique():
            this_group_names = group_info[group_info == group].index
            this_group_pnl = {k: v[this_group_names] for k, v in pnls.items()}
            group_metrics[group] = self._get_metrics(this_group_pnl, sigs[this_group_names])
            gmvs[group] = sigs[this_group_names].abs().sum().sum()
        # Group PnLs by sector
        group_pnls = {}
        for group in group_info.unique():
            group_stocks = group_info[group_info == group].index
            group_pnls[group] = total_pnl[group_stocks]#.sum(axis=1)

        # Calculate overall contribution
        total_contribution = {group: pnl.sum().sum() for group, pnl in group_pnls.items()}
        total_contribution = sorted(total_contribution.items(), key=lambda x: x[1], reverse=True)

        return group_metrics, group_pnls, total_contribution, gmvs

    def _get_pnls(self, sigs: pd.DataFrame) -> Dict[str, pd.DataFrame]:

        _hedged_ret = self.db['resid']
        _hedged_ret.index.name = 'date' # type: ignore
        assert sigs.columns.isin(_hedged_ret.columns).all(), 'signal must have the same equities as DB' # type: ignore
        sigs = sigs.reindex(_hedged_ret.columns, axis=1)# type: ignore
        long_pnl = sigs.where(sigs > 0).shift() * _hedged_ret
        short_pnl = sigs.where(sigs < 0).shift() * _hedged_ret
        total_pnl = long_pnl.add(short_pnl, fill_value=0)
        raw_ret = self.db['last'].pct_change() # type: ignore
        _hedged_inst_ret = raw_ret - _hedged_ret
        h_long_pnl = _hedged_inst_ret * sigs.where(sigs > 0).shift()
        h_short_pnl = _hedged_inst_ret * sigs.where(sigs < 0).shift()
        h_total_pnl = h_long_pnl.add(h_short_pnl, fill_value=0)

        return {'long': long_pnl, 'short': short_pnl, 'total': total_pnl, 'hedge_long': h_long_pnl, 'hedge_short': h_short_pnl, 'hedge_total': h_total_pnl}


    
    def _get_metrics(self, 
                    pnls: Dict[str, pd.DataFrame],
                    signal: pd.DataFrame,
                    mask=None):
        '''
        Calculate metrics for total, long, and short pnl.
        ''' 
        if mask is not None:
            total_pnl = pnls['total'] * mask
            long_pnl = pnls['long'] * mask
            short_pnl = pnls['short'] * mask
            hedge_total = pnls['hedge_total'] * mask
            signal = signal * mask
        else:
            total_pnl = pnls['total']
            long_pnl = pnls['long']
            short_pnl = pnls['short']
            hedge_total = pnls['hedge_total']

        daily_total_pnl = total_pnl.sum(axis=1)
        daily_long_pnl = long_pnl.sum(axis=1)
        daily_short_pnl = short_pnl.sum(axis=1) 
        daily_hedge_total = hedge_total.sum(axis=1)

        lmv = signal.where(signal > 0).sum(axis=1)
        smv = -1 * signal.where(signal < 0).sum(axis=1)
        gmv = lmv + smv
        daily_total_ret = daily_total_pnl / gmv
        daily_long_ret = daily_long_pnl / lmv
        daily_short_ret = daily_short_pnl / smv
        daily_hedge_ret = daily_hedge_total / gmv

        total_ann_ret = daily_total_ret.mean() * 252
        long_ann_ret = daily_long_ret.mean() * 252
        short_ann_ret = daily_short_ret.mean() * 252
        hedge_ann_ret = daily_hedge_ret.mean() * 252


        summary_stats = pd.DataFrame({
            'Daily PnL': [daily_total_pnl.mean(), daily_long_pnl.mean(), daily_short_pnl.mean(), daily_hedge_total.mean()],
            'Daily Ret': [daily_total_ret.mean(), daily_long_ret.mean(), daily_short_ret.mean(), daily_hedge_ret.mean()],
            'Ann Ret': [total_ann_ret, long_ann_ret, short_ann_ret, hedge_ann_ret],
            'Ann Vol': [daily_total_ret.std()* np.sqrt(252), daily_long_ret.std()* np.sqrt(252), 
                           daily_short_ret.std()* np.sqrt(252), daily_hedge_ret.std()* np.sqrt(252)],
            'Sharpe': [(daily_total_ret.mean() / daily_total_ret.std()) * np.sqrt(252),
                        (daily_long_ret.mean() / daily_long_ret.std()) * np.sqrt(252),
                        (daily_short_ret.mean() / daily_short_ret.std()) * np.sqrt(252),
                        (daily_hedge_ret.mean() / daily_hedge_ret.std()) * np.sqrt(252)],
        }, index=['Total', 'Long', 'Short', 'Hedge'])

        return summary_stats
