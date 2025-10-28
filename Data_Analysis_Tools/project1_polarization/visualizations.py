"""Visualization helpers for political polarization analysis."""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd
import plotly.graph_objects as go

__all__ = [
    "build_silhouette_shift_figure",
    "build_party_mismatch_figure",
]


def build_silhouette_shift_figure(
    enriched_df: pd.DataFrame,
    congress_dates_df: pd.DataFrame,
    *,
    session_subset: Optional[Iterable[str]] = None,
) -> go.Figure:
    """Generate a Plotly figure with rolling average and highlighted shifts.
    
    Args:
        enriched_df: DataFrame with session silhouette scores and metadata
        congress_dates_df: DataFrame with congress session date mappings
        session_subset: Optional iterable of session identifiers (zero padded
            strings or integers) to include in the plot. When provided, only
            matching sessions are rendered.
        
    Returns:
        Plotly Figure object with time-series visualization
    """
    figure = go.Figure()

    if enriched_df.empty:
        figure.update_layout(
            title="Silhouette Score with Rolling Average",
            xaxis_title="Year",
            yaxis_title="Silhouette Score",
        )
        return figure

    plot_df = enriched_df.copy()

    if session_subset:
        allowed_sessions = {str(item).zfill(3) for item in session_subset}
        plot_df = plot_df[plot_df["session_num"].isin(allowed_sessions)]

    if plot_df.empty:
        figure.update_layout(
            title="Silhouette Score with Rolling Average",
            xaxis_title="Year",
            yaxis_title="Silhouette Score",
        )
        return figure

    plot_df = plot_df.sort_values("session_num")
    
    # Merge with congress dates to get year information
    dates_dict = congress_dates_df.set_index("session_num").to_dict("index")
    
    plot_df["start_year"] = plot_df["session_num"].apply(
        lambda x: dates_dict.get(x, {}).get("start_year", int(x))
    )
    plot_df["congress_label"] = plot_df["session_num"].apply(
        lambda x: dates_dict.get(x, {}).get("label", f"Session {x}")
    )

    figure.add_trace(
        go.Scatter(
            x=plot_df["start_year"],
            y=plot_df["silhouette_score"],
            mode="lines+markers",
            name="Silhouette Score",
            customdata=plot_df[["session_num", "congress_label"]],
            hovertemplate="<b>%{customdata[1]}</b><br>Year: %{x}<br>Score: %{y:.3f}<extra></extra>",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=4),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=plot_df["start_year"],
            y=plot_df["rolling_avg"],
            mode="lines",
            name="4-Session Rolling Avg",
            line=dict(color="#ff7f0e", dash="dash"),
            hovertemplate="Year: %{x}<br>Rolling Avg: %{y:.3f}<extra></extra>",
        )
    )

    highlight_df = plot_df[plot_df["significant_shift"]]
    if not highlight_df.empty:
        figure.add_trace(
            go.Scatter(
                x=highlight_df["start_year"],
                y=highlight_df["silhouette_score"],
                mode="markers+text",
                name="Significant Shift",
                customdata=highlight_df[["session_num", "congress_label"]],
                hovertemplate="<b>Significant Shift</b><br>%{customdata[1]}<br>Year: %{x}<br>Score: %{y:.3f}<extra></extra>",
                marker=dict(color="#d62728", size=10, symbol="circle-open", line=dict(width=2)),
                text=highlight_df["congress_label"],
                textposition="top center",
            )
        )

    figure.update_layout(
        title="Senate Polarization Over Time (Silhouette Score)",
        xaxis_title="Year",
        yaxis_title="Silhouette Score",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=60, r=20, b=40, l=60),
        hovermode="closest",
    )

    return figure


def build_party_mismatch_figure(
    alignment_df: pd.DataFrame,
    congress_dates_df: pd.DataFrame,
) -> go.Figure:
    """Generate a Plotly figure showing party-cluster mismatch percentage over time.
    
    Args:
        alignment_df: DataFrame with session_num and mismatch_pct columns
        congress_dates_df: DataFrame with congress session date mappings
        
    Returns:
        Plotly Figure object with time-series visualization
    """
    figure = go.Figure()
    if alignment_df.empty:
        figure.update_layout(
            title="Party-Cluster Mismatch Percentage",
            xaxis_title="Year",
            yaxis_title="Mismatch %",
        )
        return figure

    plot_df = alignment_df.copy()
    plot_df = plot_df.sort_values("session_num")
    
    # Merge with congress dates to get year information
    dates_dict = congress_dates_df.set_index("session_num").to_dict("index")
    
    plot_df["start_year"] = plot_df["session_num"].apply(
        lambda x: dates_dict.get(x, {}).get("start_year", int(x))
    )
    plot_df["congress_label"] = plot_df["session_num"].apply(
        lambda x: dates_dict.get(x, {}).get("label", f"Session {x}")
    )

    figure.add_trace(
        go.Scatter(
            x=plot_df["start_year"],
            y=plot_df["mismatch_pct"],
            mode="lines+markers",
            name="Mismatch %",
            customdata=plot_df[["session_num", "congress_label", "total_members"]],
            hovertemplate="<b>%{customdata[1]}</b><br>Year: %{x}<br>Mismatch: %{y:.1f}%<br>Members: %{customdata[2]}<extra></extra>",
            line=dict(color="#e74c3c", width=2),
            marker=dict(size=5),
        )
    )
    
    # Add reference line at 50% (random clustering)
    figure.add_hline(
        y=50,
        line_dash="dash",
        line_color="gray",
        annotation_text="Random clustering (50%)",
        annotation_position="right"
    )

    figure.update_layout(
        title="Party-Cluster Alignment (Lower = Better Party Separation)",
        xaxis_title="Year",
        yaxis_title="Mismatch Percentage (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=60, r=20, b=40, l=60),
        hovermode="closest",
    )

    return figure
