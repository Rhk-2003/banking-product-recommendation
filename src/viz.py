"""
Chart helpers. Colour roles come from a validated palette:
 - single-series marks use categorical slot 1 (blue #257de0)
 - the four customer segments are an *ordered* value tier, so they use an
   ordinal single-hue blue ramp (steps 250/350/500/650), not categorical hues.
   An ordered quantity encoded with unrelated hues is a misread waiting to
   happen, and a 4-hue categorical set does not clear all-pairs CVD separation
   for scatter plots anyway.
"""
import plotly.graph_objects as go

SURFACE = "#ffffff"
TEXT_PRIMARY = "#191919"
TEXT_SECONDARY = "#666666"
GRID = "#e8e8e8"

SERIES_1 = "#257de0"
SERIES_2 = "#f47230"
SERIES_3 = "#00b894"

# ordinal ramp, lowest value tier -> highest
SEGMENT_RAMP = ["#84b9f3", "#519ced", "#2370cd", "#0f4a96"]

STATUS = {"good": "#0aac23", "warning": "#fbb315", "serious": "#f18a59", "critical": "#dc3c3d"}


def segment_colors(labels_by_rank: list[str]) -> dict:
    """Map segment label -> ramp step, darkest for the highest value tier."""
    n = len(labels_by_rank)
    ramp = SEGMENT_RAMP if n == 4 else [SEGMENT_RAMP[i * 3 // max(n - 1, 1)] for i in range(n)]
    # labels_by_rank is ordered highest-value first -> reverse so darkest = highest
    return {lab: ramp[n - 1 - i] for i, lab in enumerate(labels_by_rank)}


def style(fig: go.Figure, height: int = 360, showlegend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=78 if showlegend else 44, b=8),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif",
                  size=13, color=TEXT_PRIMARY),
        # title pinned to the top of the margin so the horizontal legend below it
        # has room; otherwise the two overlap on legended charts
        title=dict(font=dict(size=15, color=TEXT_PRIMARY), x=0, xanchor="left",
                   y=0.99, yanchor="top"),
        showlegend=showlegend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    font=dict(size=12, color=TEXT_SECONDARY), title=None),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=GRID,
                        font=dict(size=12, color=TEXT_PRIMARY)),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=GRID,
                     tickfont=dict(size=12, color=TEXT_SECONDARY),
                     title_font=dict(size=12, color=TEXT_SECONDARY))
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, linecolor=GRID,
                     tickfont=dict(size=12, color=TEXT_SECONDARY),
                     title_font=dict(size=12, color=TEXT_SECONDARY))
    return fig


def hbar(y, x, title: str, xtitle: str, color=SERIES_1, texts=None, height=None):
    """Horizontal bar with rounded data-ends and direct value labels."""
    fig = go.Figure(go.Bar(
        y=y, x=x, orientation="h", marker=dict(color=color, cornerradius=4),
        text=texts if texts is not None else x, textposition="outside",
        textfont=dict(size=12, color=TEXT_SECONDARY),
        hovertemplate="%{y}<br>" + xtitle + ": %{x}<extra></extra>",
    ))
    fig.update_layout(title=title, bargap=0.35)
    # outside labels are drawn past the bar end, so pad the range or they clip
    xmax = max(float(v) for v in x) if len(list(x)) else 1.0
    fig.update_xaxes(title=xtitle, showgrid=True, gridcolor=GRID,
                     range=[0, xmax * 1.18])
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return style(fig, height=height or max(240, 34 * len(list(y)) + 90), showlegend=False)
