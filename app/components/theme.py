"""Single theme module: palette, colorblind-safe tier colors, Plotly template, CSS.

One accent color and one tier palette are reused everywhere so risk tiers always read
the same. The Plotly template avoids the default rainbow and sets a clean editorial
look. Plotly is imported lazily so the module loads without it.
"""

from __future__ import annotations

ACCENT = "#2166AC"
INK = "#1A1A1A"
MUTED = "#6B7280"
GRID = "#E6E8EC"

#: Colorblind-safe risk-tier palette (red-blue diverging + amber), used identically
#: across every page and chart.
TIER_COLORS = {
    "High": "#B2182B",
    "Elevated": "#EF8A62",
    "Watch": "#F7C948",
    "Low": "#2166AC",
}
TIER_ORDER = ["High", "Elevated", "Watch", "Low"]

#: Two-tone sequence for non-tier categorical charts (cluster contrasts).
CLUSTER_COLORS = {"Fast Food": "#B2182B", "Non-Fast Food": "#2166AC"}

CARD_CSS = """
<style>
.lrr-card { border:1px solid #E6E8EC; border-radius:10px; padding:14px 16px;
            background:#FFFFFF; margin-bottom:10px; }
.lrr-badge { color:#FFFFFF; padding:2px 10px; border-radius:12px; font-size:0.8rem;
             font-weight:600; display:inline-block; }
.lrr-quote { background:#FFF6D6; padding:1px 3px; border-radius:3px; }
.lrr-muted { color:#6B7280; font-size:0.85rem; }
.lrr-strike { text-decoration:line-through; color:#B2182B; }
</style>
"""


def plotly_template():
    """Return a clean Plotly template, or None if plotly is unavailable."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    return go.layout.Template(
        layout=go.Layout(
            font=dict(family="sans-serif", color=INK, size=13),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            colorway=[ACCENT, "#B2182B", "#EF8A62", "#F7C948", "#67A9CF", MUTED],
            xaxis=dict(gridcolor=GRID, zeroline=False),
            yaxis=dict(gridcolor=GRID, zeroline=False),
            margin=dict(l=50, r=20, t=50, b=45),
        )
    )


def tier_color(tier: str) -> str:
    return TIER_COLORS.get(str(tier), MUTED)
