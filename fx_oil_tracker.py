"""
FX + Oil Shock Tracker -- Monitor Diario de Impacto CPI para EMs
================================================================
Acompanha variacoes diarias de FX e petroleo desde uma data de referencia
e calcula impacto mecanico em CPI (bps) para Mexico, Africa do Sul, Chile e Colombia.
Inclui timeline de falas de bancos centrais.

Input: tracker_data.xlsx (3 abas: countries, timeseries, speeches)
Output: output/fx_oil_tracker.html (HTML estatico com Plotly)
"""

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
from datetime import datetime

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# ==============================================================================
# 0. CONFIGURACAO
# ==============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
TRACKER_XLSX = os.path.join(SCRIPT_DIR, "tracker_data.xlsx")

TONE_COLORS = {"hawkish": "#c0392b", "dovish": "#27ae60", "neutral": "#7f8c8d"}
TONE_SYMBOLS = {"hawkish": "triangle-up", "dovish": "triangle-down", "neutral": "diamond"}

LINE_STYLES = {
    "Mexico":       {"dash": "solid",   "symbol": "circle",       "width": 2.5},
    "South Africa": {"dash": "dash",    "symbol": "triangle-up",  "width": 2.5},
    "Chile":        {"dash": "dashdot", "symbol": "square",       "width": 2.5},
    "Colombia":     {"dash": "dot",     "symbol": "diamond",      "width": 2.5},
}

PLOTLY_TEMPLATE = "plotly_white"

# ==============================================================================
# 1. DATA LOADING
# ==============================================================================

def load_tracker_data():
    """Load all data from tracker_data.xlsx (3 sheets: countries, timeseries, speeches)."""
    # -- Countries sheet --
    countries_df = pd.read_excel(TRACKER_XLSX, sheet_name="countries")
    countries = {}
    meeting_refs = {}
    for _, row in countries_df.iterrows():
        name = row["country"]
        countries[name] = {
            "currency_pair": row["currency_pair"],
            "color": row["color"],
            "current_rate_pct": row["current_rate_pct"],
            "next_meeting": str(row["next_meeting"]).split(" ")[0] if pd.notna(row["next_meeting"]) else "",
            "last_meeting_date": str(row["last_meeting_date"]).split(" ")[0] if pd.notna(row["last_meeting_date"]) else "",
            "oil_passthrough_per_10pct": row["oil_passthrough_per_10pct"],
            "erpt_fx_to_cpi_per_1pct": row["erpt_fx_to_cpi_per_1pct"],
        }
        meeting_refs[name] = {
            "fx": row["meeting_ref_fx"],
            "brent": row["meeting_ref_brent"],
        }

    data = {
        "countries": countries,
        "meeting_references": meeting_refs,
    }

    # -- Timeseries sheet --
    ts = pd.read_excel(TRACKER_XLSX, sheet_name="timeseries")
    ts["date"] = pd.to_datetime(ts["date"])
    ts = ts.sort_values("date").reset_index(drop=True)

    return data, ts


def load_bc_speeches():
    """Load BC speeches from the 'speeches' sheet of tracker_data.xlsx."""
    try:
        df = pd.read_excel(TRACKER_XLSX, sheet_name="speeches")
    except (ValueError, KeyError):
        return []
    if df.empty:
        return []
    df["date"] = pd.to_datetime(df["date"])
    speeches = df.to_dict("records")
    for sp in speeches:
        if pd.isna(sp.get("tone", "")):
            sp["tone"] = "neutral"
        if pd.isna(sp.get("short_label", "")):
            sp["short_label"] = sp.get("speaker", "")[:12]
    return speeches


# ==============================================================================
# 2. CPI IMPACT CALCULATION
# ==============================================================================

def compute_daily_cpi_impact(data, ts):
    """Compute daily CPI impact in bps for each country relative to meeting references."""
    countries = data["countries"]
    meeting_refs = data["meeting_references"]
    results = {"date": ts["date"].values}

    for name, params in countries.items():
        pair = params["currency_pair"]
        oil_pt = params["oil_passthrough_per_10pct"]
        erpt = params["erpt_fx_to_cpi_per_1pct"]

        ref = meeting_refs[name]
        fx_ref = ref["fx"]
        oil_ref = ref["brent"]

        fx_vals = ts[pair].values.astype(float)
        oil_vals = ts["brent"].values.astype(float)

        fx_var_pct = (fx_vals / fx_ref - 1) * 100
        oil_var_pct = (oil_vals / oil_ref - 1) * 100

        oil_cpi_bps = (oil_var_pct / 10) * oil_pt * 100
        fx_cpi_bps = fx_var_pct * erpt * 100
        total_bps = oil_cpi_bps + fx_cpi_bps

        results[f"{name}_total"] = total_bps
        results[f"{name}_oil"] = oil_cpi_bps
        results[f"{name}_fx"] = fx_cpi_bps
        results[f"{name}_fx_var"] = fx_var_pct
        results[f"{name}_oil_var"] = oil_var_pct

    return pd.DataFrame(results)


# ==============================================================================
# 3. SUMMARY TABLE
# ==============================================================================

def build_summary_table(data, ts, impacts):
    """Build HTML summary table matching user's format."""
    countries = data["countries"]
    meeting_refs = data["meeting_references"]
    last_row = impacts.iloc[-1]
    last_ts = ts.iloc[-1]

    rows_html = ""
    for name, params in countries.items():
        pair = params["currency_pair"]
        color = params["color"]
        ref = meeting_refs[name]
        fx_ref = ref["fx"]
        oil_ref = ref["brent"]

        fx_now = last_ts[pair]
        oil_now = last_ts["brent"]
        fx_var = last_row[f"{name}_fx_var"]
        oil_var = last_row[f"{name}_oil_var"]
        total_bps = last_row[f"{name}_total"]
        oil_bps = last_row[f"{name}_oil"]
        fx_bps = last_row[f"{name}_fx"]

        fx_var_cls = "st-neg" if fx_var < 0 else "st-pos"
        oil_var_cls = "st-neg" if oil_var < 0 else "st-pos"

        # CPI bar
        max_bps = 120
        bar_width = min(abs(total_bps) / max_bps * 100, 100)
        bar_color = "#c0392b" if total_bps > 0 else "#27ae60"
        bps_cls = "st-pos" if total_bps > 0 else "st-neg"

        # Format FX values
        if fx_ref > 1000:
            fx_ref_fmt = f"{fx_ref:,.0f}"
            fx_now_fmt = f"{fx_now:,.0f}"
        elif fx_ref > 100:
            fx_ref_fmt = f"{fx_ref:,.2f}"
            fx_now_fmt = f"{fx_now:,.2f}"
        else:
            fx_ref_fmt = f"{fx_ref:.2f}"
            fx_now_fmt = f"{fx_now:.2f}"

        rows_html += f"""<tr>
          <td class="st-country"><span class="st-dot" style="background:{color}"></span>{name}</td>
          <td>{params['current_rate_pct']:.2f}%</td>
          <td>{params['next_meeting']}</td>
          <td>{fx_ref_fmt}</td>
          <td>{fx_now_fmt}</td>
          <td class="{fx_var_cls}">{fx_var:+.1f}%</td>
          <td>{oil_ref:.2f}</td>
          <td>{oil_now:.2f}</td>
          <td class="{oil_var_cls}">{oil_var:+.1f}%</td>
          <td class="{bps_cls}" style="position:relative; min-width:100px;">
            <div style="position:absolute;left:2px;top:50%;transform:translateY(-50%);
                        width:{bar_width}%;height:60%;background:{bar_color};opacity:0.15;
                        border-radius:3px;"></div>
            <span style="position:relative; z-index:1;">{total_bps:.1f}</span>
          </td>
        </tr>"""

    return f"""<table class="summary-table">
      <thead>
        <tr>
          <th rowspan="2" style="border-right:2px solid #2980b9;">Politica Monetaria:<br>Efeitos mecanicos</th>
          <th rowspan="2">Juros atuais</th>
          <th rowspan="2">Prox. Reuniao</th>
          <th colspan="3" style="border-left:2px solid #2980b9; background:#1a3a5c;">FX</th>
          <th colspan="3" style="border-left:2px solid #2980b9; background:#1a3a5c;">Oil</th>
          <th rowspan="2" style="border-left:2px solid #2980b9;">Impacto CPI<br>(bps)</th>
        </tr>
        <tr>
          <th style="border-left:2px solid #2980b9;">Ult. reuniao</th>
          <th>Agora</th>
          <th>Var %</th>
          <th style="border-left:2px solid #2980b9;">Ult. reuniao</th>
          <th>Agora</th>
          <th>Var %</th>
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>"""


# ==============================================================================
# 4. PLOTLY CHARTS
# ==============================================================================

def plot_cpi_timeline(impacts, data, speeches):
    """Main chart: CPI impact in bps over time for all countries."""
    fig = go.Figure()
    countries = data["countries"]

    for name, params in countries.items():
        style = LINE_STYLES[name]
        fig.add_trace(go.Scatter(
            x=impacts["date"],
            y=impacts[f"{name}_total"],
            name=name,
            mode="lines+markers",
            line=dict(color=params["color"], width=style["width"], dash=style["dash"]),
            marker=dict(symbol=style["symbol"], size=7),
            hovertemplate=f"<b>{name}</b><br>%{{x|%d/%b}}<br>CPI: %{{y:.1f}} bps<extra></extra>",
        ))

    add_speech_annotations(fig, speeches, impacts, data)

    fig.update_layout(
        title=dict(text="Impacto Mecanico no CPI (bps) desde o Choque", font=dict(size=16)),
        xaxis=dict(title="Data", tickformat="%d/%b", dtick="D1"),
        yaxis=dict(title="Impacto CPI (bps)", zeroline=True, zerolinewidth=1.5, zerolinecolor="#ccc"),
        template=PLOTLY_TEMPLATE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450,
        margin=dict(l=60, r=30, t=80, b=60),
        hovermode="x unified",
    )
    return fig


def plot_fx_oil_panel(ts, data, speeches):
    """Two subplots: FX indexed to 100 (left) and Brent price (right)."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["FX (indexado a 100 na data ref.)", "Brent (USD/bbl)"],
        horizontal_spacing=0.08,
    )
    countries = data["countries"]

    for name, params in countries.items():
        pair = params["currency_pair"]
        style = LINE_STYLES[name]
        fx_vals = ts[pair].astype(float)
        fx_indexed = (fx_vals / fx_vals.iloc[0]) * 100

        fig.add_trace(go.Scatter(
            x=ts["date"], y=fx_indexed,
            name=name, legendgroup=name,
            mode="lines+markers",
            line=dict(color=params["color"], width=style["width"], dash=style["dash"]),
            marker=dict(symbol=style["symbol"], size=6),
            hovertemplate=f"<b>{name}</b><br>%{{x|%d/%b}}<br>FX idx: %{{y:.1f}}<extra></extra>",
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=ts["date"], y=ts["brent"],
        name="Brent", legendgroup="Brent",
        mode="lines+markers",
        line=dict(color="#1a5276", width=2.5),
        marker=dict(symbol="circle", size=6, color="#1a5276"),
        hovertemplate="<b>Brent</b><br>%{x|%d/%b}<br>$%{y:.2f}/bbl<extra></extra>",
    ), row=1, col=2)

    # Add speech annotations to both panels
    for sp in speeches:
        for col in [1, 2]:
            color = TONE_COLORS.get(sp["tone"], "#7f8c8d")
            fig.add_vline(
                x=sp["date"].timestamp() * 1000 if hasattr(sp["date"], "timestamp") else sp["date"],
                line_dash="dot", line_color=color, line_width=1, opacity=0.4,
                row=1, col=col,
            )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=420,
        margin=dict(l=60, r=30, t=80, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_xaxes(tickformat="%d/%b", dtick="D1")
    fig.update_yaxes(title_text="Indice (100=ref)", row=1, col=1)
    fig.update_yaxes(title_text="USD/bbl", row=1, col=2)

    return fig


def plot_decomposition(impacts, data):
    """Horizontal stacked bar: Oil vs FX contribution to CPI impact."""
    countries = data["countries"]
    last = impacts.iloc[-1]

    names = list(countries.keys())
    oil_vals = [last[f"{n}_oil"] for n in names]
    fx_vals = [last[f"{n}_fx"] for n in names]
    colors = [countries[n]["color"] for n in names]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=names, x=oil_vals, name="Oil pass-through",
        orientation="h",
        marker_color="#1a5276",
        text=[f"{v:.1f}" for v in oil_vals],
        textposition="inside",
        hovertemplate="<b>%{y}</b><br>Oil: %{x:.1f} bps<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        y=names, x=fx_vals, name="FX pass-through (ERPT)",
        orientation="h",
        marker_color="#e67e22",
        text=[f"{v:.1f}" for v in fx_vals],
        textposition="inside",
        hovertemplate="<b>%{y}</b><br>FX: %{x:.1f} bps<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="Decomposicao do Impacto CPI: Oil vs FX (bps)", font=dict(size=14)),
        barmode="stack",
        template=PLOTLY_TEMPLATE,
        height=280,
        margin=dict(l=100, r=30, t=50, b=40),
        xaxis=dict(title="bps"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_meeting_countdown(data):
    """Visual countdown to next CB meeting for each country."""
    countries = data["countries"]
    today = pd.Timestamp.now().normalize()

    names = []
    days_left = []
    colors = []

    for name, params in countries.items():
        meeting = pd.to_datetime(params["next_meeting"])
        delta = (meeting - today).days
        names.append(name)
        days_left.append(max(delta, 0))
        colors.append(params["color"])

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names, x=days_left,
        orientation="h",
        marker_color=colors,
        text=[f"{d}d" for d in days_left],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>%{x} dias<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="Dias ate Proxima Reuniao", font=dict(size=14)),
        template=PLOTLY_TEMPLATE,
        height=220,
        margin=dict(l=100, r=60, t=50, b=30),
        xaxis=dict(title="Dias"),
    )
    return fig


# ==============================================================================
# 5. SPEECH ANNOTATIONS
# ==============================================================================

def add_speech_annotations(fig, speeches, impacts, data):
    """Add BC speech markers to the CPI timeline chart."""
    if not speeches:
        return

    # Get y-range for positioning
    y_max = 0
    for name in data["countries"]:
        y_max = max(y_max, impacts[f"{name}_total"].max())
    y_max = max(y_max, 10) * 1.1

    for sp in speeches:
        color = TONE_COLORS.get(sp["tone"], "#7f8c8d")
        country_color = data["countries"].get(sp.get("country", ""), {}).get("color", color)

        fig.add_vline(
            x=sp["date"], line_dash="dot", line_color=color,
            line_width=1, opacity=0.5,
        )
        fig.add_annotation(
            x=sp["date"], y=y_max * 0.95,
            text=sp["short_label"],
            showarrow=True, arrowhead=2, arrowcolor=color, arrowwidth=1,
            font=dict(size=9, color=color),
            bgcolor="white", bordercolor=color, borderwidth=1, borderpad=3,
            opacity=0.9,
        )


# ==============================================================================
# 6. BC SPEECHES TIMELINE TABLE
# ==============================================================================

def build_speeches_table(speeches, data):
    """Build HTML table listing all BC speeches."""
    if not speeches:
        return "<p style='color:#888; font-style:italic;'>Nenhuma fala de BC registrada.</p>"

    rows = ""
    for sp in sorted(speeches, key=lambda x: x["date"]):
        tone = sp.get("tone", "neutral")
        tone_color = TONE_COLORS.get(tone, "#7f8c8d")
        tone_label = {"hawkish": "Hawkish", "dovish": "Dovish", "neutral": "Neutro"}.get(tone, tone)
        country_color = data["countries"].get(sp.get("country", ""), {}).get("color", "#333")

        link = sp.get("link", "")
        link_html = ""
        if link and not (isinstance(link, float) and pd.isna(link)):
            link_html = f'<a href="{link}" target="_blank" style="color:#2980b9; text-decoration:none;">link</a>'

        rows += f"""<tr>
          <td style="white-space:nowrap;">{sp['date'].strftime('%d/%b')}</td>
          <td><span style="color:{country_color}; font-weight:600;">{sp.get('country','')}</span></td>
          <td>{sp.get('speaker','')}</td>
          <td style="text-align:left;">{sp.get('title','')}</td>
          <td><span style="color:{tone_color}; font-weight:600;">{tone_label}</span></td>
          <td>{link_html}</td>
        </tr>"""

    return f"""<table class="summary-table" style="font-size:12px;">
      <thead>
        <tr>
          <th>Data</th><th>Pais</th><th>Speaker</th><th style="text-align:left;">Resumo</th><th>Tom</th><th>Link</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>"""


# ==============================================================================
# 7. HTML ASSEMBLY
# ==============================================================================

def build_tracker_html(summary_html, charts, speeches_table):
    """Assemble the final HTML page."""
    cpi_chart_html = pio.to_html(charts["cpi_timeline"], full_html=False, include_plotlyjs=False)
    fx_oil_html = pio.to_html(charts["fx_oil_panel"], full_html=False, include_plotlyjs=False)
    decomp_html = pio.to_html(charts["decomposition"], full_html=False, include_plotlyjs=False)
    countdown_html = pio.to_html(charts["countdown"], full_html=False, include_plotlyjs=False)

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f'''<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FX + Oil Tracker: EM Central Banks</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    max-width: 1250px; margin: 0 auto; padding: 20px;
    background: #fafafa; color: #333;
  }}
  h1 {{
    color: #2c3e50; border-bottom: 3px solid #2980b9;
    padding-bottom: 10px; font-size: 22px;
  }}
  h2 {{
    color: #2c3e50; font-size: 16px; margin-top: 30px; margin-bottom: 10px;
    border-left: 4px solid #2980b9; padding-left: 10px;
  }}
  .timestamp {{
    color: #999; font-size: 12px;
  }}
  .summary-box {{
    background: #fff; border-left: 4px solid #2980b9;
    padding: 12px 20px; margin: 15px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-radius: 0 5px 5px 0;
    overflow-x: auto;
  }}
  .chart-container {{
    margin: 15px 0; background: #fff;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    border-radius: 5px; padding: 10px;
  }}
  table {{ border-collapse: collapse; width: 100%; margin: 8px 0; background: #fff; font-size: 12px; }}
  th, td {{ border: 1px solid #ddd; padding: 7px 10px; text-align: center; }}
  th {{ background: #2c3e50; color: #fff; font-weight: 600; font-size: 11px; }}
  tr:nth-child(even) {{ background: #f8f9fa; }}
  .summary-table {{
    border-collapse: separate; border-spacing: 0;
    border-radius: 8px; overflow: hidden;
    box-shadow: 0 1px 6px rgba(0,0,0,0.08);
    font-size: 13px; margin: 0;
  }}
  .summary-table thead th {{
    background: #1a252f; color: #ecf0f1;
    padding: 10px 14px; font-weight: 600; font-size: 11.5px;
    text-transform: uppercase; letter-spacing: 0.3px;
    border: none; border-bottom: 2px solid #2980b9;
  }}
  .summary-table tbody td {{
    padding: 10px 14px; border: none;
    border-bottom: 1px solid #eee; font-variant-numeric: tabular-nums;
    font-family: 'Consolas', 'SF Mono', monospace; font-size: 13px;
  }}
  .summary-table tbody tr:last-child td {{ border-bottom: none; }}
  .summary-table tbody tr:nth-child(even) {{ background: #f7f9fb; }}
  .summary-table tbody tr:hover {{ background: #eaf2f8; transition: background 0.15s; }}
  .summary-table .st-country {{
    text-align: left; font-family: 'Segoe UI', sans-serif; white-space: nowrap;
  }}
  .summary-table .st-dot {{
    display: inline-block; width: 8px; height: 8px;
    border-radius: 50%; margin-right: 6px; vertical-align: middle;
  }}
  .summary-table .st-pos {{ color: #c0392b; font-weight: 600; }}
  .summary-table .st-neg {{ color: #27ae60; font-weight: 600; }}
  .source-note {{ font-size: 11px; color: #888; margin-top: 2px; font-style: italic; }}
  .nav-bar {{
    background: #2c3e50; padding: 8px 16px; border-radius: 5px; margin-bottom: 15px;
    display: flex; gap: 16px; align-items: center;
  }}
  .nav-bar a {{
    color: #ecf0f1; text-decoration: none; font-size: 13px; font-weight: 500;
    padding: 4px 10px; border-radius: 4px; transition: background 0.2s;
  }}
  .nav-bar a:hover {{ background: #34495e; }}
  .nav-bar a.active {{ background: #2980b9; }}
</style>
</head>
<body>

<div class="nav-bar">
  <a href="tracker.html" class="active">FX + Oil Tracker</a>
  <a href="index.html">Oil Shock Simulator</a>
</div>

<h1>FX + Oil Tracker: EM Central Banks</h1>
<p class="timestamp">Atualizado em: {now_str}</p>

<h2>Resumo: Efeitos Mecanicos</h2>
<div class="summary-box">
  {summary_html}
</div>

<h2>Impacto CPI ao Longo do Tempo</h2>
<div class="chart-container">
  {cpi_chart_html}
</div>
<p class="source-note">Impacto mecanico calculado com pass-through de oil e FX (ERPT) calibrados por pais. Marcadores indicam falas de BCs.</p>

<h2>FX e Petroleo</h2>
<div class="chart-container">
  {fx_oil_html}
</div>

<h2>Decomposicao: Oil vs FX</h2>
<div class="chart-container">
  {decomp_html}
</div>

<h2>Proximas Reunioes</h2>
<div class="chart-container">
  {countdown_html}
</div>

<h2>Falas de Bancos Centrais</h2>
<div class="summary-box">
  {speeches_table}
</div>


</body>
</html>'''

    return html


# ==============================================================================
# 8. MAIN
# ==============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 55)
    print("FX + OIL TRACKER - EM CENTRAL BANKS")
    print("=" * 55)

    print("Carregando dados...")
    data, ts = load_tracker_data()
    speeches = load_bc_speeches()

    ref_date = ts["date"].iloc[0].strftime("%Y-%m-%d")
    n_days = len(ts)
    n_speeches = len(speeches)

    print(f"  Referencia: {ref_date}")
    print(f"  Dias de dados: {n_days}")
    print(f"  Falas de BCs: {n_speeches}")
    print(f"  Paises: {', '.join(data['countries'].keys())}")
    print()

    print("Calculando impacto CPI...")
    impacts = compute_daily_cpi_impact(data, ts)

    # Print latest values
    last = impacts.iloc[-1]
    for name in data["countries"]:
        total = last[f"{name}_total"]
        oil = last[f"{name}_oil"]
        fx = last[f"{name}_fx"]
        print(f"  {name:15s}: {total:+6.1f} bps (oil: {oil:+.1f}, fx: {fx:+.1f})")
    print()

    print("Gerando graficos...")
    charts = {
        "cpi_timeline": plot_cpi_timeline(impacts, data, speeches),
        "fx_oil_panel": plot_fx_oil_panel(ts, data, speeches),
        "decomposition": plot_decomposition(impacts, data),
        "countdown": plot_meeting_countdown(data),
    }

    summary_html = build_summary_table(data, ts, impacts)
    speeches_table = build_speeches_table(speeches, data)

    print("Montando HTML...")
    html = build_tracker_html(summary_html, charts, speeches_table)

    out_path = os.path.join(OUTPUT_DIR, "fx_oil_tracker.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    # Copiar para raiz como tracker.html (GitHub Pages)
    gh_pages_path = os.path.join(SCRIPT_DIR, "tracker.html")
    with open(gh_pages_path, "w", encoding="utf-8") as f:
        f.write(html)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"\nDashboard salvo em: {out_path} ({size_kb:.0f} KB)")
    print(f"GitHub Pages: {gh_pages_path}")
    print("Abra o arquivo no navegador para visualizar.")


if __name__ == "__main__":
    main()
