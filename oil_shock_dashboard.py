"""
Choque no Preço do Petróleo — Dashboard de Impacto para Mercados Emergentes
=============================================================================
Modelo estrutural calibrado: choque petróleo → inflação + PIB + balança comercial + IRFs
Países: Chile, México, Colômbia, Brasil, África do Sul
Output: Dashboard HTML interativo (Plotly) com 7 abas + simulador de cenários
Parâmetros calibrados a partir de literatura acadêmica recente (2018-2025).
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import gamma as gamma_dist
from datetime import datetime

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# ══════════════════════════════════════════════════════════════════════════════
# 0. CONFIGURAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
BASELINE_OIL_PRICE = 70.0  # USD/barril
HORIZON_QUARTERS = 12

SCENARIOS = [
    {"name": "Queda Moderada",   "target": 55,  "desc": "Desaceleração global / excesso de oferta"},
    {"name": "Moderado",         "target": 85,  "desc": "Tensão geopolítica localizada"},
    {"name": "$70 a $100",        "target": 100, "desc": "Choque significativo de oferta", "default": True},
    {"name": "Guerra da Ucrania","target": 125, "desc": "Pico ~$128/bbl (Mar/2022, invasao Russia-Ucrania)", "highlight": True},
    {"name": "Extremo",          "target": 150, "desc": "Cenário de cauda / disrupção severa"},
]

SLIDER_PRICES = list(range(40, 165, 5))  # $40 a $160, step $5
DEFAULT_PRICE = next(s["target"] for s in SCENARIOS if s.get("default"))
DEFAULT_SLIDER_IDX = SLIDER_PRICES.index(DEFAULT_PRICE)

COUNTRIES = {
    "Chile": {
        "color": "#e74c3c",
        "oil_position": "net_importer",
        "gdp_nominal_bn": 335,  # USD bn, IMF WEO 2024 ($330-344bn range)
        # BCCh WP 417: pass-through em declínio; MEPCO amortece curto prazo
        # Fuel weight CPI ~3.5-4.5%; ~95% dependente de importação
        "headline_passthrough_per_10pct": {"low": 0.25, "mid": 0.35, "high": 0.45},
        "core_passthrough_ratio": 0.50,
        "gdp_impact_per_10pct": {"low": -0.25, "mid": -0.18, "high": -0.10},
        "irf_shape": {
            "headline_peak_q": 2, "core_peak_q": 5, "gdp_peak_q": 3,
            "decay_half_life_q": 2.5, "full_dissipation_q": 10,
        },
        "channels": {"demanda": 0.35, "oferta": 0.45, "termos_de_troca": 0.20},
        "trade": {
            # Crude imports ~$5bn + refined petroleum ~$7bn = ~$12bn (OEC/WITS 2024)
            # ENAP re-exporta ~$0.8bn/ano em refinados
            # Produção doméstica ~15k bbl/d (vs consumo ~386k bbl/d) → 95-97% importado
            "oil_import_bn": 12.0, "oil_export_bn": 0.8,
            "total_imports_bn": 85.0, "total_exports_bn": 100.0,
            "oil_import_volume_elasticity_sr": 0.10,
            "oil_import_volume_elasticity_lr": 0.35,
            "fx_oil_beta": -0.05,
        },
        "fonte_inflacao": "BCCh WP 747; BCCh WP 417; MEPCO/CNE",
        "fonte_pib": "FLAR (2024); consensus net importer",
        "fonte_trade": "OEC World (2024); WITS; ENAP; IEA Chile",
    },
    "México": {
        "color": "#2980b9",
        "oil_position": "mixed",
        "gdp_nominal_bn": 1400,  # USD bn, IMF WEO 2024
        "headline_passthrough_per_10pct": {"low": 0.20, "mid": 0.30, "high": 0.40},
        "core_passthrough_ratio": 0.45,
        "gdp_impact_per_10pct": {"low": -0.15, "mid": -0.10, "high": -0.05},
        "irf_shape": {
            "headline_peak_q": 2, "core_peak_q": 4, "gdp_peak_q": 3,
            "decay_half_life_q": 3, "full_dissipation_q": 10,
        },
        "channels": {"demanda": 0.30, "oferta": 0.40, "termos_de_troca": 0.30},
        "trade": {
            "oil_import_bn": 74.0, "oil_export_bn": 39.0,
            "total_imports_bn": 605.0, "total_exports_bn": 578.0,
            "oil_import_volume_elasticity_sr": 0.10,
            "oil_import_volume_elasticity_lr": 0.30,
            "fx_oil_beta": -0.02,
        },
        "fonte_inflacao": "Energia 9.5% CPI; Banxico",
        "fonte_pib": "IMF WP/17/15; misto exportador/consumidor",
        "fonte_trade": "OEC World; PEMEX; EIA",
    },
    "Colômbia": {
        "color": "#f39c12",
        "oil_position": "net_exporter",
        "gdp_nominal_bn": 420,  # USD bn, IMF WEO 2024 ($418.5bn)
        # Net exporter: oil↑ → peso aprecia → efeito deflacionário parcial
        # FEPC (Fundo de Estabilização) bufferiza preços domésticos de combustível
        # SVAR (Scielo/Redalyc): oil price increases são levemente deflacionários na Colômbia
        "headline_passthrough_per_10pct": {"low": 0.10, "mid": 0.20, "high": 0.35},
        "core_passthrough_ratio": 0.45,
        # SVAR: +0.4% contemporâneo por 10% oil shock; FLAR (2024) confirma efeito positivo
        # Oil rents ~3-4% PIB; Ecopetrol ~10% receita fiscal
        "gdp_impact_per_10pct": {"low": 0.15, "mid": 0.25, "high": 0.40},
        "irf_shape": {
            "headline_peak_q": 3, "core_peak_q": 5, "gdp_peak_q": 3,
            "decay_half_life_q": 3, "full_dissipation_q": 10,
        },
        "channels": {"demanda": 0.25, "oferta": 0.30, "termos_de_troca": 0.45},
        "trade": {
            # 2024 DANE: crude+refined exports ~$14.8bn (29.8% of exports)
            # Imports: refined $4.6bn + gas $1.1bn + crude $0.5bn ≈ $6.2bn
            # Net oil exporter ~$8.6bn
            "oil_import_bn": 6.0, "oil_export_bn": 14.8,
            "total_imports_bn": 64.0, "total_exports_bn": 50.0,
            "oil_import_volume_elasticity_sr": 0.10,
            "oil_import_volume_elasticity_lr": 0.30,
            # Petrocurrency: correlação COP-Brent ~ -0.84, R² ~50%
            "fx_oil_beta": -0.10,
        },
        "fonte_inflacao": "Banco de la República; SVAR Scielo; FEPC buffer",
        "fonte_pib": "FLAR (2024); SVAR Redalyc; Ecopetrol fiscal channel",
        "fonte_trade": "DANE (2024); OEC World; ANH Colômbia",
    },
    "Brasil": {
        "color": "#FFD700",
        "oil_position": "mixed",
        "gdp_nominal_bn": 2100,  # USD bn, IMF WEO 2024
        # BCB WP 556 (Correa & Minella, 2012): ERPT 0.04-0.08 contemporâneo
        # BCB Relatório de Inflação (2023): petróleo → IPCA via gasolina (~5% peso) + diesel indireto (~3%)
        # Petrobras abandonou PPI em 2023 → pricing opaco, dampening short-run pass-through
        # IDB (2023): Brasil pass-through abaixo da média LatAm por política de preços administrados
        "headline_passthrough_per_10pct": {"low": 0.15, "mid": 0.25, "high": 0.35},
        # BCB expectativas ancoradas (meta 3.0% ± 1.5pp), indexação salarial moderada
        "core_passthrough_ratio": 0.35,
        # IPEA Carta de Conjuntura: net exporter → efeito PIB próximo de zero ou levemente positivo
        # BCB SVAR: efeito ambíguo — custo energético vs receita exportadora + fiscal
        # Oil rents ~3-4% PIB; Petrobras dividendos + royalties significativos para fiscal
        "gdp_impact_per_10pct": {"low": -0.05, "mid": 0.05, "high": 0.15},
        "irf_shape": {
            "headline_peak_q": 3, "core_peak_q": 5, "gdp_peak_q": 4,
            "decay_half_life_q": 2.5, "full_dissipation_q": 10,
        },
        # ToT dominante (exportador líquido de crude); demanda e oferta simétricos
        "channels": {"demanda": 0.30, "oferta": 0.30, "termos_de_troca": 0.40},
        "trade": {
            # ANP (2024): crude exports ~$28-32bn; refined product imports ~$15-18bn
            # Net oil exporter desde ~2019-2020 (pré-sal ramp-up)
            "oil_import_bn": 18.0, "oil_export_bn": 32.0,
            "total_imports_bn": 240.0, "total_exports_bn": 340.0,
            "oil_import_volume_elasticity_sr": 0.08,
            "oil_import_volume_elasticity_lr": 0.30,
            # BRL moderadamente petrocurrency: pré-sal + commodity ToT
            # Mais fraco que COP (petróleo ~10% exports vs ~30% Colômbia)
            "fx_oil_beta": -0.04,
        },
        "fonte_inflacao": "BCB Relatório de Inflação; BCB WP 556; IPEA; IDB (2023); Petrobras pricing",
        "fonte_pib": "IPEA Carta de Conjuntura; BCB SVAR; ANP; Min Fazenda/SPE",
        "fonte_trade": "MDIC/Comex Stat (2024); ANP; OEC World; WITS",
    },
    "África do Sul": {
        "color": "#27ae60",
        "oil_position": "net_importer",
        "gdp_nominal_bn": 400,  # USD bn, IMF WEO 2024
        # Sobocinski (2017, SA J. Econ.): long-run elasticity 0.12pp/10%
        # Tandfonline (2025): pass-through ~3x menor dentro da meta 3-6%
        # Fuel weight CPI ~4.5% (Stats SA 2025); coal insulates parcialmente
        "headline_passthrough_per_10pct": {"low": 0.06, "mid": 0.14, "high": 0.25},
        "core_passthrough_ratio": 0.30,
        # Essama-Nssah et al. (2007, World Bank CGE): -0.16pp/10%
        # Memela (2016, DSGE SA): efeito "short-lived" e "statistically insignificant"
        # Coal-based energy (70% primary) + Sasol synthetic fuels → parcialmente isolado
        "gdp_impact_per_10pct": {"low": -0.15, "mid": -0.10, "high": -0.05},
        "irf_shape": {
            "headline_peak_q": 2, "core_peak_q": 5, "gdp_peak_q": 5,
            "decay_half_life_q": 3, "full_dissipation_q": 10,
        },
        "channels": {"demanda": 0.40, "oferta": 0.40, "termos_de_troca": 0.20},
        "trade": {
            # Crude imports 138k bbl/d × $70/bbl × 365 ≈ $3.5bn (WITS: $4.8bn @$82)
            # Sasol 150k bbl/d sintético (28% consumo) isola parcialmente do crude
            # Exposição líquida ao crude: ~$6bn; exports petroquímicos Sasol ~$0.5bn
            "oil_import_bn": 6.0, "oil_export_bn": 0.5,
            "total_imports_bn": 107.0, "total_exports_bn": 110.0,
            "oil_import_volume_elasticity_sr": 0.08,
            "oil_import_volume_elasticity_lr": 0.30,
            "fx_oil_beta": -0.04,
        },
        "fonte_inflacao": "Sobocinski (2017); Tandfonline (2025)",
        "fonte_pib": "Memela (2016 DSGE); Essama-Nssah (2007 WB CGE); SARB",
        "fonte_trade": "WITS/UN Comtrade (2023); EIA; Sasol",
    },
}

LINE_STYLES = {
    "Chile":         {"dash": "solid", "symbol": "circle",      "width": 2.5},
    "México":        {"dash": "solid", "symbol": "square",      "width": 2.5},
    "Colômbia":      {"dash": "dash",  "symbol": "diamond",     "width": 2.5},
    "Brasil":        {"dash": "dashdot", "symbol": "star",        "width": 2.5},
    "África do Sul": {"dash": "dot",   "symbol": "triangle-up", "width": 3.0},
}

# ── Dados históricos — Choques geopolíticos recentes (2022–2024) ──────────────
# Fontes: oil.xlsx (Brent) e fx.xlsx (USDMXN, USDCLP, USDCOP, USDBRL, USDZAR)

_DATA_DIR = os.path.dirname(os.path.abspath(__file__))

GEOPOLITICAL_EVENTS = [
    {"name": "Invasão da Ucrânia",       "eve": "2022-02-23", "shock": "2022-02-24",
     "desc": "Rússia invade a Ucrânia; Brent dispara nas semanas seguintes"},
    {"name": "Ataque Hamas–Israel",       "eve": "2023-10-06", "shock": "2023-10-09",
     "desc": "Ataque do Hamas em 7/Out (sábado); mercado reage na segunda-feira"},
    {"name": "Houthis — Mar Vermelho",    "eve": "2023-12-15", "shock": "2023-12-18",
     "desc": "Houthis atacam navios no Mar Vermelho; risco de disrupção logística"},
    {"name": "Escalada Israel–Líbano",    "eve": "2024-09-27", "shock": "2024-09-30",
     "desc": "Israel intensifica operações contra Hezbollah no Líbano"},
]

def _load_geopolitical_data():
    """Carrega Brent + FX para todos os eventos geopolíticos."""
    from datetime import timedelta

    oil = pd.read_excel(os.path.join(_DATA_DIR, "oil.xlsx"))
    oil["date"] = pd.to_datetime(oil["date"], origin="1899-12-30", unit="D")
    oil = oil.sort_values("date").set_index("date")

    fx = pd.read_excel(os.path.join(_DATA_DIR, "fx.xlsx"))
    fx["Date"] = pd.to_datetime(fx["Date"]).dt.normalize()
    _fx_cols = ["Date", "usdclp", "usdcop", "usdmxn", "usdbrl", "usdzar"]
    fx = fx[[c for c in _fx_cols if c in fx.columns]]
    fx = fx.sort_values("Date").set_index("Date")

    results = []
    for ev in GEOPOLITICAL_EVENTS:
        eve_dt = pd.Timestamp(ev["eve"])
        start = eve_dt - timedelta(days=2)
        end = eve_dt + timedelta(days=20)

        oil_slice = oil.loc[start:end, "brent"].dropna()
        fx_slice = fx.loc[start:end].dropna()

        # Merge no inner join de datas
        merged = pd.DataFrame({"brent": oil_slice}).join(fx_slice, how="inner")
        merged = merged.sort_index()

        if merged.empty:
            continue

        dates = merged.index
        def _closest(target):
            diffs = abs(dates - target)
            return diffs.argmin()

        eve_idx = _closest(eve_dt)
        shock_idx = _closest(pd.Timestamp(ev["shock"]))
        d7_idx = _closest(eve_dt + timedelta(days=7))
        d14_idx = _closest(eve_dt + timedelta(days=14))

        snapshots = {
            "eve": int(eve_idx), "d0": int(shock_idx),
            "d7": int(d7_idx), "d14": int(d14_idx),
        }

        results.append({
            "name": ev["name"],
            "desc": ev["desc"],
            "dates": merged.index.strftime("%Y-%m-%d").tolist(),
            "brent": merged["brent"].tolist(),
            "USDMXN": merged["usdmxn"].tolist(),
            "USDCLP": merged["usdclp"].tolist(),
            "USDCOP": merged["usdcop"].tolist(),
            "USDZAR": merged["usdzar"].tolist(),
            "USDBRL": merged["usdbrl"].tolist(),
            "snap": snapshots,
        })

    return results

GEOPOLITICAL_DATA = _load_geopolitical_data()

# ══════════════════════════════════════════════════════════════════════════════
# EUA — PARÂMETROS CALIBRADOS (aba separada, não mistura com EMs)
# ══════════════════════════════════════════════════════════════════════════════

US_PARAMS = {
    "color": "#1a5276",
    "oil_position": "mixed",
    "gdp_nominal_bn": 28000,

    # HEADLINE PCE — Presno & Prestipino (FEDS Notes 2024): +0.15pp/10%
    # Hobijn (NY Fed 2008): +0.29pp;  FEDS Notes 2023: +0.40pp total (incl. 2nd round)
    "headline_passthrough_per_10pct": {"low": 0.15, "mid": 0.25, "high": 0.40},

    # CORE PCE — Presno & Prestipino (2024): +0.06pp; FEDS Notes 2017: limitado
    # Kilian & Zhou (Dallas Fed 2023): "indistinguível de zero" para gasolina
    "core_passthrough_ratio": 0.25,

    # GDP — Oladosu et al. (2018 meta): -0.020; Presno & Prestipino: -0.08% Q10
    # Gagliardone & Gertler (NBER 2023): -0.20 a -0.30%
    "gdp_impact_per_10pct": {"low": -0.20, "mid": -0.10, "high": -0.06},

    # DESEMPREGO — IMF WP 2025/145: +0.13pp Q7, +0.23pp Q8
    # Kanzig (AER 2021): employment-to-pop -0.25pp Q7
    "unemployment_impact_per_10pct": {"low": 0.10, "mid": 0.15, "high": 0.25},

    "irf_shape": {
        "headline_peak_q": 2,
        "core_peak_q": 6,
        "gdp_peak_q": 5,
        "unemp_peak_q": 7,
        "decay_half_life_q": 3,
        "full_dissipation_q": 10,
    },

    "channels": {"demanda": 0.40, "oferta": 0.50, "termos_de_troca": 0.10},

    "trade": {
        "oil_import_bn": 200.0,
        "oil_export_bn": 120.0,
        "total_imports_bn": 3200.0,
        "total_exports_bn": 3500.0,
        "oil_import_volume_elasticity_sr": 0.10,
        "oil_import_volume_elasticity_lr": 0.30,
        "fx_oil_beta": 0.0,
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# 1. FUNÇÕES AUXILIARES E KERNEL DE IRF
# ══════════════════════════════════════════════════════════════════════════════

def hex_to_rgba(hex_color, alpha=0.1):
    """Converte cor hex (#rrggbb) para rgba string."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"


def hex_to_rgb(hex_color):
    """Converte cor hex (#rrggbb) para rgb string."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"rgb({r},{g},{b})"


def gamma_irf_kernel(t, peak_q, half_life_q, peak_impact):
    """
    Gera IRF hump-shaped via distribuição gamma normalizada.
    Retorna array com o impacto em cada trimestre.
    """
    if peak_impact == 0 or half_life_q <= 0 or peak_q <= 0:
        return np.zeros(len(t))

    b = max(half_life_q / np.log(2), 0.5)
    a = peak_q / b + 1
    a = max(a, 1.01)

    raw = gamma_dist.pdf(t, a, scale=b)
    raw_peak = raw.max()
    if raw_peak > 0:
        return raw * (peak_impact / raw_peak)
    return np.zeros(len(t))


# ══════════════════════════════════════════════════════════════════════════════
# 2. BLOCO DE INFLAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

def inflation_headline_irf(params, shock_pct, horizon=HORIZON_QUARTERS):
    """IRF da inflação headline (pp) para um dado choque percentual no petróleo."""
    t = np.arange(0, horizon + 1, dtype=float)
    scale = shock_pct / 10.0
    shape = params["irf_shape"]
    pt = params["headline_passthrough_per_10pct"]

    irf_mid = gamma_irf_kernel(t, shape["headline_peak_q"], shape["decay_half_life_q"],
                               pt["mid"] * scale)
    irf_low = gamma_irf_kernel(t, shape["headline_peak_q"], shape["decay_half_life_q"],
                               pt["low"] * scale)
    irf_high = gamma_irf_kernel(t, shape["headline_peak_q"], shape["decay_half_life_q"],
                                pt["high"] * scale)
    return t, irf_mid, irf_low, irf_high


def inflation_core_irf(params, shock_pct, horizon=HORIZON_QUARTERS):
    """IRF da inflação core (pp)."""
    t = np.arange(0, horizon + 1, dtype=float)
    scale = shock_pct / 10.0
    shape = params["irf_shape"]
    pt_mid = params["headline_passthrough_per_10pct"]["mid"] * params["core_passthrough_ratio"]
    pt_low = params["headline_passthrough_per_10pct"]["low"] * params["core_passthrough_ratio"]
    pt_high = params["headline_passthrough_per_10pct"]["high"] * params["core_passthrough_ratio"]

    irf_mid = gamma_irf_kernel(t, shape["core_peak_q"], shape["decay_half_life_q"] + 1,
                               pt_mid * scale)
    irf_low = gamma_irf_kernel(t, shape["core_peak_q"], shape["decay_half_life_q"] + 1,
                               pt_low * scale)
    irf_high = gamma_irf_kernel(t, shape["core_peak_q"], shape["decay_half_life_q"] + 1,
                                pt_high * scale)
    return t, irf_mid, irf_low, irf_high


# ══════════════════════════════════════════════════════════════════════════════
# 3. BLOCO DE CRESCIMENTO
# ══════════════════════════════════════════════════════════════════════════════

def unemployment_irf(params, shock_pct, horizon=HORIZON_QUARTERS):
    """IRF do desemprego (pp) — exclusivo EUA."""
    t = np.arange(0, horizon + 1, dtype=float)
    scale = shock_pct / 10.0
    shape = params["irf_shape"]
    unemp = params["unemployment_impact_per_10pct"]

    irf_mid = gamma_irf_kernel(t, shape["unemp_peak_q"], shape["decay_half_life_q"] + 1,
                               unemp["mid"] * scale)
    irf_low = gamma_irf_kernel(t, shape["unemp_peak_q"], shape["decay_half_life_q"] + 1,
                               unemp["low"] * scale)
    irf_high = gamma_irf_kernel(t, shape["unemp_peak_q"], shape["decay_half_life_q"] + 1,
                                unemp["high"] * scale)
    return t, irf_mid, irf_low, irf_high


def gdp_total_irf(params, shock_pct, horizon=HORIZON_QUARTERS):
    """IRF do impacto no PIB (pp)."""
    t = np.arange(0, horizon + 1, dtype=float)
    scale = shock_pct / 10.0
    shape = params["irf_shape"]
    gd = params["gdp_impact_per_10pct"]

    irf_mid = gamma_irf_kernel(t, shape["gdp_peak_q"], shape["decay_half_life_q"],
                               gd["mid"] * scale)
    irf_low = gamma_irf_kernel(t, shape["gdp_peak_q"], shape["decay_half_life_q"],
                               gd["low"] * scale)
    irf_high = gamma_irf_kernel(t, shape["gdp_peak_q"], shape["decay_half_life_q"],
                                gd["high"] * scale)
    return t, irf_mid, irf_low, irf_high


def gdp_channel_decomposition(params, shock_pct, horizon=HORIZON_QUARTERS):
    """Decompõe impacto no PIB em canais: demanda, oferta, termos de troca."""
    t = np.arange(0, horizon + 1, dtype=float)
    scale = shock_pct / 10.0
    shape = params["irf_shape"]
    total_impact = params["gdp_impact_per_10pct"]["mid"] * scale
    ch = params["channels"]

    channels = {}
    channels["demanda"] = gamma_irf_kernel(
        t, max(shape["gdp_peak_q"] - 1, 1), shape["decay_half_life_q"],
        total_impact * ch["demanda"]
    )
    channels["oferta"] = gamma_irf_kernel(
        t, shape["gdp_peak_q"], shape["decay_half_life_q"],
        total_impact * ch["oferta"]
    )
    channels["termos_de_troca"] = gamma_irf_kernel(
        t, shape["gdp_peak_q"] + 1, shape["decay_half_life_q"] + 1,
        total_impact * ch["termos_de_troca"]
    )
    total = channels["demanda"] + channels["oferta"] + channels["termos_de_troca"]
    return t, total, channels


# ══════════════════════════════════════════════════════════════════════════════
# 4. BLOCO DE BALANÇA COMERCIAL
# ══════════════════════════════════════════════════════════════════════════════

def trade_balance_irf(params, shock_pct, horizon=HORIZON_QUARTERS):
    """
    Calcula impacto na balança comercial (USD bn por trimestre).
    Retorna: t, delta_exports, delta_imports, delta_balance
    """
    t = np.arange(0, horizon + 1, dtype=float)
    tr = params["trade"]
    shock_frac = shock_pct / 100.0

    # --- EXPORTAÇÕES ---
    oil_exp_value_direct = np.full(len(t), tr["oil_export_bn"] * shock_frac / 4.0)
    oil_exp_value_direct *= np.where(t <= 2, 1.0, np.maximum(0.7, 1.0 - 0.05 * (t - 2)))

    fx_change = tr["fx_oil_beta"] * shock_pct
    non_oil_exp = (tr["total_exports_bn"] - tr["oil_export_bn"]) / 4.0
    fx_export_boost = gamma_irf_kernel(t, 3, 3, non_oil_exp * abs(fx_change) / 100.0 * 0.3)
    if shock_pct > 0:
        fx_export_boost = np.abs(fx_export_boost)
    else:
        fx_export_boost = -np.abs(fx_export_boost)

    delta_exports = oil_exp_value_direct + fx_export_boost

    # --- IMPORTAÇÕES ---
    elast_path = np.linspace(tr["oil_import_volume_elasticity_sr"],
                             tr["oil_import_volume_elasticity_lr"], len(t))
    volume_adj = 1.0 - elast_path * abs(shock_frac)
    oil_imp_cost_change = tr["oil_import_bn"] / 4.0 * shock_frac * volume_adj

    non_oil_imp = (tr["total_imports_bn"] - tr["oil_import_bn"]) / 4.0
    gdp_mid = params["gdp_impact_per_10pct"]["mid"] * shock_pct / 10.0
    demand_effect = gamma_irf_kernel(t, 3, 3, non_oil_imp * abs(gdp_mid) / 100.0 * 0.5)
    if gdp_mid < 0:
        demand_effect = -np.abs(demand_effect)
    else:
        demand_effect = np.abs(demand_effect)

    delta_imports = oil_imp_cost_change + demand_effect
    delta_balance = delta_exports - delta_imports

    return t, delta_exports, delta_imports, delta_balance


# ══════════════════════════════════════════════════════════════════════════════
# 5. MOTOR DE SIMULAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

def run_scenario(target_price):
    """Roda todas as simulações para um dado preço-alvo do petróleo."""
    shock_pct = (target_price - BASELINE_OIL_PRICE) / BASELINE_OIL_PRICE * 100.0
    results = {}

    for name, params in COUNTRIES.items():
        t, hl_mid, hl_low, hl_high = inflation_headline_irf(params, shock_pct)
        _, cr_mid, cr_low, cr_high = inflation_core_irf(params, shock_pct)
        _, gdp_mid, gdp_low, gdp_high = gdp_total_irf(params, shock_pct)
        _, gdp_tot, gdp_channels = gdp_channel_decomposition(params, shock_pct)
        _, d_exp, d_imp, d_bal = trade_balance_irf(params, shock_pct)

        results[name] = {
            "t": t, "shock_pct": shock_pct,
            "headline_mid": hl_mid, "headline_low": hl_low, "headline_high": hl_high,
            "core_mid": cr_mid, "core_low": cr_low, "core_high": cr_high,
            "gdp_mid": gdp_mid, "gdp_low": gdp_low, "gdp_high": gdp_high,
            "gdp_total": gdp_tot, "gdp_channels": gdp_channels,
            "trade_exports": d_exp, "trade_imports": d_imp, "trade_balance": d_bal,
        }
    return results


def run_us_scenario(target_price):
    """Roda simulação para os EUA (aba separada)."""
    shock_pct = (target_price - BASELINE_OIL_PRICE) / BASELINE_OIL_PRICE * 100.0
    params = US_PARAMS

    t, hl_mid, hl_low, hl_high = inflation_headline_irf(params, shock_pct)
    _, cr_mid, cr_low, cr_high = inflation_core_irf(params, shock_pct)
    _, gdp_mid, gdp_low, gdp_high = gdp_total_irf(params, shock_pct)
    _, gdp_tot, gdp_channels = gdp_channel_decomposition(params, shock_pct)
    _, d_exp, d_imp, d_bal = trade_balance_irf(params, shock_pct)
    _, un_mid, un_low, un_high = unemployment_irf(params, shock_pct)

    return {
        "t": t, "shock_pct": shock_pct,
        "headline_mid": hl_mid, "headline_low": hl_low, "headline_high": hl_high,
        "core_mid": cr_mid, "core_low": cr_low, "core_high": cr_high,
        "gdp_mid": gdp_mid, "gdp_low": gdp_low, "gdp_high": gdp_high,
        "gdp_total": gdp_tot, "gdp_channels": gdp_channels,
        "trade_exports": d_exp, "trade_imports": d_imp, "trade_balance": d_bal,
        "unemp_mid": un_mid, "unemp_low": un_low, "unemp_high": un_high,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. TAB 1 — INFLAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

def plot_inflation_paths(results):
    """Grid 3x2: headline vs core por país."""
    names = list(results.keys())
    fig = make_subplots(rows=3, cols=2, subplot_titles=names,
                        vertical_spacing=0.12, horizontal_spacing=0.10)

    for i, name in enumerate(names):
        row, col = divmod(i, 2)
        row += 1; col += 1
        r = results[name]
        t = r["t"]
        color = COUNTRIES[name]["color"]

        # Fill entre headline e core
        fig.add_trace(go.Scatter(
            x=np.concatenate([t, t[::-1]]),
            y=np.concatenate([r["headline_mid"], r["core_mid"][::-1]]),
            fill="toself", fillcolor=hex_to_rgba(color, 0.1),
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip",
        ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=t, y=r["headline_mid"], mode="lines",
            name="Headline" if i == 0 else "", line=dict(color=color, width=2.5),
            showlegend=(i == 0), legendgroup="headline",
        ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=t, y=r["core_mid"], mode="lines",
            name="Core" if i == 0 else "", line=dict(color=color, width=2, dash="dash"),
            showlegend=(i == 0), legendgroup="core",
        ), row=row, col=col)

        fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.8,
                      row=row, col=col)

    shock = list(results.values())[0]["shock_pct"]
    direction = "alta" if shock > 0 else "queda"
    fig.update_layout(
        title=dict(text=f"Impacto na Inflação ({direction} de {abs(shock):.1f}%)",
                   font=dict(size=15)),
        height=850, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.07, xanchor="center", x=0.5),
        margin=dict(t=50, b=80, l=50, r=30),
    )
    fig.update_xaxes(dtick=2)
    fig.update_xaxes(title_text="Trimestres", row=3)
    fig.update_yaxes(title_text="pp")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 7. TAB 2 — CRESCIMENTO
# ══════════════════════════════════════════════════════════════════════════════

def plot_gdp_paths(results):
    """Overlay: 5 países num único gráfico."""
    fig = go.Figure()
    for name, r in results.items():
        color = COUNTRIES[name]["color"]
        style = LINE_STYLES[name]
        fig.add_trace(go.Scatter(
            x=r["t"], y=r["gdp_mid"], mode="lines+markers",
            name=name,
            line=dict(color=color, width=style["width"], dash=style["dash"]),
            marker=dict(color=color, size=6, symbol=style["symbol"]),
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)

    shock = list(results.values())[0]["shock_pct"]
    direction = "alta" if shock > 0 else "queda"
    fig.update_layout(
        title=dict(text=f"Impacto no PIB ({direction} de {abs(shock):.1f}%)",
                   font=dict(size=15)),
        xaxis_title="Trimestres", yaxis_title="pp",
        height=420, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5),
        margin=dict(t=50, b=70, l=50, r=30),
    )
    fig.update_xaxes(dtick=2)
    return fig


def plot_gdp_channels(results):
    """Grouped bar: decomposição por canal, impacto cumulativo até Q4."""
    names = list(results.keys())
    channel_names = ["demanda", "oferta", "termos_de_troca"]
    channel_labels = ["Demanda", "Oferta/Custo", "Termos de Troca"]
    channel_colors = ["#3498db", "#e67e22", "#2ecc71"]

    fig = go.Figure()
    for ch_name, ch_label, ch_color in zip(channel_names, channel_labels, channel_colors):
        vals = [np.sum(results[n]["gdp_channels"][ch_name][:5]) for n in names]
        fig.add_trace(go.Bar(
            name=ch_label, x=names, y=vals,
            marker_color=ch_color, opacity=0.85,
        ))

    totals = [np.sum(results[n]["gdp_total"][:5]) for n in names]
    fig.add_trace(go.Scatter(
        x=names, y=totals, mode="markers+text",
        name="Total", marker=dict(color="black", size=10, symbol="diamond"),
        text=[f"{v:+.2f}" for v in totals], textposition="top center",
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)

    shock = list(results.values())[0]["shock_pct"]
    direction = "alta" if shock > 0 else "queda"
    fig.update_layout(
        title=dict(text=f"Decomposição por Canal — Cumulativo até Q4 ({direction} de {abs(shock):.1f}%)",
                   font=dict(size=15)),
        yaxis_title="pp (cumulativo)", barmode="group",
        height=420, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5),
        margin=dict(t=50, b=70, l=50, r=30),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 8. TAB 3 — BALANÇA COMERCIAL
# ══════════════════════════════════════════════════════════════════════════════

def plot_trade_paths(results):
    """Grid 3x2: saldo comercial como % do PIB anual por país."""
    names = list(results.keys())
    fig = make_subplots(rows=3, cols=2, subplot_titles=names,
                        vertical_spacing=0.12, horizontal_spacing=0.10)

    for i, name in enumerate(names):
        row, col = divmod(i, 2)
        row += 1; col += 1
        r = results[name]
        t = r["t"]
        gdp_annual = COUNTRIES[name]["gdp_nominal_bn"]
        bal_pct = r["trade_balance"] / gdp_annual * 100 if gdp_annual > 0 else r["trade_balance"] * 0
        color = COUNTRIES[name]["color"]

        fig.add_trace(go.Scatter(
            x=t, y=bal_pct, mode="lines", fill="tozeroy",
            name=name if i == 0 else "",
            line=dict(color=color, width=2.5),
            fillcolor=hex_to_rgba(color, 0.15),
            showlegend=False,
        ), row=row, col=col)

        fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.8,
                      row=row, col=col)

    shock = list(results.values())[0]["shock_pct"]
    direction = "alta" if shock > 0 else "queda"
    fig.update_layout(
        title=dict(text=f"Saldo Comercial — % do PIB Anual ({direction} de {abs(shock):.1f}%)",
                   font=dict(size=15)),
        height=850, template="plotly_white",
        margin=dict(t=50, b=80, l=50, r=30),
    )
    fig.update_xaxes(dtick=2)
    fig.update_xaxes(title_text="Trimestres", row=3)
    fig.update_yaxes(title_text="% PIB anual")
    return fig


def plot_trade_waterfall(results):
    """Bar chart: saldo comercial acumulado Q0-Q4 como % do PIB anual."""
    names = list(results.keys())
    bal_vals = [np.sum(results[n]["trade_balance"][:5]) for n in names]
    gdp_annual = [COUNTRIES[n]["gdp_nominal_bn"] for n in names]
    bal_pct = [b / g * 100 if g > 0 else 0 for b, g in zip(bal_vals, gdp_annual)]
    colors = [COUNTRIES[n]["color"] for n in names]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=bal_pct,
        marker_color=colors, opacity=0.85,
        text=[f"{p:+.2f}%" for p in bal_pct],
        textposition="outside", textfont=dict(size=12, weight="bold"),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)

    shock = list(results.values())[0]["shock_pct"]
    direction = "alta" if shock > 0 else "queda"
    fig.update_layout(
        title=dict(text=f"Saldo Comercial Acumulado (Q0–Q4) — % do PIB Anual ({direction} de {abs(shock):.1f}%)",
                   font=dict(size=15)),
        yaxis_title="% PIB anual (acum. Q0–Q4)",
        height=420, template="plotly_white",
        margin=dict(t=50, b=70, l=50, r=30),
        showlegend=False,
    )
    return fig


def plot_fx_depreciation_bar(results):
    """Bar chart: depreciação cambial necessária ≈ −Δ conta petróleo líquida / PIB."""
    names = list(results.keys())
    depreciation = []

    for name in names:
        bal_cum = np.sum(results[name]["trade_balance"][:5])
        gdp = COUNTRIES[name]["gdp_nominal_bn"]
        depreciation.append(-bal_cum / gdp * 100 if gdp > 0 else 0)

    colors = [COUNTRIES[n]["color"] for n in names]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=depreciation,
        marker_color=colors, opacity=0.85,
        text=[f"{d:+.2f}%" for d in depreciation],
        textposition="outside", textfont=dict(size=13, weight="bold"),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)

    shock = list(results.values())[0]["shock_pct"]
    direction = "alta" if shock > 0 else "queda"
    fig.update_layout(
        title=dict(
            text=(f"Pressão Cambial: Depreciação Necessária "
                  f"({direction} de {abs(shock):.1f}%)<br>"
                  f"<span style='font-size:11px; color:#888;'>"
                  f"Depreciação % ≈ −Δ Conta Petróleo Líquida (acum. Q0–Q4) / PIB</span>"),
            font=dict(size=14),
        ),
        yaxis_title="Depreciação necessária (%)",
        height=420, template="plotly_white",
        margin=dict(t=80, b=70, l=50, r=30),
        showlegend=False,
    )
    return fig


def plot_oil_import_share():
    """Bar chart: importações de petróleo como % das importações totais."""
    names = list(COUNTRIES.keys())
    shares = [COUNTRIES[n]["trade"]["oil_import_bn"] / COUNTRIES[n]["trade"]["total_imports_bn"] * 100 for n in names]
    colors = [COUNTRIES[n]["color"] for n in names]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=shares,
        marker_color=colors, opacity=0.85,
        text=[f"{s:.1f}%" for s in shares],
        textposition="outside", textfont=dict(size=13, weight="bold"),
    ))
    fig.update_layout(
        title=dict(text="Importações de Petróleo como % das Importações Totais", font=dict(size=15)),
        yaxis_title="% das importações totais",
        height=420, template="plotly_white",
        margin=dict(t=50, b=70, l=50, r=30),
        showlegend=False,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 9. TAB 4 — IRFs COM BANDAS DE CONFIANÇA
# ══════════════════════════════════════════════════════════════════════════════

def _plot_irf_grid(results, var_prefix, title_prefix, y_label):
    """Grid 3x2 genérico para IRFs com bandas de confiança."""
    names = list(results.keys())
    fig = make_subplots(rows=3, cols=2, subplot_titles=names,
                        vertical_spacing=0.12, horizontal_spacing=0.10)

    for i, name in enumerate(names):
        row, col = divmod(i, 2)
        row += 1; col += 1
        r = results[name]
        t = r["t"]
        color = COUNTRIES[name]["color"]

        mid = r[f"{var_prefix}_mid"]
        low = r[f"{var_prefix}_low"]
        high = r[f"{var_prefix}_high"]
        band_68_low = 0.5 * (mid + low)
        band_68_high = 0.5 * (mid + high)

        # 90% band
        fig.add_trace(go.Scatter(
            x=np.concatenate([t, t[::-1]]),
            y=np.concatenate([high, low[::-1]]),
            fill="toself", fillcolor=hex_to_rgba(color, 0.12),
            line=dict(color="rgba(0,0,0,0)"),
            name="90%" if i == 0 else "", showlegend=(i == 0), legendgroup="90",
            hoverinfo="skip",
        ), row=row, col=col)

        # 68% band
        fig.add_trace(go.Scatter(
            x=np.concatenate([t, t[::-1]]),
            y=np.concatenate([band_68_high, band_68_low[::-1]]),
            fill="toself", fillcolor=hex_to_rgba(color, 0.25),
            line=dict(color="rgba(0,0,0,0)"),
            name="68%" if i == 0 else "", showlegend=(i == 0), legendgroup="68",
            hoverinfo="skip",
        ), row=row, col=col)

        # Central
        fig.add_trace(go.Scatter(
            x=t, y=mid, mode="lines",
            name="Central" if i == 0 else "",
            line=dict(color=hex_to_rgb(color), width=2.5),
            showlegend=(i == 0), legendgroup="central",
        ), row=row, col=col)

        fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.8,
                      row=row, col=col)

    shock = list(results.values())[0]["shock_pct"]
    direction = "alta" if shock > 0 else "queda"
    fig.update_layout(
        title=dict(text=f"IRF: {title_prefix} ({direction} de {abs(shock):.1f}%)",
                   font=dict(size=15)),
        height=850, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.07, xanchor="center", x=0.5),
        margin=dict(t=50, b=80, l=50, r=30),
    )
    fig.update_xaxes(dtick=2)
    fig.update_xaxes(title_text="Trimestres", row=3)
    fig.update_yaxes(title_text=y_label)
    return fig


def plot_irf_inflation(results):
    return _plot_irf_grid(results, "headline", "Inflação Headline", "pp")


def plot_irf_gdp(results):
    return _plot_irf_grid(results, "gdp", "Crescimento do PIB", "pp")


# ══════════════════════════════════════════════════════════════════════════════
# 9b. TAB 5 — EUA (ABA DEDICADA)
# ══════════════════════════════════════════════════════════════════════════════

def _us_irf_band(fig, t, mid, low, high, color, row, col):
    """Helper: adiciona bandas 68%/90% + linha central para um painel EUA."""
    band_68_low = 0.5 * (mid + low)
    band_68_high = 0.5 * (mid + high)

    # 90% band
    fig.add_trace(go.Scatter(
        x=np.concatenate([t, t[::-1]]),
        y=np.concatenate([high, low[::-1]]),
        fill="toself", fillcolor=hex_to_rgba(color, 0.10),
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip",
    ), row=row, col=col)

    # 68% band
    fig.add_trace(go.Scatter(
        x=np.concatenate([t, t[::-1]]),
        y=np.concatenate([band_68_high, band_68_low[::-1]]),
        fill="toself", fillcolor=hex_to_rgba(color, 0.22),
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip",
    ), row=row, col=col)

    # Central
    fig.add_trace(go.Scatter(
        x=t, y=mid, mode="lines",
        line=dict(color=hex_to_rgb(color), width=2.5),
        showlegend=False,
    ), row=row, col=col)

    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.8,
                  row=row, col=col)


def plot_us_tab(us_r):
    """Aba EUA: grid 3x2 com Headline PCE, Core PCE, GDP, Desemprego, Bal. Comercial, Headline vs Core."""
    color = US_PARAMS["color"]
    t = us_r["t"]
    shock = us_r["shock_pct"]
    direction = "alta" if shock > 0 else "queda"

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "Headline PCE (pp)", "Core PCE (pp)",
            "PIB (pp)", "Desemprego (pp)",
            "Balança Comercial (% PIB anual)", "Headline vs Core PCE",
        ],
        vertical_spacing=0.12, horizontal_spacing=0.10,
    )

    # (1,1) Headline PCE — IRF com bandas
    _us_irf_band(fig, t, us_r["headline_mid"], us_r["headline_low"], us_r["headline_high"],
                 color, 1, 1)

    # (1,2) Core PCE — IRF com bandas
    _us_irf_band(fig, t, us_r["core_mid"], us_r["core_low"], us_r["core_high"],
                 "#8e44ad", 1, 2)

    # (2,1) GDP — IRF com bandas
    _us_irf_band(fig, t, us_r["gdp_mid"], us_r["gdp_low"], us_r["gdp_high"],
                 "#c0392b", 2, 1)

    # (2,2) Desemprego — IRF com bandas
    _us_irf_band(fig, t, us_r["unemp_mid"], us_r["unemp_low"], us_r["unemp_high"],
                 "#d35400", 2, 2)

    # (3,1) Balança Comercial — % PIB
    gdp_annual = US_PARAMS["gdp_nominal_bn"]
    bal_pct = us_r["trade_balance"] / gdp_annual * 100
    fig.add_trace(go.Scatter(
        x=t, y=bal_pct, mode="lines", fill="tozeroy",
        line=dict(color="#27ae60", width=2.5),
        fillcolor=hex_to_rgba("#27ae60", 0.15),
        showlegend=False,
    ), row=3, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.8, row=3, col=1)

    # (3,2) Headline vs Core overlay
    fig.add_trace(go.Scatter(
        x=t, y=us_r["headline_mid"], mode="lines",
        name="Headline PCE", line=dict(color=color, width=2.5),
        showlegend=True, legendgroup="hl_us",
    ), row=3, col=2)
    fig.add_trace(go.Scatter(
        x=t, y=us_r["core_mid"], mode="lines",
        name="Core PCE", line=dict(color="#8e44ad", width=2, dash="dash"),
        showlegend=True, legendgroup="cr_us",
    ), row=3, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.8, row=3, col=2)

    fig.update_layout(
        title=dict(
            text=f"Estados Unidos — Impacto do Choque no Petróleo ({direction} de {abs(shock):.1f}%)",
            font=dict(size=15),
        ),
        height=920, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.05, xanchor="center", x=0.5),
        margin=dict(t=70, b=80, l=50, r=30),
    )
    fig.update_xaxes(dtick=2)
    fig.update_xaxes(title_text="Trimestres", row=3)
    for row in [1, 2]:
        for col in [1, 2]:
            fig.update_yaxes(title_text="pp", row=row, col=col)
    fig.update_yaxes(title_text="% PIB anual", row=3, col=1)
    fig.update_yaxes(title_text="pp", row=3, col=2)

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 9c. TAB 7 — GUERRA UCRANIANA (dados históricos, não varia com cenário)
# ══════════════════════════════════════════════════════════════════════════════

def build_geopolitical_table():
    """Tabela comparativa: Δ% de Brent + moedas EM em D+0, D+7, D+14 para cada evento."""
    data_list = GEOPOLITICAL_DATA

    series = [
        ("Brent", "brent", "#1a5276"),
        ("USDMXN", "USDMXN", "#2980b9"),
        ("USDCLP", "USDCLP", "#e74c3c"),
        ("USDCOP", "USDCOP", "#f39c12"),
        ("USDBRL", "USDBRL", "#FFD700"),
        ("USDZAR", "USDZAR", "#27ae60"),
    ]
    time_labels = ["D+0", "D+7", "D+14"]
    snap_keys = ["d0", "d7", "d14"]

    # Header: Ativo | Evento1 D+0 D+7 D+14 | Evento2 D+0 D+7 D+14 | ...
    parts = ['<table class="summary-table geo-table"><thead>']

    # Row 1: event names spanning 3 cols each
    parts.append('<tr><th class="st-country" rowspan="2">Ativo</th>')
    for ev in data_list:
        parts.append(f'<th colspan="3" class="geo-event-header">{ev["name"]}</th>')
    parts.append('</tr>')

    # Row 2: D+0, D+7, D+14 for each event
    parts.append('<tr>')
    for _ in data_list:
        for tl in time_labels:
            parts.append(f'<th class="geo-sub-header">{tl}</th>')
    parts.append('</tr></thead><tbody>')

    for s_name, s_key, s_color in series:
        parts.append(f'<tr><td class="st-country">'
                     f'<span class="st-dot" style="background:{s_color}"></span>'
                     f'<b>{s_name}</b></td>')
        for ev in data_list:
            vals = ev[s_key]
            snap = ev["snap"]
            base_val = vals[snap["eve"]]
            for sk in snap_keys:
                idx = snap[sk]
                v = vals[idx]
                pct = (v / base_val - 1) * 100 if base_val != 0 else 0
                css = "st-pos" if pct > 0.05 else ("st-neg" if pct < -0.05 else "")
                parts.append(f'<td class="{css}">{pct:+.1f}%</td>')
        parts.append('</tr>')

    parts.append('</tbody></table>')
    return ''.join(parts)


def plot_fx_distribution():
    """3 painéis (D+0, D+7, D+14): distribuição dos efeitos cambiais across events."""
    data_list = GEOPOLITICAL_DATA
    currencies = [
        ("MXN", "USDMXN", "#2980b9"),
        ("CLP", "USDCLP", "#e74c3c"),
        ("COP", "USDCOP", "#f39c12"),
        ("BRL", "USDBRL", "#FFD700"),
        ("ZAR", "USDZAR", "#27ae60"),
    ]
    horizons = [
        ("D+0 (Dia do Choque)", "d0"),
        ("D+7", "d7"),
        ("D+14", "d14"),
    ]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[h[0] for h in horizons],
        horizontal_spacing=0.08,
    )

    for col_idx, (h_label, h_key) in enumerate(horizons, 1):
        for ccy_name, ccy_key, ccy_color in currencies:
            # Coletar Δ% para esta moeda neste horizonte, across all events
            pct_vals = []
            for ev in data_list:
                vals = ev[ccy_key]
                snap = ev["snap"]
                base = vals[snap["eve"]]
                v = vals[snap[h_key]]
                pct_vals.append((v / base - 1) * 100 if base != 0 else 0)

            arr = np.array(pct_vals)
            mn, mx = arr.min(), arr.max()
            p20, p80 = np.percentile(arr, 20), np.percentile(arr, 80)
            med = np.median(arr)
            mean = arr.mean()

            show_leg = (col_idx == 1)

            # Whisker: min to max (thin line)
            fig.add_trace(go.Scatter(
                x=[mn, mx], y=[ccy_name, ccy_name],
                mode="lines",
                line=dict(color=ccy_color, width=1.5),
                showlegend=False, hoverinfo="skip",
            ), row=1, col=col_idx)

            # Min/Max caps
            fig.add_trace(go.Scatter(
                x=[mn, mx], y=[ccy_name, ccy_name],
                mode="markers",
                marker=dict(color=ccy_color, size=8, symbol="line-ns-open",
                            line=dict(width=1.5)),
                showlegend=False,
                hovertemplate=(f"<b>{ccy_name}</b><br>"
                               f"Min: {mn:+.2f}%<br>Max: {mx:+.2f}%<extra></extra>"),
            ), row=1, col=col_idx)

            # P20–P80 range (thick bar)
            fig.add_trace(go.Scatter(
                x=[p20, p80], y=[ccy_name, ccy_name],
                mode="lines",
                line=dict(color=ccy_color, width=14),
                opacity=0.40,
                showlegend=False,
                hovertemplate=(f"<b>{ccy_name}</b><br>"
                               f"P20: {p20:+.2f}%<br>P80: {p80:+.2f}%<extra></extra>"),
            ), row=1, col=col_idx)

            # Median (circle)
            fig.add_trace(go.Scatter(
                x=[med], y=[ccy_name],
                mode="markers",
                marker=dict(color="white", size=10, symbol="circle",
                            line=dict(color=ccy_color, width=2.5)),
                name="Mediana" if (show_leg and ccy_name == "MXN") else "",
                showlegend=(show_leg and ccy_name == "MXN"),
                legendgroup="median",
                hovertemplate=(f"<b>{ccy_name}</b><br>"
                               f"Mediana: {med:+.2f}%<extra></extra>"),
            ), row=1, col=col_idx)

            # Mean (diamond)
            fig.add_trace(go.Scatter(
                x=[mean], y=[ccy_name],
                mode="markers",
                marker=dict(color=ccy_color, size=9, symbol="diamond"),
                name="Média" if (show_leg and ccy_name == "MXN") else "",
                showlegend=(show_leg and ccy_name == "MXN"),
                legendgroup="mean",
                hovertemplate=(f"<b>{ccy_name}</b><br>"
                               f"Média: {mean:+.2f}%<extra></extra>"),
            ), row=1, col=col_idx)

        # Linha vertical em 0%
        fig.add_vline(x=0, line_dash="dash", line_color="#aaa", line_width=0.8,
                      row=1, col=col_idx)

    fig.update_layout(
        title=dict(
            text=(f"Distribuição dos Efeitos Cambiais — {len(data_list)} Choques Geopolíticos<br>"
                  "<span style='font-size:11px; color:#888;'>"
                  "Barra: P20–P80 | Whisker: Min–Max | "
                  "◇ Média | ○ Mediana | "
                  "Valores positivos = depreciação da moeda local</span>"),
            font=dict(size=14),
        ),
        height=350, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.20,
                    xanchor="center", x=0.5),
        margin=dict(t=80, b=60, l=50, r=20),
    )
    fig.update_xaxes(title_text="Δ% vs véspera", zeroline=True)
    fig.update_yaxes(autorange="reversed")

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 10. TAB 6 — REFERÊNCIAS
# ══════════════════════════════════════════════════════════════════════════════

REFERENCES = {
    "Bancos Centrais": [
        {"autor": "Banco Central de Chile", "titulo": "Exchange Rate Pass-Through to Prices: VAR Evidence for Chile", "ref": "Working Paper No. 747", "ano": 2016, "url": "https://www.bcentral.cl/en/web/banco-central/content/-/detalle/working-papers-2016"},
        {"autor": "Banco Central de Chile", "titulo": "Global monetary policy surprises and their transmission to emerging market economies", "ref": "Working Paper No. 975", "ano": 2023, "url": "https://www.bcentral.cl/en/web/banco-central/content/-/detalle/working-papers-2023"},
        {"autor": "Banco de la República (Colômbia)", "titulo": "Commodity Price Shocks and Inflation within An Optimal Monetary Policy Framework", "ref": "Borradores de Economía No. 858", "ano": 2014, "url": "https://www.banrep.gov.co/en/borrador-858"},
        {"autor": "Banxico (México)", "titulo": "Quarterly Inflation Report / Monetary Policy Statements", "ref": "Relatórios periódicos sobre pass-through energético", "ano": 2024, "url": "https://www.banxico.org.mx/publicaciones-y-prensa/informes-trimestrales/"},
        {"autor": "South African Reserve Bank", "titulo": "Monetary Policy Review / Oil price transmission studies", "ref": "SARB Occasional Bulletins", "ano": 2024, "url": "https://www.resbank.co.za/en/home/publications/review"},
        {"autor": "Banco Central do Brasil", "titulo": "Relatório de Inflação — Análise de pass-through de preços de combustíveis", "ref": "Relatório de Inflação (vários trimestres)", "ano": 2024, "url": "https://www.bcb.gov.br/publicacoes/ri"},
        {"autor": "Correa, A. & Minella, A.", "titulo": "Nonlinear Mechanisms of the Exchange Rate Pass-Through", "ref": "BCB Working Paper 556", "ano": 2012, "url": "https://www.bcb.gov.br/pec/wps/ingl/wps556.pdf"},
    ],
    "FMI / World Bank": [
        {"autor": "Baba, C. & Lee, J.", "titulo": "Second-Round Effects of Oil Price Shocks", "ref": "IMF WP/22/173", "ano": 2022, "url": "https://www.imf.org/-/media/Files/Publications/WP/2022/English/wpiea2022173-print-pdf.ashx"},
        {"autor": "Choi, S. et al.", "titulo": "Oil Prices and Inflation Dynamics: Evidence from Advanced and Developing Economies", "ref": "IMF WP/17/196", "ano": 2017, "url": "https://www.imf.org/-/media/Files/Publications/WP/2017/wp17196.pdf"},
        {"autor": "Arezki, R. & Blanchard, O.", "titulo": "Oil Prices and the Global Economy", "ref": "IMF WP/17/15", "ano": 2017, "url": "https://www.imf.org/-/media/Files/Publications/WP/wp1715.ashx"},
        {"autor": "Hakura, D.", "titulo": "Trade Elasticities in the Middle East and Central Asia: What is the Role of Oil?", "ref": "IMF WP/08/216", "ano": 2008, "url": "https://www.imf.org/external/pubs/ft/wp/2008/wp08216.pdf"},
        {"autor": "Coady, D. et al.", "titulo": "To Pass (or Not to Pass) Through International Fuel Price Changes", "ref": "IMF WP/20/194", "ano": 2020, "url": "https://www.imf.org/-/media/Files/Publications/WP/2020/English/wpiea2020194-print-pdf.ashx"},
        {"autor": "World Bank", "titulo": "Global Economic Prospects", "ref": "June 2025 Report", "ano": 2025, "url": "https://thedocs.worldbank.org/en/doc/8bf0b62ec6bcb886d97295ad930059e9-0050012025/original/GEP-June-2025.pdf"},
        {"autor": "World Bank", "titulo": "Commodity Markets Outlook", "ref": "April 2025 Report", "ano": 2025, "url": "https://thedocs.worldbank.org/en/doc/1b388949805c9a0ae3736bdacb32ea94-0050012025/original/CMO-April-2025.pdf"},
    ],
    "Federal Reserve": [
        {"autor": "Presno, I. & Prestipino, A.", "titulo": "Oil Price Shocks and Inflation in a DSGE Model of the Global Economy", "ref": "FEDS Notes", "ano": 2024, "url": "https://www.federalreserve.gov/econres/notes/feds-notes/oil-price-shocks-and-inflation-in-a-dsge-model-of-the-global-economy-20240802.html"},
        {"autor": "Federal Reserve Board", "titulo": "Second-Round Effects of Oil Prices on Inflation in the AFEs", "ref": "FEDS Notes", "ano": 2023, "url": "https://www.federalreserve.gov/econres/notes/feds-notes/second-round-effects-of-oil-prices-on-inflation-in-the-advanced-foreign-economies-20231215.html"},
        {"autor": "Federal Reserve Board", "titulo": "Oil Price Pass-Through into Core Inflation", "ref": "FEDS Notes", "ano": 2017, "url": "https://www.federalreserve.gov/econres/notes/feds-notes/oil-price-pass-through-into-core-inflation-20171019.html"},
        {"autor": "Kilian, L. & Zhou, X.", "titulo": "Oil Price Shocks and Inflation", "ref": "Dallas Fed WP 2312", "ano": 2023, "url": "https://www.dallasfed.org/research/papers/2023/wp2312"},
        {"autor": "Kilian, L. & Zhou, X.", "titulo": "The Impact of Rising Oil Prices on U.S. Inflation and Inflation Expectations in 2020-23", "ref": "Dallas Fed WP 2116", "ano": 2022, "url": "https://www.dallasfed.org/research/papers/2021/wp2116"},
        {"autor": "Caldara, D. et al.", "titulo": "Geopolitical Oil Price Risk and Economic Fluctuations", "ref": "Dallas Fed WP 2403", "ano": 2024, "url": "https://www.dallasfed.org/~/media/documents/research/papers/2024/wp2403.pdf"},
        {"autor": "Hobijn, B.", "titulo": "Commodity Price Movements and PCE Inflation", "ref": "NY Fed Current Issues Vol. 14 No. 8", "ano": 2008, "url": "https://www.newyorkfed.org/research/current_issues/ci14-8.html"},
        {"autor": "Newell, R. & Prest, B.", "titulo": "Oil Price Elasticities and Oil Price Fluctuations", "ref": "IFDP 1173", "ano": 2016, "url": "https://www.federalreserve.gov/econresdata/ifdp/2016/files/ifdp1173.pdf"},
    ],
    "Calibração EUA": [
        {"autor": "Gagliardone, L. & Gertler, M.", "titulo": "Oil Prices, Monetary Policy and Inflation Surges", "ref": "NBER WP 31263", "ano": 2023, "url": "https://www.nber.org/papers/w31263"},
        {"autor": "Oladosu, G. et al.", "titulo": "Impacts of oil price shocks on the US economy: A meta-analysis of the oil price elasticity of GDP", "ref": "Energy Policy 115", "ano": 2018, "url": "https://www.sciencedirect.com/science/article/abs/pii/S0301421518300417"},
        {"autor": "Kängzig, D.", "titulo": "The Macroeconomic Effects of Oil Supply News", "ref": "AER 111(4)", "ano": 2021, "url": "https://www.aeaweb.org/articles?id=10.1257/aer.20190964"},
        {"autor": "IMF", "titulo": "Oil Shocks and Labor Market Developments", "ref": "IMF WP 2025/145", "ano": 2025, "url": "https://www.imf.org/-/media/Files/Publications/WP/2025/English/WPIEA2025145.ashx"},
        {"autor": "Kilian, L.", "titulo": "Not All Oil Price Shocks Are Alike: Disentangling Demand and Supply Shocks", "ref": "AER 99(3)", "ano": 2009, "url": "https://www.aeaweb.org/articles?id=10.1257/aer.99.3.1053"},
        {"autor": "Hamilton, J.", "titulo": "Causes and Consequences of the Oil Shock of 2007-08", "ref": "Brookings Papers on Economic Activity", "ano": 2009, "url": "https://www.brookings.edu/wp-content/uploads/2016/07/2009a_bpea_hamilton-1.pdf"},
    ],
    "Calibração Brasil": [
        {"autor": "IPEA", "titulo": "Carta de Conjuntura — Impactos de choques de commodities na economia brasileira", "ref": "Carta de Conjuntura (vários números)", "ano": 2024, "url": "https://www.ipea.gov.br/cartadeconjuntura/"},
        {"autor": "IPEA", "titulo": "Impactos macroeconômicos de variações nos preços do petróleo sobre a economia brasileira", "ref": "Texto para Discussão", "ano": 2023, "url": "https://www.ipea.gov.br/portal/publicacoes"},
        {"autor": "Ministério da Fazenda / SPE", "titulo": "Receitas governamentais do setor de petróleo e gás", "ref": "Boletim de Receitas de Petróleo", "ano": 2024, "url": "https://www.gov.br/fazenda/pt-br"},
        {"autor": "ANP", "titulo": "Anuário Estatístico Brasileiro do Petróleo, Gás Natural e Biocombustíveis", "ref": "Anuário Estatístico 2024", "ano": 2024, "url": "https://www.gov.br/anp/pt-br/centrais-de-conteudo/publicacoes/anuario-estatistico"},
        {"autor": "Petrobras", "titulo": "Relatório de Produção e Vendas / Política de Preços", "ref": "RI Petrobras", "ano": 2024, "url": "https://www.investidorpetrobras.com.br/"},
    ],
    "Literatura Acadêmica": [
        {"autor": "Tandfonline (2025)", "titulo": "Oil price passthrough to consumer price inflation in South Africa: the role of the inflation environment", "ref": "Latin American Economic Review", "ano": 2025, "url": "https://www.tandfonline.com/doi/full/10.1080/15140326.2025.2509228"},
        {"autor": "MDPI (2024)", "titulo": "Comparative Analysis of VAR and SVAR Models — Oil Price Shocks in South Africa", "ref": "Econometrics 13(1):8", "ano": 2024, "url": "https://www.mdpi.com/2225-1146/13/1/8"},
        {"autor": "FLAR", "titulo": "Oil Price Volatility and Latin American Growth", "ref": "FLAR Research Paper", "ano": 2024, "url": "https://flar.com/wp-content/uploads/2024/04/Paper_Oil-Price-Volatility-and-Latin-American-Growth.pdf"},
        {"autor": "IDB", "titulo": "Fuel-Price Shocks and Inflation in Latin America and the Caribbean", "ref": "Inter-American Development Bank", "ano": 2023, "url": "https://publications.iadb.org/publications/english/document/fuel-price-shocks-and-inflation-in-latin-america-and-the-caribbean-january-2023.pdf"},
    ],
    "Dados e Estatísticas": [
        {"autor": "OEC World", "titulo": "Trade data — Chile, México, Colômbia, Brasil, África do Sul", "ref": "Observatory of Economic Complexity", "ano": 2024, "url": "https://oec.world/"},
        {"autor": "IEA", "titulo": "Oil balances and energy statistics", "ref": "International Energy Agency", "ano": 2024, "url": "https://www.iea.org/countries/"},
        {"autor": "PEMEX / ANH / ANP / EIA", "titulo": "Oil production and export data", "ref": "Agências nacionais de petróleo", "ano": 2024, "url": ""},
        {"autor": "MDIC / Comex Stat", "titulo": "Balança comercial brasileira — Petróleo e derivados", "ref": "Ministério da Indústria e Comércio", "ano": 2024, "url": "https://comexstat.mdic.gov.br/"},
    ],
}


def build_references_html():
    """Gera HTML da aba de referências."""
    parts = [
        '<div style="max-width: 900px; margin: 0 auto;">',
        '<p style="color: #555; font-style: italic; margin-bottom: 20px;">',
        '<b>Nota metodológica:</b> Os parâmetros deste dashboard são calibrados a partir da ',
        'literatura acadêmica e estudos de bancos centrais listados abaixo. Não são estimados ',
        'diretamente via modelos econométricos com dados primários. As elasticidades e coeficientes ',
        'de pass-through representam estimativas centrais da literatura recente (2018–2025), ',
        'com bandas de incerteza derivadas da dispersão entre estudos.</p>',
    ]
    for category, refs in REFERENCES.items():
        parts.append(f'<h3 style="color: #2c3e50; border-bottom: 2px solid #2980b9; '
                     f'padding-bottom: 5px; margin-top: 25px;">{category}</h3>')
        parts.append('<ul>')
        for r in refs:
            link = f' <a href="{r["url"]}" target="_blank" style="color: #2980b9;">[link]</a>' if r.get("url") else ""
            parts.append(
                f'<li style="margin-bottom: 8px;">'
                f'<b>{r["autor"]}</b> ({r["ano"]}). '
                f'<i>{r["titulo"]}</i>. {r["ref"]}.{link}</li>'
            )
        parts.append('</ul>')
    parts.append('</div>')
    return '\n'.join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# 11. TABELA RESUMO
# ══════════════════════════════════════════════════════════════════════════════

def compute_summary_table(results, us_results=None):
    """Tabela resumo com principais métricas por país (+ EUA se fornecido)."""
    rows = []
    for name, r in results.items():
        hl, cr, gd = r["headline_mid"], r["core_mid"], r["gdp_mid"]
        bal = r["trade_balance"]
        peak_hl = hl[np.argmax(np.abs(hl))]
        peak_cr = cr[np.argmax(np.abs(cr))]
        peak_gd = gd[np.argmax(np.abs(gd))]
        peak_q = int(np.argmax(np.abs(hl)))
        peak_bal = bal[np.argmax(np.abs(bal))]
        gdp_annual = COUNTRIES[name]["gdp_nominal_bn"]
        peak_bal_pct = peak_bal / gdp_annual * 100 if gdp_annual > 0 else 0

        rows.append({
            "País": name,
            "Pico Headline (pp)": f"{peak_hl:+.2f}",
            "Pico Core (pp)": f"{peak_cr:+.2f}",
            "Q do Pico": f"Q{peak_q}",
            "Pico PIB (pp)": f"{peak_gd:+.2f}",
            "Pico Desemp. (pp)": "—",
            "Pico Bal. Com. (% PIB)": f"{peak_bal_pct:+.2f}%",
        })

    if us_results is not None:
        r = us_results
        hl, cr, gd = r["headline_mid"], r["core_mid"], r["gdp_mid"]
        un = r["unemp_mid"]
        bal = r["trade_balance"]
        peak_hl = hl[np.argmax(np.abs(hl))]
        peak_cr = cr[np.argmax(np.abs(cr))]
        peak_gd = gd[np.argmax(np.abs(gd))]
        peak_un = un[np.argmax(np.abs(un))]
        peak_q = int(np.argmax(np.abs(hl)))
        peak_bal = bal[np.argmax(np.abs(bal))]
        gdp_annual = US_PARAMS["gdp_nominal_bn"]
        peak_bal_pct = peak_bal / gdp_annual * 100 if gdp_annual > 0 else 0

        rows.append({
            "País": "EUA",
            "Pico Headline (pp)": f"{peak_hl:+.2f}",
            "Pico Core (pp)": f"{peak_cr:+.2f}",
            "Q do Pico": f"Q{peak_q}",
            "Pico PIB (pp)": f"{peak_gd:+.2f}",
            "Pico Desemp. (pp)": f"{peak_un:+.2f}",
            "Pico Bal. Com. (% PIB)": f"{peak_bal_pct:+.2f}%",
        })

    return pd.DataFrame(rows).set_index("País")


def df_to_html_table(df):
    """Converte DataFrame em tabela HTML formatada com design profissional."""
    country_colors = {
        "Chile": "#e74c3c", "México": "#2980b9",
        "Colômbia": "#f39c12", "Brasil": "#FFD700",
        "África do Sul": "#27ae60", "EUA": "#1a5276",
    }

    parts = ['<table class="summary-table"><thead><tr>']
    parts.append(f'<th class="st-country">{df.index.name or ""}</th>')
    for col in df.columns:
        parts.append(f'<th>{col}</th>')
    parts.append('</tr></thead><tbody>')

    for idx, row in df.iterrows():
        color = country_colors.get(idx, "#333")
        parts.append(f'<tr><td class="st-country">'
                     f'<span class="st-dot" style="background:{color}"></span>'
                     f'<b>{idx}</b></td>')
        for val in row:
            css = ""
            s = str(val).strip()
            if s.startswith("+") and s != "+0.00" and s != "+0.00%":
                css = ' class="st-pos"'
            elif s.startswith("-"):
                css = ' class="st-neg"'
            parts.append(f'<td{css}>{val}</td>')
        parts.append('</tr>')

    parts.append('</tbody></table>')
    return ''.join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# 12. MONTAGEM DO HTML — SEM str.format(), USA REPLACE
# ══════════════════════════════════════════════════════════════════════════════

CHART_IDS = [
    "chart-infl-paths",
    "chart-gdp-paths",
    "chart-trade-paths", "chart-trade-waterfall", "chart-fx-depreciation", "chart-oil-import-share",
    "chart-irf-infl", "chart-irf-gdp",
    "chart-us-panel",
]


def generate_all_charts(results, us_results):
    """Gera todos os gráficos para um cenário (EMs + EUA)."""
    to_div = lambda fig: pio.to_html(fig, full_html=False, include_plotlyjs=False)
    return {
        "chart-infl-paths": to_div(plot_inflation_paths(results)),
        "chart-gdp-paths": to_div(plot_gdp_paths(results)),
        "chart-trade-paths": to_div(plot_trade_paths(results)),
        "chart-trade-waterfall": to_div(plot_trade_waterfall(results)),
        "chart-fx-depreciation": to_div(plot_fx_depreciation_bar(results)),
        "chart-oil-import-share": to_div(plot_oil_import_share()),
        "chart-irf-infl": to_div(plot_irf_inflation(results)),
        "chart-irf-gdp": to_div(plot_irf_gdp(results)),
        "chart-us-panel": to_div(plot_us_tab(us_results)),
    }


def build_html_dashboard():
    """Constrói o dashboard HTML com cenários pré-renderizados via show/hide."""

    import json as _json

    references_html = build_references_html()

    # Pré-renderizar aba Choques Geopolíticos (estática, não depende do cenário)
    _to_div = lambda fig: pio.to_html(fig, full_html=False, include_plotlyjs=False)
    geopolitical_table_html = build_geopolitical_table()
    fx_dist_html = _to_div(plot_fx_distribution())

    # Pré-computar todos os cenários (slider de $5 em $5)
    all_slider = []
    n_total = len(SLIDER_PRICES)
    for idx, price in enumerate(SLIDER_PRICES):
        shock_pct = (price - BASELINE_OIL_PRICE) / BASELINE_OIL_PRICE * 100.0
        sign = "+" if shock_pct > 0 else ""
        label = f"${price}/bbl ({sign}{shock_pct:.1f}%)"

        print(f"  Cenário {idx+1}/{n_total}: ${price}/bbl ({sign}{shock_pct:.1f}%)")
        results = run_scenario(price)
        us_results = run_us_scenario(price)
        charts = generate_all_charts(results, us_results)
        summary_html = df_to_html_table(compute_summary_table(results, us_results))

        all_slider.append({
            "price": price,
            "label": label,
            "charts": charts,
            "summary_html": summary_html,
        })

    # Mapear cenários nomeados para índices do slider
    scenario_to_idx = {}
    for sc in SCENARIOS:
        scenario_to_idx[sc["target"]] = SLIDER_PRICES.index(sc["target"])

    # Descrições dos cenários nomeados indexadas pelo slider
    named_descs = {}
    for sc in SCENARIOS:
        sidx = SLIDER_PRICES.index(sc["target"])
        named_descs[sidx] = sc["desc"]

    # ── Construir botões de cenários nomeados ──
    buttons_html = []
    for sc in SCENARIOS:
        sidx = SLIDER_PRICES.index(sc["target"])
        cls = "scenario-btn"
        if sc.get("default"):
            cls += " active"
        if sc.get("highlight"):
            cls += " highlight"
        buttons_html.append(
            f'<button class="{cls}" data-idx="{sidx}" onclick="jumpToScenario({sidx}, this)" '
            f'title="{sc["desc"]}">{sc["name"]}<br>'
            f'<small>${sc["target"]}/bbl</small></button>'
        )

    # ── Construir divs de gráficos (todos os cenários do slider) ──
    chart_divs = {cid: [] for cid in CHART_IDS}
    summary_divs = []

    for i, sc in enumerate(all_slider):
        display = "block" if i == DEFAULT_SLIDER_IDX else "none"
        for cid in CHART_IDS:
            chart_divs[cid].append(
                f'<div id="{cid}-{i}" class="scenario-panel s-{i}" style="display:{display}">'
                f'{sc["charts"][cid]}</div>'
            )
        summary_divs.append(
            f'<div id="summary-{i}" class="scenario-panel s-{i}" style="display:{display}">'
            f'{sc["summary_html"]}</div>'
        )

    # ── JSON para JavaScript ──
    slider_prices_json = _json.dumps(SLIDER_PRICES)
    named_descs_json = _json.dumps(named_descs)
    default_shock = (DEFAULT_PRICE - BASELINE_OIL_PRICE) / BASELINE_OIL_PRICE * 100.0
    default_shock_label = f"{'+'if default_shock > 0 else ''}{default_shock:.1f}%"

    # ── Construir HTML final ──
    html = f'''<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Oil Shock: EM Effects</title>
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
  /* ── Simulador ── */
  .simulator {{
    background: #fff; border-left: 4px solid #e74c3c;
    padding: 15px 20px; margin: 15px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-radius: 0 5px 5px 0;
  }}
  .simulator-title {{
    font-weight: 700; font-size: 15px; color: #2c3e50; margin-bottom: 12px;
  }}
  .price-control {{
    display: flex; align-items: center; gap: 12px; margin-bottom: 12px;
    flex-wrap: wrap;
  }}
  .price-control label {{
    font-weight: 600; font-size: 14px; color: #2c3e50; white-space: nowrap;
  }}
  .price-input {{
    width: 80px; padding: 6px 8px; font-size: 15px; font-weight: 600;
    border: 2px solid #bdc3c7; border-radius: 5px; text-align: center;
    color: #2c3e50; transition: border-color 0.2s;
  }}
  .price-input:focus {{
    border-color: #2980b9; outline: none;
  }}
  .price-unit {{
    font-size: 13px; color: #666;
  }}
  .price-slider {{
    flex: 1; min-width: 200px; height: 6px;
    -webkit-appearance: none; appearance: none;
    background: linear-gradient(to right, #27ae60 0%, #f39c12 50%, #e74c3c 100%);
    border-radius: 3px; outline: none;
  }}
  .price-slider::-webkit-slider-thumb {{
    -webkit-appearance: none; width: 20px; height: 20px;
    border-radius: 50%; background: #2980b9; cursor: pointer;
    border: 2px solid #fff; box-shadow: 0 1px 4px rgba(0,0,0,0.3);
  }}
  .price-slider::-moz-range-thumb {{
    width: 20px; height: 20px; border-radius: 50%;
    background: #2980b9; cursor: pointer;
    border: 2px solid #fff; box-shadow: 0 1px 4px rgba(0,0,0,0.3);
  }}
  .shock-label {{
    font-size: 14px; font-weight: 700; min-width: 70px; text-align: center;
    padding: 4px 10px; border-radius: 4px;
  }}
  .shock-positive {{ color: #c0392b; background: #fdecea; }}
  .shock-negative {{ color: #27ae60; background: #e8f8f0; }}
  .shock-zero {{ color: #666; background: #f0f0f0; }}
  .scenario-desc {{
    font-size: 13px; color: #666; margin-top: 8px; font-style: italic;
    min-height: 18px;
  }}
  .scenario-buttons {{ display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }}
  .scenario-label {{
    font-size: 12px; color: #888; margin-right: 2px; white-space: nowrap;
  }}
  .scenario-btn {{
    padding: 6px 12px; border: 2px solid #bdc3c7; border-radius: 6px;
    background: #fff; cursor: pointer; font-size: 12px; text-align: center;
    transition: all 0.2s; color: #333; line-height: 1.3;
  }}
  .scenario-btn:hover {{ border-color: #2980b9; background: #eaf2f8; }}
  .scenario-btn.active {{
    border-color: #2980b9; background: #2980b9; color: #fff;
  }}
  .scenario-btn.highlight {{
    border-color: #e74c3c;
  }}
  .scenario-btn.highlight.active {{
    background: #c0392b; border-color: #c0392b; color: #fff;
  }}
  .scenario-btn small {{ font-size: 10px; opacity: 0.8; }}
  /* ── Tabs ── */
  .tab-container {{ margin: 20px 0; }}
  .tab-buttons {{
    display: flex; flex-wrap: wrap; background: #2c3e50;
    border-radius: 5px 5px 0 0; overflow: hidden;
  }}
  .tab-btn {{
    padding: 12px 18px; color: #ecf0f1; cursor: pointer;
    border: none; background: none; font-size: 13px; font-weight: 500;
    transition: background 0.2s; white-space: nowrap;
  }}
  .tab-btn:hover {{ background: #34495e; }}
  .tab-btn.active {{ background: #2980b9; color: white; }}
  .tab-content {{
    display: none; padding: 20px; background: white;
    border: 1px solid #ddd; border-top: none; border-radius: 0 0 5px 5px;
  }}
  .tab-content.active {{ display: block; }}
  /* ── Resumo ── */
  .summary-box {{
    background: #fff; border-left: 4px solid #2980b9;
    padding: 12px 20px; margin: 15px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-radius: 0 5px 5px 0;
    overflow-x: auto;
  }}
  /* ── Tabelas ── */
  table {{ border-collapse: collapse; width: 100%; margin: 8px 0; background: #fff; font-size: 12px; }}
  th, td {{ border: 1px solid #ddd; padding: 7px 10px; text-align: center; }}
  th {{ background: #2c3e50; color: #fff; font-weight: 600; font-size: 11px; }}
  tr:nth-child(even) {{ background: #f8f9fa; }}
  /* ── Summary Table ── */
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
    text-align: left; font-family: 'Segoe UI', sans-serif;
    white-space: nowrap;
  }}
  .summary-table .st-dot {{
    display: inline-block; width: 8px; height: 8px;
    border-radius: 50%; margin-right: 6px; vertical-align: middle;
  }}
  .summary-table .st-pos {{ color: #c0392b; font-weight: 600; }}
  .summary-table .st-neg {{ color: #27ae60; font-weight: 600; }}
  /* ── Geopolitical table ── */
  .geo-table {{ font-size: 12px; }}
  .geo-table .geo-event-header {{
    background: #1a252f; color: #ecf0f1; text-align: center;
    border-left: 2px solid #2980b9; font-size: 11px;
  }}
  .geo-table .geo-sub-header {{
    background: #2c3e50; color: #bdc3c7; font-size: 10px;
    text-transform: uppercase; letter-spacing: 0.5px; padding: 6px 8px;
  }}
  .geo-table tbody td {{
    padding: 8px 10px; font-size: 12px; text-align: center;
    min-width: 60px;
  }}
  .chart-container {{ margin: 15px 0; }}
  .timestamp {{ color: #999; font-size: 12px; }}
  .source-note {{ font-size: 11px; color: #888; margin-top: 2px; font-style: italic; }}
</style>
</head>
<body>
<h1>Oil Shock: EM Effects</h1>
<p class="timestamp">Gerado em: {datetime.now().strftime("%Y-%m-%d %H:%M")} | Base: ${BASELINE_OIL_PRICE:.0f}/bbl</p>

<div class="simulator">
  <div class="simulator-title">Simulador de Cenários</div>

  <div class="price-control">
    <label for="price-input">Preço do Petróleo:</label>
    <input type="number" id="price-input" class="price-input"
           min="{SLIDER_PRICES[0]}" max="{SLIDER_PRICES[-1]}" step="5"
           value="{DEFAULT_PRICE}"
           onchange="onPriceInput(this.value)">
    <span class="price-unit">USD/bbl</span>
    <input type="range" id="price-slider" class="price-slider"
           min="0" max="{len(SLIDER_PRICES)-1}" value="{DEFAULT_SLIDER_IDX}"
           oninput="onSliderChange(this.value)">
    <span id="shock-label" class="shock-label shock-positive">{default_shock_label}</span>
  </div>

  <div class="scenario-buttons">
    <span class="scenario-label">Cenários:</span>
    {''.join(buttons_html)}
  </div>
  <div class="scenario-desc" id="scenario-desc">{named_descs.get(DEFAULT_SLIDER_IDX, "")}</div>
</div>

<div class="summary-box">
  {''.join(summary_divs)}
</div>

<div class="tab-container">
  <div class="tab-buttons">
    <button class="tab-btn active" onclick="openTab(event, 'inflacao')">Inflação</button>
    <button class="tab-btn" onclick="openTab(event, 'crescimento')">Crescimento</button>
    <button class="tab-btn" onclick="openTab(event, 'comercio')">Balança Comercial</button>
    <button class="tab-btn" onclick="openTab(event, 'irf')">Impulso-Resposta</button>
    <button class="tab-btn" onclick="openTab(event, 'eua')" style="border-left: 2px solid #1a5276; font-weight: 700;">EUA</button>
    <button class="tab-btn" onclick="openTab(event, 'choques-geo')" style="border-left: 2px solid #c0392b;">Choques Geopolíticos</button>
    <button class="tab-btn" onclick="openTab(event, 'referencias')">Referências</button>
  </div>

  <div id="inflacao" class="tab-content active">
    <div class="chart-container">{''.join(chart_divs["chart-infl-paths"])}</div>
    <p class="source-note">Fontes: BCCh WP 747; Banxico; Banco de la República; BCB; Tandfonline (2025).</p>
  </div>

  <div id="crescimento" class="tab-content">
    <div class="chart-container">{''.join(chart_divs["chart-gdp-paths"])}</div>
    <p class="source-note">Fontes: FLAR (2024); IMF WP/17/15; Fed DSGE (2024); IPEA. Colômbia e Brasil positivos (exportadores).</p>
  </div>

  <div id="comercio" class="tab-content">
    <div class="chart-container">{''.join(chart_divs["chart-trade-paths"])}</div>
    <p class="source-note">Saldo = exportações − importações (incl. efeitos de preço, volume, câmbio e demanda). Fontes: OEC World; IEA; IMF WP/08/216.</p>
    <div class="chart-container">{''.join(chart_divs["chart-trade-waterfall"])}</div>
    <p class="source-note">Saldo acumulado Q0–Q4 como % do PIB anual. Fonte: IMF WEO (2024).</p>
    <div class="chart-container">{''.join(chart_divs["chart-fx-depreciation"])}</div>
    <p class="source-note">Depreciação % ≈ −Δ Conta Petróleo Líquida (acum. Q0–Q4) / PIB. Valores positivos = pressão de depreciação; negativos = pressão de apreciação (exportadores).</p>
    <div class="chart-container">{''.join(chart_divs["chart-oil-import-share"])}</div>
    <p class="source-note">Fontes: OEC World (2024); WITS/UN Comtrade; DANE; PEMEX; ANP; MDIC/Comex Stat.</p>
  </div>

  <div id="irf" class="tab-content">
    <div class="chart-container">{''.join(chart_divs["chart-irf-infl"])}</div>
    <p class="source-note">Bandas: 68% (escura) e 90% (clara), derivadas dos bounds low/mid/high.</p>
    <div class="chart-container">{''.join(chart_divs["chart-irf-gdp"])}</div>
    <p class="source-note">IRFs calibradas via kernel gamma. Formato hump-shaped (VAR/SVAR).</p>
  </div>

  <div id="eua" class="tab-content">
    <div class="chart-container">{''.join(chart_divs["chart-us-panel"])}</div>
    <p class="source-note">Fontes: Presno & Prestipino (FEDS Notes 2024); Kilian & Zhou (Dallas Fed 2023); Gagliardone & Gertler (NBER 2023); Oladosu et al. (2018); IMF WP 2025/145; Kanzig (AER 2021).</p>
  </div>

  <div id="choques-geo" class="tab-content">
    <div style="background: #fdf2f2; border-left: 4px solid #c0392b; padding: 10px 16px; margin-bottom: 15px; border-radius: 0 5px 5px 0; font-size: 13px;">
      <strong>Choques Geopolíticos (2022–2024):</strong> Variação % do Brent e moedas EM em D+0, D+7 e D+14
      após cada evento (vs véspera). Dados de fechamento diário — não varia com o simulador de cenários.
    </div>
    <div class="summary-box" style="margin-bottom: 20px; overflow-x: auto;">
      {geopolitical_table_html}
      <p class="source-note" style="margin-top: 8px;">Variações % relativas à véspera de cada evento. Fontes: oil.xlsx (Brent), fx.xlsx (câmbio). Valores positivos em moedas = depreciação da moeda local vs USD.</p>
    </div>
    <div class="chart-container">{fx_dist_html}</div>
    <p class="source-note">Distribuição across {len(GEOPOLITICAL_EVENTS)} eventos geopolíticos (2022–2024). Barra = intervalo P20–P80; whiskers = min–max.</p>
  </div>

  <div id="referencias" class="tab-content">
    {references_html}
  </div>
</div>

<script>
var sliderPrices = {slider_prices_json};
var baseline = {BASELINE_OIL_PRICE};
var namedDescs = {named_descs_json};
var currentIdx = {DEFAULT_SLIDER_IDX};

function selectByIdx(idx) {{
  if (idx === currentIdx) return;

  var old = document.querySelectorAll('.s-' + currentIdx);
  for (var i = 0; i < old.length; i++) old[i].style.display = 'none';
  var nw = document.querySelectorAll('.s-' + idx);
  for (var i = 0; i < nw.length; i++) nw[i].style.display = 'block';

  currentIdx = idx;

  document.getElementById('price-input').value = sliderPrices[idx];
  document.getElementById('price-slider').value = idx;

  var shock = ((sliderPrices[idx] - baseline) / baseline * 100).toFixed(1);
  var sign = parseFloat(shock) >= 0 ? '+' : '';
  var lbl = document.getElementById('shock-label');
  lbl.textContent = sign + shock + '%';
  lbl.className = 'shock-label ' + (parseFloat(shock) > 0 ? 'shock-positive' : parseFloat(shock) < 0 ? 'shock-negative' : 'shock-zero');

  // Atualizar botões de cenário nomeado
  var btns = document.querySelectorAll('.scenario-btn');
  for (var i = 0; i < btns.length; i++) {{
    btns[i].classList.remove('active');
    if (parseInt(btns[i].getAttribute('data-idx')) === idx) {{
      btns[i].classList.add('active');
    }}
  }}

  // Atualizar descrição
  var desc = namedDescs[String(idx)] || '$' + sliderPrices[idx] + '/bbl';
  document.getElementById('scenario-desc').textContent = desc;

  window.dispatchEvent(new Event('resize'));
}}

function onSliderChange(val) {{
  selectByIdx(parseInt(val));
}}

function onPriceInput(val) {{
  var price = parseInt(val);
  if (isNaN(price)) return;
  price = Math.round(price / 5) * 5;
  price = Math.max({SLIDER_PRICES[0]}, Math.min({SLIDER_PRICES[-1]}, price));
  var idx = (price - {SLIDER_PRICES[0]}) / 5;
  document.getElementById('price-input').value = price;
  selectByIdx(idx);
}}

function jumpToScenario(idx, btn) {{
  selectByIdx(idx);
}}

function openTab(evt, tabName) {{
  var contents = document.getElementsByClassName("tab-content");
  for (var i = 0; i < contents.length; i++) contents[i].classList.remove("active");
  var btns = document.getElementsByClassName("tab-btn");
  for (var i = 0; i < btns.length; i++) btns[i].classList.remove("active");
  document.getElementById(tabName).classList.add("active");
  evt.currentTarget.classList.add("active");
  window.dispatchEvent(new Event('resize'));
}}
</script>

</body>
</html>'''

    return html


# ══════════════════════════════════════════════════════════════════════════════
# 13. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("CHOQUE NO PETROLEO - DASHBOARD DE IMPACTO PARA EMs")
    print("=" * 65)
    print(f"  Preco base: ${BASELINE_OIL_PRICE:.0f}/bbl")
    print(f"  Slider: ${SLIDER_PRICES[0]}-${SLIDER_PRICES[-1]}/bbl (step $5, {len(SLIDER_PRICES)} cenarios)")
    print(f"  Cenarios nomeados: {[s['name'] + ' ($' + str(s['target']) + ')' for s in SCENARIOS]}")
    print(f"  Paises: {', '.join(COUNTRIES.keys())}")
    print(f"  Horizonte: {HORIZON_QUARTERS} trimestres")
    print()

    print("Gerando cenários e gráficos...")
    html = build_html_dashboard()

    out_path = os.path.join(OUTPUT_DIR, "oil_shock_dashboard.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"\nDashboard salvo em: {out_path} ({size_kb:.0f} KB)")
    print("Abra o arquivo no navegador para visualizar.")


if __name__ == "__main__":
    main()
