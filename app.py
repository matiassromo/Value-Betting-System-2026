"""
app.py — Sistema de Value Betting para el Mundial 2026
Ejecutar con: streamlit run app.py

MODO EN VIVO : Ingresa tu API key gratuita de football-data.org en la barra lateral.
MODO DEMO    : Sin API key, carga partidos de demostración para Jun 26-30 de 2026.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
from typing import Dict, List, Optional

from analyzer import analyze_match, KELLY_DEFAULT, MIN_VALUE_PCT
from data import get_team, DEMO_MATCHES
from fetcher import fetch_wc_matches

# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mundial 2026 | Value Betting",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color: #0a0d14; }
  section[data-testid="stSidebar"] { background-color: #0f1320; }

  .card-hdr-val {
    background: linear-gradient(90deg,#064e3b,#065f46);
    border-radius:10px 10px 0 0; padding:12px 20px;
    display:flex; justify-content:space-between; align-items:center;
  }
  .card-hdr-noval {
    background: linear-gradient(90deg,#1c1f2b,#1a1d28);
    border-radius:10px 10px 0 0; padding:12px 20px;
    display:flex; justify-content:space-between; align-items:center;
  }
  .badge-val   { background:linear-gradient(90deg,#10b981,#059669); color:#fff;
                 font-weight:700; border-radius:6px; padding:3px 12px; font-size:.88em; }
  .badge-noval { background:#374151; color:#9ca3af;
                 border-radius:6px; padding:3px 12px; font-size:.88em; }
  .badge-live  { background:linear-gradient(90deg,#7c3aed,#4f46e5); color:#fff;
                 border-radius:6px; padding:2px 9px; font-size:.78em; }
  .badge-demo  { background:#374151; color:#9ca3af;
                 border-radius:6px; padding:2px 9px; font-size:.78em; }
  .badge-crit  { background:#7f1d1d; color:#fca5a5;
                 border-radius:6px; padding:2px 9px; font-size:.78em; }
  .badge-high  { background:#78350f; color:#fcd34d;
                 border-radius:6px; padding:2px 9px; font-size:.78em; }
  .badge-med   { background:#1e3a5f; color:#93c5fd;
                 border-radius:6px; padding:2px 9px; font-size:.78em; }

  .stake-box {
    background:linear-gradient(135deg,#0c2340,#0a1929);
    border:1px solid #1d4ed8; border-radius:10px;
    padding:14px; text-align:center;
  }
  .alert-box {
    background:linear-gradient(90deg,#1c1a10,#1a1a10);
    border-left:4px solid #f59e0b; border-radius:0 8px 8px 0;
    padding:10px 16px; color:#d1d5db; font-size:.84em; line-height:1.6;
  }
  .no-bet-box {
    background:#1c0a0a; border:1px solid #7f1d1d;
    border-radius:10px; padding:14px 20px; color:#fca5a5; font-size:.9em;
  }
  .odds-panel {
    background:#111827; border:1px solid #1e2a40;
    border-radius:10px; padding:16px 20px; margin:8px 0;
  }
  .team-vs { font-size:1.45em; font-weight:800; color:#f9fafb; }
  .team-sep { color:#4b5563; margin:0 10px; }

  div[data-testid="metric-container"] {
    background-color:#141827; border:1px solid #1e2a40;
    border-radius:10px; padding:12px 16px;
  }
  div[data-testid="metric-container"] label { color:#6b7280 !important; font-size:.78em !important; }
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color:#e5e7eb !important; font-size:1.3em !important; font-weight:700;
  }
  hr { border-color:#1e2a40 !important; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

IMPORTANCE_META = {
    "critical": ("🔴 CRÍTICO", "badge-crit"),
    "high":     ("🟠 ALTO",   "badge-high"),
    "medium":   ("🟡 MEDIO",  "badge-med"),
    "normal":   ("⚪ NORMAL", "badge-med"),
    "low":      ("🟢 BAJO",   "badge-med"),
}
MARKET_KEYS = ["1", "X", "2", "over_2_5", "under_2_5", "over_1_5", "under_1_5", "btts_yes", "btts_no"]
MARKET_LABELS = {
    "1": "Victoria Local (1)", "X": "Empate (X)", "2": "Victoria Visitante (2)",
    "over_2_5": "Over 2.5 G", "under_2_5": "Under 2.5 G",
    "over_1_5": "Over 1.5 G", "under_1_5": "Under 1.5 G",
    "btts_yes": "Ambos Anotan: SÍ", "btts_no": "Ambos Anotan: NO",
}
MARKET_DEFAULTS = {
    "1": 2.50, "X": 3.20, "2": 2.80,
    "over_2_5": 1.90, "under_2_5": 1.90,
    "over_1_5": 1.40, "under_1_5": 2.80,
    "btts_yes": 1.90, "btts_no": 1.90,
}


def _odds_key(match_id: str, market: str) -> str:
    return f"odds_{match_id}_{market}"


def _get_odds(match_id: str) -> Dict[str, float]:
    return {k: st.session_state.get(_odds_key(match_id, k), MARKET_DEFAULTS[k]) for k in MARKET_KEYS}


def _odds_panel(match_id: str, t1_name: str, t2_name: str) -> Dict[str, float]:
    """Renderiza inputs de cuotas y devuelve los valores actuales."""
    labels_custom = {
        "1": f"1 — {t1_name[:12]} gana",
        "X": "X — Empate",
        "2": f"2 — {t2_name[:12]} gana",
        "over_2_5": "Over 2.5 Goles",
        "under_2_5": "Under 2.5 Goles",
        "over_1_5": "Over 1.5 Goles",
        "under_1_5": "Under 1.5 Goles",
        "btts_yes": "Ambos Anotan: SÍ",
        "btts_no": "Ambos Anotan: NO",
    }
    st.markdown('<div class="odds-panel">', unsafe_allow_html=True)
    st.markdown("**Ingresa las cuotas de tu casa de apuestas:**")
    cols_a = st.columns(3)
    cols_b = st.columns(3)
    cols_c = st.columns(3)
    grid = [
        (cols_a, ["1", "X", "2"]),
        (cols_b, ["over_2_5", "under_2_5", "over_1_5"]),
        (cols_c, ["under_1_5", "btts_yes", "btts_no"]),
    ]
    for cols, keys in grid:
        for col, k in zip(cols, keys):
            with col:
                st.number_input(
                    labels_custom[k],
                    min_value=1.01, max_value=1000.0,
                    value=float(st.session_state.get(_odds_key(match_id, k), MARKET_DEFAULTS[k])),
                    step=0.01, format="%.2f",
                    key=_odds_key(match_id, k),
                )
    st.markdown("</div>", unsafe_allow_html=True)
    return _get_odds(match_id)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚽ Mundial 2026")
    st.markdown("### Sistema de Value Betting")
    st.markdown("---")

    selected_date = st.date_input(
        "📅 Fecha de la Jornada",
        value=date(2026, 6, 26),
        min_value=date(2026, 6, 11),
        max_value=date(2026, 7, 19),
    )

    st.markdown("---")
    bankroll = st.number_input(
        "💰 Banca Actual (USD)",
        min_value=1.0, max_value=1_000_000.0,
        value=47.46, step=0.01, format="%.2f",
    )

    st.markdown("---")
    kelly_pct = st.slider("🎯 Kelly Fraccionado (%)", 10, 50, 25, 5)
    kelly_fraction = kelly_pct / 100.0

    min_edge = st.slider("📈 Edge Mínimo (%)", 3, 15, 5, 1)

    st.markdown("---")
    st.markdown("**🌐 Datos en Vivo (Opcional)**")
    api_key = st.text_input(
        "API Key — football-data.org",
        type="password",
        placeholder="Pega tu API key aquí",
        help=(
            "Regístrate GRATIS en football-data.org → Obtén tu token → "
            "Pégalo aquí para cargar los partidos reales del día seleccionado."
        ),
    )
    if not api_key:
        st.caption("Sin API key → se usan datos de demostración para las fechas Jun 26-30.")

    st.markdown("---")
    st.markdown("**Metodología**")
    st.markdown("- Poisson + Dixon-Coles (ρ=−0.13)\n- xG histórico + factor táctico\n- Kelly Fraccionado")
    st.caption("⚠️ Solo para análisis cuantitativo. Las apuestas conllevan riesgo de pérdida.")


# ──────────────────────────────────────────────────────────────────────────────
# Cargar partidos
# ──────────────────────────────────────────────────────────────────────────────
date_str = selected_date.strftime("%Y-%m-%d")
api_error: Optional[str] = None
live_mode = False

if api_key:
    with st.spinner("🔄 Conectando con football-data.org..."):
        result = fetch_wc_matches(date_str, api_key)
    api_error = result.get("error")
    raw_matches = result.get("matches", [])
    if not api_error and raw_matches:
        live_mode = True
        day_matches = raw_matches
    elif not api_error and not raw_matches:
        # Día válido pero sin partidos según la API
        day_matches = []
    else:
        # Hubo error → fallback a demo
        day_matches = [m for m in DEMO_MATCHES if m["date"] == date_str]
else:
    day_matches = [m for m in DEMO_MATCHES if m["date"] == date_str]

# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("# 🏆 Value Betting System — FIFA World Cup 2026")
mode_badge = (
    '<span style="background:#4f46e5;color:#fff;border-radius:6px;padding:2px 10px;'
    'font-size:.82em">⚡ DATOS EN VIVO</span>'
    if live_mode else
    '<span style="background:#374151;color:#9ca3af;border-radius:6px;padding:2px 10px;'
    'font-size:.82em">🗂️ MODO DEMO</span>'
)
st.markdown(
    f"**Motor:** Poisson (Dixon-Coles) · **Gestión:** Kelly {kelly_pct}% · "
    f"**Edge mínimo:** {min_edge}% &nbsp;&nbsp; {mode_badge}",
    unsafe_allow_html=True,
)

if api_error:
    st.warning(f"⚠️ API: {api_error} — Mostrando datos de demostración.")

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# Sin partidos
# ──────────────────────────────────────────────────────────────────────────────
if not day_matches:
    if live_mode:
        st.info(
            f"La API de football-data.org no reporta partidos del Mundial 2026 "
            f"para el **{selected_date.strftime('%d/%m/%Y')}**. "
            f"Puede ser un día de descanso o los partidos aún no están cargados."
        )
    else:
        st.info(
            f"No hay partidos de demostración para el **{selected_date.strftime('%d/%m/%Y')}**. "
            f"Prueba con fechas entre el 26 y 30 de junio de 2026, "
            f"o ingresa tu API key para datos reales de cualquier fecha."
        )
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Expander de instrucciones para cuotas
# ──────────────────────────────────────────────────────────────────────────────
with st.expander("ℹ️ Cómo usar los inputs de cuotas", expanded=False):
    st.markdown("""
    Cada partido tiene **9 campos de cuota** (formato decimal, ej: 1.85):

    | Campo | Significado |
    |---|---|
    | **1** | Cuota para que gane el equipo local |
    | **X** | Cuota para el empate |
    | **2** | Cuota para que gane el equipo visitante |
    | **Over/Under 2.5** | Cuota para más/menos de 2.5 goles totales |
    | **Over/Under 1.5** | Cuota para más/menos de 1.5 goles totales |
    | **Ambos Anotan SÍ/NO** | Cuota para que ambos equipos marquen |

    - Ingresa las cuotas de **tu casa de apuestas** (Bet365, Codere, Betway, etc.)
    - El sistema calculará automáticamente qué mercados tienen **valor matemático**
    - Si no cambias los valores, se usarán cuotas genéricas de referencia
    """)

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# Análisis de todos los partidos (con odds de session_state)
# ──────────────────────────────────────────────────────────────────────────────

# Primero construimos todos los análisis con las odds actuales en session_state
analyses = []
for m in day_matches:
    odds = _get_odds(m["id"])
    match_with_odds = {**m, "bookmaker_odds": odds}
    try:
        a = analyze_match(match_with_odds, min_edge=min_edge, kelly_fraction=kelly_fraction)
        analyses.append((m, a))
    except Exception as exc:
        st.error(f"Error analizando {m['id']}: {exc}")

analyses.sort(key=lambda x: x[1].max_edge, reverse=True)

# ──────────────────────────────────────────────────────────────────────────────
# Dashboard de métricas globales
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(f"## 📊 Dashboard — {selected_date.strftime('%d de %B de %Y')}")
total_m = len(analyses)
val_m   = sum(1 for _, a in analyses if a.recommended)
edges   = [a.max_edge for _, a in analyses if a.max_edge > 0]
avg_edge = np.mean(edges) if edges else 0.0
avg_xg  = np.mean([a.expected_total for _, a in analyses]) if analyses else 0.0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Partidos del Día",   str(total_m))
c2.metric("Con Valor",          f"{val_m} / {total_m}")
c3.metric("Edge Máx. Medio",    f"{avg_edge:.1f}%")
c4.metric("xG Total Medio",     f"{avg_xg:.2f}")
c5.metric("Banca",              f"${bankroll:.2f}")

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# Tabla resumen
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 📋 Pronósticos Ordenados por % de Valor")

rows = []
for match, a in analyses:
    best = a.best
    rows.append({
        "Partido":        f"{a.team1} vs {a.team2}",
        "Gr/MD":          f"{match['group']}·MD{match['matchday']}",
        "Max Edge":       f"{a.max_edge:+.1f}%",
        "Mercado Rec.":   best.label if best else "—",
        "Cuota Casa":     f"{best.bk_odds:.2f}"  if best else "—",
        "Cuota Justa":    f"{best.fair_odds:.3f}" if best else "—",
        "Stake":          f"{best.stake_pct:.1f}% = ${bankroll*best.kelly_frac:.2f}" if best else "—",
        "Confianza":      f"{best.confidence}/100" if best else "—",
        "Decisión":       "✅ APOSTAR" if best else "🚫 NO APOSTAR",
    })

df = pd.DataFrame(rows)

def _row_style(row: pd.Series):
    if "APOSTAR" in row["Decisión"] and "NO" not in row["Decisión"]:
        return ["background-color:#052e16;color:#86efac"] * len(row)
    return ["background-color:#1a1a1a;color:#6b7280"] * len(row)

st.dataframe(
    df.style.apply(_row_style, axis=1),
    use_container_width=True, hide_index=True,
    height=min(55 + len(rows) * 38, 420),
)

st.markdown("---")
st.markdown("### 🔍 Análisis Detallado por Partido")

# ──────────────────────────────────────────────────────────────────────────────
# Tarjetas de partido
# ──────────────────────────────────────────────────────────────────────────────
for match, a in analyses:
    t1_data  = get_team(match["team1"], match.get("team1_name_raw",""))
    t2_data  = get_team(match["team2"], match.get("team2_name_raw",""))
    best     = a.best
    imp_lbl, imp_cls = IMPORTANCE_META.get(match.get("importance","normal"), ("⚪","badge-med"))
    src_badge = '<span class="badge-live">⚡ EN VIVO</span>' if match.get("live") else '<span class="badge-demo">📁 DEMO</span>'

    hdr_cls = "card-hdr-val" if a.recommended else "card-hdr-noval"
    val_badge = (f'<span class="badge-val">💎 VALOR +{a.max_edge:.1f}%</span>'
                 if a.recommended else '<span class="badge-noval">🚫 SIN VALOR</span>')

    # Marcador si el partido ya terminó (modo API)
    score_txt = ""
    if match.get("score_str"):
        score_txt = f" · **{match['score_str']}**"
    status_txt = match.get("status", "")
    if status_txt in ("FINISHED",):
        score_txt += " (FT)"
    elif status_txt in ("IN_PLAY", "PAUSED"):
        score_txt += " 🔴 EN JUEGO"

    st.markdown(f"""
    <div class="{hdr_cls}">
      <span class="team-vs">
        {a.team1}<span class="team-sep">vs</span>{a.team2}
        <span style="font-size:.75em;color:#9ca3af;font-weight:400">{score_txt}</span>
      </span>
      <span>
        {src_badge}&nbsp;
        <span class="{imp_cls}">{imp_lbl}</span>&nbsp;&nbsp;
        {val_badge}
      </span>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.caption(
            f"🏟️ {match.get('venue') or '—'}  ·  "
            f"📅 {match['date']}  ·  "
            f"Grupo {match['group']} · Jornada {match['matchday']}"
        )
        if match.get("context"):
            st.caption(f"📋 {match['context']}")

        st.markdown("---")

        # ── Input de cuotas ────────────────────────────────────────────────────
        with st.expander("✏️ Introducir / Editar Cuotas de Casa de Apuestas", expanded=not a.recommended):
            new_odds = _odds_panel(match["id"], a.team1, a.team2)
            # Recalcular si el usuario cambió las cuotas
            match_recalc = {**match, "bookmaker_odds": new_odds}
            try:
                a = analyze_match(match_recalc, min_edge=min_edge, kelly_fraction=kelly_fraction)
                best = a.best
            except Exception:
                pass

        # ── Probabilidades ─────────────────────────────────────────────────────
        st.markdown("**📊 Probabilidades del Modelo (Poisson + Dixon-Coles)**")
        p1c, p2c, p3c, p4c, p5c, p6c = st.columns(6)
        p1c.metric(f"1 — {a.team1[:8]}", f"{a.p1*100:.1f}%")
        p2c.metric("X — Empate",          f"{a.px*100:.1f}%")
        p3c.metric(f"2 — {a.team2[:8]}", f"{a.p2*100:.1f}%")
        p4c.metric("Over 2.5",            f"{a.p_over25*100:.1f}%")
        p5c.metric("Over 1.5",            f"{a.p_over15*100:.1f}%")
        p6c.metric("Ambos Anotan",        f"{a.p_btts*100:.1f}%")
        lc1, lc2 = st.columns(2)
        lc1.caption(f"λ {a.team1}: **{a.lam1:.3f}** goles esperados · estado: `{t1_data.get('wc_status','normal')}`")
        lc2.caption(f"λ {a.team2}: **{a.lam2:.3f}** goles esperados · estado: `{t2_data.get('wc_status','normal')}`")
        st.caption(f"xG total proyectado: **{a.expected_total:.3f}** goles combinados")

        st.markdown("---")

        # ── Tabla de mercados ──────────────────────────────────────────────────
        st.markdown("**💰 Análisis de Mercados — Cuota Casa vs Cuota Justa**")
        mkt_rows = [{
            "Mercado":       m.label,
            "Prob. Modelo":  f"{m.prob*100:.2f}%",
            "Cuota Justa":   f"{m.fair_odds:.4f}",
            "Cuota Casa":    f"{m.bk_odds:.4f}",
            "Edge":          f"{m.edge_pct:+.2f}%",
            "¿Valor?":       "💎 SÍ" if m.is_value else "❌ NO",
            "Confianza":     f"{m.confidence}/100",
        } for m in a.markets]

        def _mkt_style(row: pd.Series):
            if "💎" in row["¿Valor?"]:
                return ["background-color:#052e16;color:#86efac"] * len(row)
            try:
                if float(row["Edge"].replace("%","").replace("+","")) < -8:
                    return ["background-color:#450a0a;color:#fca5a5"] * len(row)
            except ValueError:
                pass
            return ["background-color:#111827;color:#9ca3af"] * len(row)

        st.dataframe(
            pd.DataFrame(mkt_rows).style.apply(_mkt_style, axis=1),
            use_container_width=True, hide_index=True,
            height=min(45 + len(mkt_rows) * 38, 380),
        )

        # ── Pronóstico + stake ─────────────────────────────────────────────────
        st.markdown("---")
        if a.recommended and best:
            st.markdown("**✅ PRONÓSTICO DE VALOR**")
            sc1, sc2, sc3 = st.columns(3)
            stake_usd = bankroll * best.kelly_frac
            potential = stake_usd * best.bk_odds
            profit    = potential - stake_usd
            conf_col  = "#34d399" if best.confidence >= 65 else "#f59e0b"

            with sc1:
                st.markdown(f"""
                <div class="stake-box">
                  <div style="color:#9ca3af;font-size:.75em;text-transform:uppercase;
                              letter-spacing:.06em">Mercado</div>
                  <div style="font-size:1.05em;font-weight:700;color:#34d399;margin:6px 0">
                    {best.label}</div>
                  <div style="color:#9ca3af;font-size:.82em">
                    Cuota: <strong style="color:#e5e7eb">{best.bk_odds:.2f}</strong>
                    &nbsp;·&nbsp;
                    Justa: <strong style="color:#e5e7eb">{best.fair_odds:.3f}</strong>
                  </div>
                </div>""", unsafe_allow_html=True)
            with sc2:
                st.markdown(f"""
                <div class="stake-box">
                  <div style="color:#9ca3af;font-size:.75em;text-transform:uppercase;
                              letter-spacing:.06em">Stake — Kelly {kelly_pct}%</div>
                  <div style="font-size:1.8em;font-weight:800;color:#34d399">
                    {best.stake_pct:.2f}%</div>
                  <div style="color:#9ca3af;font-size:.82em">de la banca</div>
                  <div style="font-size:1.2em;font-weight:600;color:#60a5fa">
                    = ${stake_usd:.2f} USD</div>
                </div>""", unsafe_allow_html=True)
            with sc3:
                st.markdown(f"""
                <div class="stake-box">
                  <div style="color:#9ca3af;font-size:.75em;text-transform:uppercase;
                              letter-spacing:.06em">Confianza del Modelo</div>
                  <div style="font-size:2em;font-weight:800;color:{conf_col}">
                    {best.confidence}/100</div>
                  <div style="color:#9ca3af;font-size:.82em">
                    Edge: <strong style="color:#e5e7eb">+{best.edge_pct:.2f}%</strong>
                    &nbsp;·&nbsp;
                    Prob: <strong style="color:#e5e7eb">{best.prob*100:.1f}%</strong>
                  </div>
                </div>""", unsafe_allow_html=True)

            st.success(
                f"💵 **Retorno potencial:** ${potential:.2f}  "
                f"(+${profit:.2f} ganancia)  ·  "
                f"Kelly completo (sin fraccionar): {best.kelly_full*100:.2f}%"
            )
            if len(a.all_value) > 1:
                with st.expander(f"📌 {len(a.all_value)-1} mercado(s) adicional(es) con valor"):
                    for ex in a.all_value[1:]:
                        ex_usd = bankroll * ex.kelly_frac
                        st.markdown(
                            f"- **{ex.label}** · Cuota {ex.bk_odds:.2f} "
                            f"· Edge +{ex.edge_pct:.1f}% "
                            f"· Stake {ex.stake_pct:.2f}% = ${ex_usd:.2f}"
                        )
        else:
            st.markdown(f"""
            <div class="no-bet-box">
              🚫 <strong>NO APOSTAR — Mercado Eficiente</strong><br>
              El modelo no detecta ventaja matemática ≥ {min_edge}% en ningún mercado
              disponible. Las cuotas de la casa reflejan la probabilidad real.
              Apostar aquí equivale a operar con esperanza matemática negativa.
            </div>""", unsafe_allow_html=True)

        # ── Alerta táctica ─────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("**⚠️ Alerta Táctica del Motor**")
        st.markdown(f'<div class="alert-box">📡 {a.tactical_alert}</div>', unsafe_allow_html=True)

        nc1, nc2 = st.columns(2)
        with nc1:
            inj1 = t1_data.get("key_injuries",[])
            st.caption(
                f"**{a.team1}:** {t1_data.get('tactical_note','—')} "
                + (f"· ⛔ {', '.join(inj1)}" if inj1 else "· Sin bajas registradas") + "  \n"
                + f"xG ataque: {t1_data.get('xg_for','?')} · xG def. concedida: {t1_data.get('xg_against','?')}"
            )
        with nc2:
            inj2 = t2_data.get("key_injuries",[])
            st.caption(
                f"**{a.team2}:** {t2_data.get('tactical_note','—')} "
                + (f"· ⛔ {', '.join(inj2)}" if inj2 else "· Sin bajas registradas") + "  \n"
                + f"xG ataque: {t2_data.get('xg_for','?')} · xG def. concedida: {t2_data.get('xg_against','?')}"
            )

        st.markdown(
            "<hr style='border:0;border-top:2px solid #1e2a40;margin:20px 0 8px 0'>",
            unsafe_allow_html=True,
        )

# ──────────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style="text-align:center;color:#374151;font-size:.78em;padding:16px 0 8px 0">
  <strong style="color:#4b5563">Motor:</strong>
  Poisson · Dixon-Coles (ρ=−0.13) · Factor táctico de necesidad clasificatoria · xG histórico
  &nbsp;|&nbsp;
  <strong style="color:#4b5563">Gestión:</strong> Kelly Fraccionado {kelly_pct}% · Tope 15% banca
  &nbsp;|&nbsp;
  <strong style="color:#4b5563">Umbral:</strong> Edge ≥ {min_edge}%
  &nbsp;|&nbsp;
  <strong style="color:#4b5563">Datos:</strong>
  {'football-data.org (EN VIVO)' if live_mode else 'Modo Demo (26-30 Jun 2026)'}
  <br><br>
  ⚠️ Este sistema es una herramienta de análisis cuantitativo. No garantiza rentabilidad.
  Las apuestas deportivas implican riesgo real de pérdida de capital.
</div>
""", unsafe_allow_html=True)
