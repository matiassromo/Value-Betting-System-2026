"""
analyzer.py — Motor Cuantitativo de Value Betting
Distribución de Poisson (Dixon-Coles) + Criterio de Kelly Fraccionado
"""

from __future__ import annotations
import numpy as np
from scipy.stats import poisson
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from data import TEAMS, TACTICAL_FACTORS, WC_AVG_XG, get_team

MAX_GOALS = 9          # máximo de goles en la matriz
KELLY_DEFAULT = 0.25   # Kelly al 25% por defecto
MIN_VALUE_PCT = 5.0    # edge mínimo para considerar value bet
MAX_STAKE_PCT = 15.0   # tope de stake (% de banca)
DC_RHO = -0.13         # corrección Dixon-Coles (correlación negativa baja-puntuación)


# ─── Estructuras de datos ──────────────────────────────────────────────────────

@dataclass
class MarketResult:
    key: str
    label: str
    prob: float
    fair_odds: float
    bk_odds: float
    edge_pct: float
    is_value: bool
    kelly_full: float
    kelly_frac: float
    stake_pct: float
    confidence: int


@dataclass
class MatchAnalysis:
    match_id: str
    team1: str
    team2: str
    lam1: float
    lam2: float
    expected_total: float
    p1: float
    px: float
    p2: float
    p_over25: float
    p_under25: float
    p_over15: float
    p_under15: float
    p_btts: float
    p_no_btts: float
    markets: List[MarketResult]
    best: Optional[MarketResult]
    all_value: List[MarketResult]
    max_edge: float
    tactical_alert: str
    recommended: bool


# ─── Núcleo probabilístico ────────────────────────────────────────────────────

def _dc_tau(i: int, j: int, l1: float, l2: float) -> float:
    """Factor de corrección Dixon-Coles para marcadores bajos."""
    if i == 0 and j == 0:
        return 1 - l1 * l2 * DC_RHO
    if i == 1 and j == 0:
        return 1 + l2 * DC_RHO
    if i == 0 and j == 1:
        return 1 + l1 * DC_RHO
    if i == 1 and j == 1:
        return 1 - DC_RHO
    return 1.0


def compute_lambdas(t1: Dict, t2: Dict) -> Tuple[float, float]:
    """
    Calcula lambda1 y lambda2 usando el modelo Dixon-Coles xG.

    lam1 = (xg_for_1 / avg) * (xg_against_2 / avg) * avg
           * tactical_attack_1 * tactical_defense_vuln_2
    """
    avg = WC_AVG_XG

    # Fuerzas relativas al promedio del torneo
    att1 = t1["xg_for"] / avg
    def2 = t2["xg_against"] / avg
    att2 = t2["xg_for"] / avg
    def1 = t1["xg_against"] / avg

    base1 = att1 * def2 * avg
    base2 = att2 * def1 * avg

    tac1 = TACTICAL_FACTORS.get(t1["wc_status"], TACTICAL_FACTORS["normal"])
    tac2 = TACTICAL_FACTORS.get(t2["wc_status"], TACTICAL_FACTORS["normal"])

    # attack_adj del propio equipo × defense_adj (vulnerabilidad) del rival
    l1 = base1 * tac1["attack_adj"] * tac2["defense_adj"]
    l2 = base2 * tac2["attack_adj"] * tac1["defense_adj"]

    # Penalización por lesiones clave (−5% por ausencia significativa, máx 2)
    pen1 = 0.05 * min(len(t1.get("key_injuries", [])), 2)
    pen2 = 0.05 * min(len(t2.get("key_injuries", [])), 2)
    l1 *= (1 - pen1)
    l2 *= (1 - pen2)

    return round(max(l1, 0.05), 4), round(max(l2, 0.05), 4)


def build_score_matrix(l1: float, l2: float) -> np.ndarray:
    """Construye matriz de probabilidades de marcadores con corrección Dixon-Coles."""
    mat = np.zeros((MAX_GOALS + 1, MAX_GOALS + 1))
    for i in range(MAX_GOALS + 1):
        for j in range(MAX_GOALS + 1):
            p = poisson.pmf(i, l1) * poisson.pmf(j, l2) * _dc_tau(i, j, l1, l2)
            mat[i, j] = max(p, 0.0)
    mat /= mat.sum()  # normalizar por la corrección DC
    return mat


def extract_market_probs(mat: np.ndarray) -> Dict[str, float]:
    """Extrae probabilidades de todos los mercados desde la matriz."""
    p1   = float(np.sum(np.tril(mat, -1)))   # team1 gana
    px   = float(np.trace(mat))              # empate
    p2   = float(np.sum(np.triu(mat, 1)))    # team2 gana

    over25 = under25 = over15 = under15 = btts = 0.0
    for i in range(MAX_GOALS + 1):
        for j in range(MAX_GOALS + 1):
            total = i + j
            v = mat[i, j]
            if total > 2.5:
                over25 += v
            if total > 1.5:
                over15 += v
            if i > 0 and j > 0:
                btts += v

    return {
        "1": p1, "X": px, "2": p2,
        "over_2_5": over25, "under_2_5": 1 - over25,
        "over_1_5": over15, "under_1_5": 1 - over15,
        "btts_yes": btts,   "btts_no": 1 - btts,
    }


# ─── Value & Kelly ────────────────────────────────────────────────────────────

MARKET_LABELS = {
    "1":         "Victoria Equipo 1",
    "X":         "Empate",
    "2":         "Victoria Equipo 2",
    "over_2_5":  "Over 2.5 Goles",
    "under_2_5": "Under 2.5 Goles",
    "over_1_5":  "Over 1.5 Goles",
    "under_1_5": "Under 1.5 Goles",
    "btts_yes":  "Ambos Anotan: SÍ",
    "btts_no":   "Ambos Anotan: NO",
}


def kelly(prob: float, odds: float, fraction: float = KELLY_DEFAULT) -> Tuple[float, float]:
    """Retorna (kelly_full, kelly_frac) como fracciones de banca."""
    b = odds - 1.0
    if b <= 0 or prob <= 0 or prob >= 1:
        return 0.0, 0.0
    kf = (b * prob - (1 - prob)) / b
    kf = max(kf, 0.0)
    kfrac = min(kf * fraction, MAX_STAKE_PCT / 100)
    return round(kf, 5), round(kfrac, 5)


def evaluate_markets(
    probs: Dict[str, float],
    bk_odds: Dict[str, float],
    min_edge: float = MIN_VALUE_PCT,
    kelly_frac: float = KELLY_DEFAULT,
) -> List[MarketResult]:
    results = []
    for key, label in MARKET_LABELS.items():
        if key not in bk_odds:
            continue
        prob = probs.get(key, 0.0)
        if prob < 0.02:
            continue
        bo = bk_odds[key]
        fair = 1.0 / prob
        edge = (bo / fair - 1.0) * 100.0
        is_val = edge >= min_edge
        kf, kfrac = kelly(prob, bo, kelly_frac) if is_val else (0.0, 0.0)
        stake = kfrac * 100

        # Confidence: ponderado por edge y nivel de probabilidad base
        if is_val:
            conf = min(100, int(45 + edge * 2.8 + prob * 25))
        else:
            conf = max(5, int(45 - abs(edge) * 1.5))

        results.append(MarketResult(
            key=key, label=label,
            prob=round(prob, 5),
            fair_odds=round(fair, 4),
            bk_odds=round(bo, 4),
            edge_pct=round(edge, 2),
            is_value=is_val,
            kelly_full=kf,
            kelly_frac=kfrac,
            stake_pct=round(stake, 3),
            confidence=conf,
        ))

    return sorted(results, key=lambda m: m.edge_pct, reverse=True)


# ─── Alerta táctica ───────────────────────────────────────────────────────────

def build_alert(t1: Dict, t2: Dict, l1: float, l2: float, markets: List[MarketResult]) -> str:
    parts = []

    # Contexto clasificatorio
    status_msgs = {
        "qualified":    ("{name} ya clasificado — el modelo proyecta rotación y "
                         "reducción de intensidad; λ atacante ajustado −13%"),
        "needs_win":    ("{name} en modo urgencia total — apertura ofensiva forzada "
                         "(+22% λ atacante) aumenta la vulnerabilidad defensiva"),
        "needs_result": ("{name} necesita resultado — presión moderada sin apertura "
                         "excesiva; perfil de equipo que defenderá el empate"),
        "eliminated":   ("{name} sin presión clasificatoria — motivación reducida, "
                         "rendimiento impredecible"),
    }
    for t in (t1, t2):
        msg = status_msgs.get(t["wc_status"], "")
        if msg:
            parts.append(msg.format(name=t["name"]))

    # Bajas clave
    for t in (t1, t2):
        if t.get("key_injuries"):
            parts.append(
                f"Baja(s) en {t['name']}: {', '.join(t['key_injuries'])} "
                f"— penalización de −5% por ausencia aplicada al modelo"
            )

    # xG total
    total = l1 + l2
    if total > 2.8:
        parts.append(
            f"xG combinado proyectado: {total:.2f} goles "
            f"— threshold Over 2.5 superado con base cuantitativa sólida"
        )
    elif total < 1.9:
        parts.append(
            f"xG combinado proyectado: {total:.2f} goles "
            f"— partido cerrado; Under y BTTS-NO favorecidos estadísticamente"
        )
    else:
        parts.append(f"xG combinado proyectado: {total:.2f} goles — mercado equilibrado en goles")

    # Mejor value encontrado
    val_markets = [m for m in markets if m.is_value]
    if val_markets:
        best = val_markets[0]
        parts.append(
            f"Anomalía detectada: cuota casa {best.bk_odds} en '{best.label}' "
            f"vs cuota justa {best.fair_odds:.3f} → ventaja matemática de +{best.edge_pct:.1f}%"
        )
    else:
        parts.append(
            "Mercado eficiente detectado — las cuotas de la casa reflejan "
            "la probabilidad real en todos los mercados disponibles"
        )

    return " | ".join(parts)


# ─── Función principal ────────────────────────────────────────────────────────

def analyze_match(
    match: Dict,
    min_edge: float = MIN_VALUE_PCT,
    kelly_fraction: float = KELLY_DEFAULT,
) -> MatchAnalysis:
    t1 = get_team(match["team1"], match.get("team1_name_raw", ""))
    t2 = get_team(match["team2"], match.get("team2_name_raw", ""))

    l1, l2 = compute_lambdas(t1, t2)
    mat = build_score_matrix(l1, l2)
    probs = extract_market_probs(mat)
    markets = evaluate_markets(probs, match["bookmaker_odds"], min_edge, kelly_fraction)

    val = [m for m in markets if m.is_value]
    best = val[0] if val else None
    max_edge = markets[0].edge_pct if markets else 0.0
    alert = build_alert(t1, t2, l1, l2, markets)

    return MatchAnalysis(
        match_id=match["id"],
        team1=t1["name"], team2=t2["name"],
        lam1=l1, lam2=l2,
        expected_total=round(l1 + l2, 3),
        p1=probs["1"], px=probs["X"], p2=probs["2"],
        p_over25=probs["over_2_5"], p_under25=probs["under_2_5"],
        p_over15=probs["over_1_5"], p_under15=probs["under_1_5"],
        p_btts=probs["btts_yes"],   p_no_btts=probs["btts_no"],
        markets=markets,
        best=best,
        all_value=val,
        max_edge=max_edge,
        tactical_alert=alert,
        recommended=best is not None,
    )
