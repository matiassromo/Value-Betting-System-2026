"""
fetcher.py — Cliente en vivo para el Mundial 2026
API: football-data.org (plan gratuito)

Los equipos se identifican por su TLA (Three Letter Abbreviation) que devuelve
la API. Donde el TLA oficial difiere del código interno del sistema, se mapea.
"""

from __future__ import annotations
import requests
import streamlit as st
from typing import Dict, List, Optional

FD_BASE = "https://api.football-data.org/v4"
WC_CODE = "WC"
REQUEST_TIMEOUT = 12

# ─── TLA oficial API → código interno del sistema ────────────────────────────
# Solo se listan los que difieren; el resto usa el TLA directamente.
TLA_TO_CODE: Dict[str, str] = {
    "GER": "ALE",   # Germany → Alemania
    "ENG": "ING",   # England → Inglaterra
    "JPN": "JAP",   # Japan → Japón
    "KOR": "COR",   # South Korea → Corea del Sur
    "NED": "PB",    # Netherlands → Países Bajos
    "SCO": "ESC",   # Scotland → Escocia
    "CZE": "CHE",   # Czechia → República Checa
}

# ─── Nombre oficial API → código interno (fallback por nombre) ───────────────
NAME_TO_CODE: Dict[str, str] = {
    # CONMEBOL
    "Brazil": "BRA", "Argentina": "ARG", "Colombia": "COL",
    "Ecuador": "ECU", "Uruguay": "URU", "Chile": "CHI",
    "Peru": "PER", "Bolivia": "BOL", "Paraguay": "PAR", "Venezuela": "VEN",
    # UEFA
    "France": "FRA", "Spain": "ESP", "Germany": "ALE",
    "England": "ING", "Netherlands": "PB", "Belgium": "BEL",
    "Portugal": "POR", "Croatia": "CRO", "Switzerland": "SUI",
    "Serbia": "SRB", "Denmark": "DIN", "Poland": "POL",
    "Ukraine": "UCR", "Austria": "AUT", "Türkiye": "TUR", "Turkey": "TUR",
    "Scotland": "ESC", "Czechia": "CHE", "Czech Republic": "CHE",
    "Slovakia": "SVK", "Romania": "ROU", "Hungary": "HUN",
    "Albania": "ALB", "Slovenia": "SVN", "Georgia": "GEO",
    "Sweden": "SWE", "Norway": "NOR", "Bosnia-Herzegovina": "BIH",
    # CONCACAF
    "United States": "USA", "Mexico": "MEX", "Canada": "CAN",
    "Honduras": "HON", "Costa Rica": "CRC", "Panama": "PAN",
    "Jamaica": "JAM", "El Salvador": "SLV", "Haiti": "HAI",
    "Curaçao": "CUW", "Cura?ao": "CUW", "Guatemala": "GUA",
    # CAF
    "Morocco": "MAR", "Senegal": "SEN", "Nigeria": "NIG",
    "Ivory Coast": "CIV", "Côte d'Ivoire": "CIV",
    "Cameroon": "CMR", "Ghana": "GHA", "Egypt": "EGY",
    "Algeria": "DZA", "Mali": "MLI", "Tunisia": "TUN",
    "South Africa": "RSA", "Cape Verde Islands": "CPV",
    "Congo DR": "COD", "DR Congo": "COD", "Guinea": "GUI",
    # AFC
    "Japan": "JAP", "South Korea": "COR", "Korea Republic": "COR",
    "Iran": "IRN", "Saudi Arabia": "KSA", "Australia": "AUS",
    "Qatar": "QAT", "Uzbekistan": "UZB", "Iraq": "IRQ",
    "Jordan": "JOR", "Oman": "OMA", "Bahrain": "BHR",
    # OFC
    "New Zealand": "NZL",
}


def _resolve_code(tla: str, name: str) -> str:
    """Obtiene el código interno a partir del TLA o del nombre."""
    if tla:
        code = TLA_TO_CODE.get(tla.upper(), tla.upper())
        return code
    return NAME_TO_CODE.get(name, name[:3].upper())


def _parse_group(raw: Optional[str]) -> str:
    if not raw:
        return "?"
    return raw.replace("GROUP_", "").strip()


def _build_match(m: Dict, date_str: str) -> Optional[Dict]:
    home_tla  = (m["homeTeam"].get("tla") or "").strip()
    away_tla  = (m["awayTeam"].get("tla") or "").strip()
    home_name = m["homeTeam"].get("name", "")
    away_name = m["awayTeam"].get("name", "")

    # Descartar partidos con equipos sin definir (TBD en fases eliminatorias)
    if not home_name or not away_name or home_name == "None" or away_name == "None":
        return None

    home_code = _resolve_code(home_tla, home_name)
    away_code = _resolve_code(away_tla, away_name)

    group   = _parse_group(m.get("group"))
    matchday = m.get("matchday") or 0
    status  = m.get("status", "SCHEDULED")
    ft      = m.get("score", {}).get("fullTime", {})
    score_str = ""
    if ft and ft.get("home") is not None:
        score_str = f"{ft['home']}-{ft['away']}"

    # Hora local aproximada (UTC en la respuesta)
    utc_time = m.get("utcDate", "")[:16].replace("T", " ")

    return {
        "id":             str(m.get("id", f"{home_code}_{away_code}")),
        "date":           date_str,
        "team1":          home_code,
        "team2":          away_code,
        "team1_name_raw": home_name,
        "team2_name_raw": away_name,
        "group":          group,
        "matchday":       matchday,
        "venue":          m.get("venue") or "",
        "utc_time":       utc_time,
        "status":         status,
        "score_str":      score_str,
        "importance":     "normal",
        "context": (
            f"Grupo {group} · Jornada {matchday} · "
            f"{status.replace('_',' ')}"
            + (f" — Resultado: {score_str}" if score_str else "")
        ),
        "bookmaker_odds": None,
        "live":           True,
    }


@st.cache_data(ttl=180, show_spinner=False)
def fetch_wc_matches(date_str: str, api_key: str) -> Dict:
    """
    Obtiene partidos del WC 2026 para la fecha dada.
    Cache de 3 minutos para no agotar el límite de 10 req/min.
    """
    headers = {"X-Auth-Token": api_key}
    url     = f"{FD_BASE}/competitions/{WC_CODE}/matches"
    params  = {
        "dateFrom": date_str,
        "dateTo":   date_str,
    }

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.ConnectionError:
        return {"matches": [], "error": "Sin conexión a Internet."}
    except requests.exceptions.Timeout:
        return {"matches": [], "error": "Timeout al conectar con football-data.org."}

    if resp.status_code == 401:
        return {"matches": [], "error": "API Key inválida. Verifica tu token en football-data.org."}
    if resp.status_code == 403:
        return {"matches": [], "error": "Plan no cubre esta competición. Usa el modo demo."}
    if resp.status_code == 429:
        return {"matches": [], "error": "Límite de peticiones alcanzado (10/min). Espera un momento y recarga."}
    if not resp.ok:
        return {"matches": [], "error": f"Error API: HTTP {resp.status_code}"}

    try:
        data = resp.json()
    except Exception as e:
        return {"matches": [], "error": f"Error parseando respuesta: {e}"}

    raw = data.get("matches", [])
    parsed = [_build_match(m, date_str) for m in raw]
    parsed = [m for m in parsed if m is not None]   # eliminar TBD

    return {"matches": parsed, "error": None}
