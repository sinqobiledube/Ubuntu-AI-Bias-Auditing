"""Professional line-style SVG icons for the Streamlit UI.

Icons are inline SVG (no external CDN). Style matches common professional /
Flaticon line-icon sets — single stroke, rounded caps, currentColor fill.
"""
from __future__ import annotations

from urllib.parse import quote

# viewBox 0 0 24 24, stroke-based outline icons
_SVG_PATHS: dict[str, str] = {
    "globe": (
        '<circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="1.75"/>'
        '<ellipse cx="12" cy="12" rx="4" ry="9" fill="none" stroke="currentColor" stroke-width="1.75"/>'
        '<path d="M3 12h18" fill="none" stroke="currentColor" stroke-width="1.75"/>'
    ),
    "database": (
        '<ellipse cx="12" cy="6" rx="7" ry="3" fill="none" stroke="currentColor" stroke-width="1.75"/>'
        '<path d="M5 6v4c0 1.7 3.1 3 7 3s7-1.3 7-3V6" fill="none" stroke="currentColor" stroke-width="1.75"/>'
        '<path d="M5 10v4c0 1.7 3.1 3 7 3s7-1.3 7-3v-4" fill="none" stroke="currentColor" stroke-width="1.75"/>'
    ),
    "users": (
        '<circle cx="9" cy="8" r="3" fill="none" stroke="currentColor" stroke-width="1.75"/>'
        '<path d="M3 19c0-3.3 2.7-5 6-5" fill="none" stroke="currentColor" stroke-width="1.75"/>'
        '<circle cx="17" cy="9" r="2.5" fill="none" stroke="currentColor" stroke-width="1.75"/>'
        '<path d="M14 19c0-2.5 2-4 5-4" fill="none" stroke="currentColor" stroke-width="1.75"/>'
    ),
    "chart": (
        '<path d="M4 19V9" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round"/>'
        '<path d="M10 19V5" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round"/>'
        '<path d="M16 19v-7" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round"/>'
        '<path d="M22 19H2" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round"/>'
    ),
    "scale": (
        '<path d="M12 4v16" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round"/>'
        '<path d="M6 8h12" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round"/>'
        '<path d="M6 8l-3 5h6L6 8z" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linejoin="round"/>'
        '<path d="M18 8l-3 5h6l-3-5z" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linejoin="round"/>'
    ),
    "shield": (
        '<path d="M12 3l7 3v6c0 4.5-3 7.5-7 9-4-1.5-7-4.5-7-9V6l7-3z" '
        'fill="none" stroke="currentColor" stroke-width="1.75" stroke-linejoin="round"/>'
    ),
    "search": (
        '<circle cx="11" cy="11" r="6" fill="none" stroke="currentColor" stroke-width="1.75"/>'
        '<path d="M16 16l4 4" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round"/>'
    ),
    "stats": (
        '<path d="M4 20V10M10 20V4M16 20v-6M22 20H2" fill="none" stroke="currentColor" '
        'stroke-width="1.75" stroke-linecap="round"/>'
    ),
    "framework": (
        '<rect x="4" y="4" width="6" height="6" rx="1" fill="none" stroke="currentColor" stroke-width="1.75"/>'
        '<rect x="14" y="4" width="6" height="6" rx="1" fill="none" stroke="currentColor" stroke-width="1.75"/>'
        '<rect x="4" y="14" width="6" height="6" rx="1" fill="none" stroke="currentColor" stroke-width="1.75"/>'
        '<rect x="14" y="14" width="6" height="6" rx="1" fill="none" stroke="currentColor" stroke-width="1.75"/>'
    ),
    "target": (
        '<circle cx="12" cy="12" r="8" fill="none" stroke="currentColor" stroke-width="1.75"/>'
        '<circle cx="12" cy="12" r="4" fill="none" stroke="currentColor" stroke-width="1.75"/>'
        '<circle cx="12" cy="12" r="1.25" fill="currentColor"/>'
    ),
    "trend_down": (
        '<path d="M4 16l6-6 4 4 6-8" fill="none" stroke="currentColor" stroke-width="1.75" '
        'stroke-linecap="round" stroke-linejoin="round"/>'
        '<path d="M14 6h6v6" fill="none" stroke="currentColor" stroke-width="1.75" '
        'stroke-linecap="round" stroke-linejoin="round"/>'
    ),
    "download": (
        '<path d="M12 4v10" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round"/>'
        '<path d="M8 10l4 4 4-4" fill="none" stroke="currentColor" stroke-width="1.75" '
        'stroke-linecap="round" stroke-linejoin="round"/>'
        '<path d="M4 18h16" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round"/>'
    ),
    "refresh": (
        '<path d="M20 12a8 8 0 10-2.3 5.7" fill="none" stroke="currentColor" stroke-width="1.75" '
        'stroke-linecap="round"/>'
        '<path d="M20 12v-5h-5" fill="none" stroke="currentColor" stroke-width="1.75" '
        'stroke-linecap="round" stroke-linejoin="round"/>'
    ),
    "play": (
        '<path d="M8 6l10 6-10 6V6z" fill="currentColor"/>'
    ),
    "bulb": (
        '<path d="M9 18h6M10 21h4" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round"/>'
        '<path d="M12 3a6 6 0 014 10.5c-.8.7-1.3 1.6-1.4 2.5h-5.2c-.1-.9-.6-1.8-1.4-2.5A6 6 0 0112 3z" '
        'fill="none" stroke="currentColor" stroke-width="1.75" stroke-linejoin="round"/>'
    ),
    "check": (
        '<circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="1.75"/>'
        '<path d="M8 12l2.5 2.5L16 9" fill="none" stroke="currentColor" stroke-width="1.75" '
        'stroke-linecap="round" stroke-linejoin="round"/>'
    ),
    "cursor": (
        '<path d="M5 4l12 6-5.5 1.5L10 18 5 4z" fill="none" stroke="currentColor" '
        'stroke-width="1.75" stroke-linejoin="round"/>'
    ),
    "info": (
        '<circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="1.75"/>'
        '<path d="M12 10v6M12 7h.01" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round"/>'
    ),
    "alert": (
        '<path d="M12 4l8 14H4L12 4z" fill="none" stroke="currentColor" stroke-width="1.75" '
        'stroke-linejoin="round"/>'
        '<path d="M12 10v3M12 16h.01" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round"/>'
    ),
}

TAB_ICON_KEYS = ("database", "search", "stats", "framework")


def icon_svg(name: str, *, size: int = 20, color: str = "currentColor") -> str:
    """Return an inline SVG icon as an HTML snippet."""
    if name not in _SVG_PATHS:
        raise KeyError(f"Unknown icon: {name}")
    inner = _SVG_PATHS[name]
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" style="color:{color};flex-shrink:0;">{inner}</svg>'
    )


def icon_html(name: str, *, size: int = 20, color: str = "currentColor") -> str:
    """Wrap icon SVG for use inside markdown HTML blocks."""
    return (
        f'<span class="dat-icon" style="display:inline-flex;align-items:center;'
        f'line-height:0;vertical-align:middle;">{icon_svg(name, size=size, color=color)}</span>'
    )


def icon_data_uri(name: str, *, color: str = "#0F6E63") -> str:
    """SVG as a CSS data URI (for tab ::before backgrounds)."""
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
        f'style="color:{color}">{_SVG_PATHS[name]}</svg>'
    )
    return f"url(\"data:image/svg+xml,{quote(svg)}\")"


def tab_icons_css(*, color: str = "#0F6E63", selected_color: str = "#FFFFFF") -> str:
    """Prefix each tab button label with a small inline SVG icon."""
    tab_btn = 'div[data-testid="stTabs"] button[data-baseweb="tab"], div[data-testid="stTabs"] button[role="tab"]'
    rules = []
    for i, key in enumerate(TAB_ICON_KEYS, start=1):
        rules.append(
            f"{tab_btn}:nth-of-type({i})::before {{"
            f'content:"";display:inline-block;flex-shrink:0;width:16px;height:16px;margin-right:8px;'
            f"background:{icon_data_uri(key, color=color)} center/contain no-repeat;"
            f"}}"
            f'{tab_btn}[aria-selected="true"]:nth-of-type({i})::before {{'
            f"background:{icon_data_uri(key, color=selected_color)} center/contain no-repeat;"
            f"}}"
        )
    return "\n".join(rules)
