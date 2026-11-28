"""Template tags for displaying club crests.

Usage in a template::

    {% load team_tags %}
    {% team_logo team 28 %}
    {% team_logo "Man City" 40 "ring-2 ring-white" %}

The tag accepts a ``Team`` instance, a plain club name, or ``None``. It returns
an ``<img>`` pointing to ``static/logos/<slug>.png``. If no crest is found it
falls back to a small initials badge so the UI never breaks.
"""

from django import template
from django.contrib.staticfiles import finders
from django.templatetags.static import static
from django.utils.html import format_html
from django.utils.text import slugify

register = template.Library()

# Names used in the database that do not slugify to the file name we ship.
LOGO_SLUGS = {
    "man city": "man-city",
    "manchester city": "man-city",
    "man united": "man-united",
    "manchester united": "man-united",
    "newcastle united": "newcastle",
    "nottingham forest": "nottingham-forest",
    "nott'm forest": "nottingham-forest",
    "sheffield united": "sheffield-united",
    "west brom": "west-brom",
    "west bromwich albion": "west-brom",
    "west ham": "west-ham",
    "west ham united": "west-ham",
    "wolverhampton": "wolves",
    "wolverhampton wanderers": "wolves",
    "tottenham hotspur": "tottenham",
    "brighton & hove albion": "brighton",
    "brighton and hove albion": "brighton",
    "afc bournemouth": "bournemouth",
    "leicester city": "leicester",
    "leeds united": "leeds",
    "hull city": "hull",
    "coventry city": "coventry",
    "luton town": "luton",
    "norwich city": "norwich",
    "ipswich town": "ipswich",
    "crystal palace": "crystal-palace",
    "aston villa": "aston-villa",
}


def _team_name(team):
    if team is None:
        return ""
    name = getattr(team, "name", None)
    if name:
        return str(name).strip()
    return str(team).strip()


@register.simple_tag
def team_logo(team, size=28, css_class=""):
    """Render a club crest, with an initials fallback."""
    name = _team_name(team)
    if not name:
        return ""

    slug = LOGO_SLUGS.get(name.casefold(), slugify(name))
    path = f"logos/{slug}.png"
    base_classes = "inline-block object-contain shrink-0 align-middle"
    classes = f"{base_classes} {css_class}".strip()

    if finders.find(path):
        return format_html(
            '<img src="{}" alt="{}" title="{}" width="{}" height="{}" '
            'loading="lazy" class="{}">',
            static(path),
            name,
            name,
            size,
            size,
            classes,
        )

    initials = "".join(part[0] for part in name.split()[:2]).upper() or "?"
    return format_html(
        '<span title="{}" aria-label="{}" '
        'class="inline-flex items-center justify-center rounded-full '
        'bg-slate-200 text-slate-600 font-bold align-middle shrink-0 {}" '
        'style="width:{}px;height:{}px;font-size:{}px;">{}</span>',
        name,
        name,
        css_class,
        size,
        size,
        max(9, int(size * 0.36)),
        initials,
    )


@register.simple_tag
def team_logo_url(team):
    """Return the static URL of a club crest, or an empty string if missing."""
    name = _team_name(team)
    if not name:
        return ""
    slug = LOGO_SLUGS.get(name.casefold(), slugify(name))
    path = f"logos/{slug}.png"
    return static(path) if finders.find(path) else ""
