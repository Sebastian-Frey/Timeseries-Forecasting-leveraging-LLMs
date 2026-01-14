# --- Location extractor module for Chronos integration ---
import re
import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher
import geonamescache
import pycountry
import pycountry_convert as pc
from unidecode import unidecode


# ----------------------------
# NLP setup
# ----------------------------
nlp = spacy.load("en_core_web_md")

# Geo data
gc = geonamescache.GeonamesCache()
country_names, country_by_name, country_alpha2 = set(), {}, {}
for c in pycountry.countries:
    canon = c.name
    country_alpha2[canon] = getattr(c, "alpha_2", None)
    for nm in {c.name, getattr(c, "official_name", c.name)}:
        low = nm.lower()
        country_names.add(low)
        country_by_name[low] = canon
for alias, canon in [
    ("usa", "United States"), ("u.s.a.", "United States"),
    ("uk", "United Kingdom"), ("u.k.", "United Kingdom"),
    ("viet nam", "Viet Nam"),
]:
    country_names.add(alias)
    country_by_name[alias] = canon

continent_names = {
    "europe", "asia", "africa", "north america", "south america", "oceania", "antarctica"
}
continent_by_name = {n: n.title().replace(" ", "") for n in continent_names}
continent_by_name["north america"] = "North America"
continent_by_name["south america"] = "South America"

def country_to_continent(country_name: str) -> str:
    if not country_name or not pc:
        return "Global"
    try:
        alpha2 = country_alpha2.get(country_name)
        if not alpha2: return "Global"
        cont_code = pc.country_alpha2_to_continent_code(alpha2)
        return pc.convert_continent_code_to_continent_name(cont_code)
    except Exception:
        return "Global"

# Cities
cities_dict = gc.get_cities()
city_names, city_to_country, _city_max_pop = set(), {}, {}
for c in cities_dict.values():
    nm, pop = c["name"], c.get("population", 0)
    city_names.add(nm.lower())
    if pop > _city_max_pop.get(nm, -1):
        cc = c.get("countrycode")
        if cc:
            try:
                pc_country = pycountry.countries.get(alpha_2=cc)
                if pc_country:
                    city_to_country[nm] = pc_country.name
                    _city_max_pop[nm] = pop
            except Exception:
                pass

CITY_ALIASES = {
    "nyc": "New York", "la": "Los Angeles", "vegas": "Las Vegas",
    "dc": "Washington", "d.c.": "Washington",
    "lisboa": "Lisbon", "roma": "Rome"
}
us_states = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC"
}
state_by_name = set(us_states.keys())
state_by_code = {v.lower(): k for k, v in us_states.items()}

# ----------------------------
# Phrase matchers
# ----------------------------
city_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
country_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
state_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
continent_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
def _nlp_list(terms): return [nlp.make_doc(t) for t in terms]
city_matcher.add("CITY", _nlp_list(city_names | set(CITY_ALIASES.keys())))
country_matcher.add("COUNTRY", _nlp_list(country_names))
state_matcher.add("STATE", _nlp_list(state_by_name | set(state_by_code.keys())))
continent_matcher.add("CONTINENT", _nlp_list(continent_names))

# ----------------------------
# Helper utilities
# ----------------------------
ABBREV_MAP = {"ft": "fort", "st": "saint", "st.": "saint"}
BRAND_BLACKLIST = {
    "adidas", "hertz", "dollar", "avis", "enterprise", "budget", "alamo", "sixt",
    "costco", "target", "nike", "zara", "ikea", "sprint", "verizon", "aaa",
    "mall", "outlet", "airport", "car", "rental", "rentals", "cars", "international",
    "downtown", "central", "center", "north", "south", "east", "west",
}
AIRPORT_CITY_REGEX = re.compile(r"(?<!\w)([a-zA-Z'.-]+(?:\s+[a-zA-Z'.-]+)?)\s+airport\b", re.I)

def fold_accents(s): return unidecode(s) if (isinstance(s, str) and unidecode) else s
def normalize_text(s):
    s = fold_accents(s or "")
    s = s.replace("+", " ")
    s = re.sub(r"[^a-zA-Z0-9\s,.'-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    toks = [ABBREV_MAP.get(tok.lower().strip(".,"), tok) for tok in s.split()]
    return " ".join(toks)

def resolve_city_alias(name): return CITY_ALIASES.get(name.lower(), name)
def city_to_country_lookup(city_name):
    for name_variant in (city_name, city_name.title()):
        if name_variant in city_to_country:
            return city_to_country[name_variant]
    return None

# ----------------------------
# Core extractor
# ----------------------------
def extract_place_struct(text: str) -> dict:
    text_norm = normalize_text(text)
    if not text_norm:
        return {"detected_city": None, "detected_country": "Global", "detected_continent": "Global"}

    doc = nlp(text_norm)
    cities, states, countries, continents, seen = [], [], [], [], set()

    # Airport pattern
    m = AIRPORT_CITY_REGEX.search(text_norm)
    if m:
        candidate = m.group(1).strip(".,' ").lower()
        if not any(tok in BRAND_BLACKLIST for tok in candidate.split()):
            if candidate in city_names or candidate in CITY_ALIASES:
                cities.append(resolve_city_alias(candidate).title())
                seen.add(candidate)

    # Gazetteer matching
    for matcher, sink in [
        (city_matcher, cities), (state_matcher, states),
        (country_matcher, countries), (continent_matcher, continents)
    ]:
        for _, s_idx, e_idx in matcher(doc):
            span = doc[s_idx:e_idx].text.strip(",. ").lower()
            if span in BRAND_BLACKLIST or span in seen:
                continue
            if matcher is city_matcher:
                sink.append(resolve_city_alias(span).title())
            elif matcher is country_matcher:
                sink.append(country_by_name.get(span, span.title()))
            elif matcher is state_matcher:
                sink.append(state_by_code.get(span, span).title())
            elif matcher is continent_matcher:
                sink.append(continent_by_name.get(span))
            seen.add(span)

    # Infer country
    detected_city = cities[0] if cities else None
    detected_country = countries[0] if countries else None
    if detected_city and not detected_country:
        cc = city_to_country_lookup(detected_city)
        if cc: detected_country = cc
    if not detected_country and states: detected_country = "United States"
    detected_country = detected_country or "Global"
    detected_continent = country_to_continent(detected_country)
    if detected_country == "Global" and continents:
        detected_continent = continents[0]
    return {
        "detected_city": detected_city,
        "detected_country": detected_country,
        "detected_continent": detected_continent,
    }

# --- Add geographical columns ---
from typing import Optional, Callable, List


def add_location_columns(
    df: pd.DataFrame,
    keywords: List[str],
    progress_callback: Optional[Callable[[int, int], None]] = None,
    batch_size: int = 1,
) -> pd.DataFrame:
    """
    Adds detected_city / detected_country / detected_continent to df
    but only for the specified keywords.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a `keyword` column.
    keywords : List[str]
        List of keywords to enrich (case-insensitive match against df['keyword']).
    progress_callback : callable(current: int, total: int) or None
        Optional callback called periodically with (current, total). Safe to pass
        a UI-updating function here (it will be wrapped in try/except).
    batch_size : int
        How often to call the progress callback (every `batch_size` keywords).
    """
    target = df[df["keyword"].str.lower().isin([k.lower() for k in keywords])].copy()
    if target.empty:
        print("⚠️ No matching keywords found in DataFrame.")
        return df

    # Work on unique keywords to avoid redundant processing
    unique_kws = target["keyword"].drop_duplicates().tolist()
    total = len(unique_kws)

    if progress_callback is None:
        # Fast vectorized path
        target["_kw_norm"] = target["keyword"].astype(str).map(normalize_text)
        res = target["_kw_norm"].apply(extract_place_struct)
        results_df = pd.DataFrame(res.tolist())
        mapping_df = pd.concat([target[["keyword"]], results_df], axis=1).drop_duplicates("keyword")
    else:
        # Iterative path with progress updates
        rows = []
        for idx, kw in enumerate(unique_kws, start=1):
            kw_norm = normalize_text(kw)
            struct = extract_place_struct(kw_norm)
            rows.append({"keyword": kw, **struct})
            if (idx % max(1, batch_size) == 0) or (idx == total):
                try:
                    progress_callback(idx, total)
                except Exception:
                    # swallow callback exceptions
                    pass
        mapping_df = pd.DataFrame(rows).drop_duplicates("keyword")

    df = df.merge(mapping_df, on="keyword", how="left", validate="many_to_one")
    return df
