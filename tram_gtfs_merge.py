#!/usr/bin/env python3
"""
tram_gtfs_merge.py — expands the Metropolitano de Tenerife (tranvía) GTFS feed into the same
pattern format gtfs_auto_update.py already produces for TITSA buses, so the app's route planner
treats tram lines exactly like bus lines (same REAL_PATTERNS array, same STOP_INDEX, same
findDirectTrips/findTransferTrips/findTwoTransferTrips search) — meaning it will automatically
pick a tram leg whenever that's genuinely the fastest way, transfer between tram and bus, etc.,
with zero special-casing needed anywhere else in index.html.

Like the bus feed, this downloads the LIVE feed every time it runs — from Transitland's Atlas
registry entry for Metropolitano de Tenerife — so the tram schedule stays current automatically,
the same way the bus schedule does:
    https://metrotenerife.com/transit/google_transit.zip
If that download ever fails (site down, URL changed, no network), this falls back to the
tram-gtfs/*.txt snapshot bundled in the repo, so a single outage never breaks tram search
entirely — it just means tram schedules stay at whatever was last successfully fetched.

A note on calendar.txt: as of the version bundled here, the feed's calendar.txt has a
start/end_date window that can lag behind today's date (transit agencies don't always bump this
field promptly). Rather than let a stale end_date silently make the tram vanish from search one
day, service is resolved purely by weekday (Mon-Fri / Saturday / Sunday) — i.e. treated as an
"evergreen" weekly timetable — which matches how a real, ongoing tram service actually behaves.

All tram-side stop_ids / route_ids / trip_ids / shape_ids are namespaced with a "TRAM-" prefix
before being merged in, so they can never collide with whatever ids the (separately, dynamically
downloaded) TITSA bus feed happens to use, now or in the future.
"""

import collections
import io
import os
import zipfile
from datetime import datetime

import pandas as pd
import requests

TRAM_GTFS_URL = "https://metrotenerife.com/transit/google_transit.zip"
TRAM_DIR = "tram-gtfs"  # local fallback snapshot bundled in the repo, used only if the live download fails
TRAM_PREFIX = "TRAM-"
TRAM_ROUTE_COLOR = "8E44AD"  # distinct purple so tram pills are visually distinct from bus pills

WEEKDAY_COLS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

_tram_zip = None
_tram_zip_tried = False


def _get_tram_zip():
    """Downloads the live tram feed once per script run; returns a ZipFile, or None if the
    download failed (callers then fall back to the bundled tram-gtfs/ snapshot)."""
    global _tram_zip, _tram_zip_tried
    if _tram_zip_tried:
        return _tram_zip
    _tram_zip_tried = True
    try:
        print(f"Descargando GTFS del tranvía desde {TRAM_GTFS_URL} ...")
        resp = requests.get(TRAM_GTFS_URL, timeout=60)
        resp.raise_for_status()
        _tram_zip = zipfile.ZipFile(io.BytesIO(resp.content))
        print("Tranvía: descarga en vivo correcta.")
    except Exception as e:
        print(f"Tranvía: no se pudo descargar el feed en vivo ({e}) — uso la copia local de tram-gtfs/ como respaldo.")
        _tram_zip = None
    return _tram_zip


def _read(name, **kw):
    z = _get_tram_zip()
    if z is not None:
        with z.open(name) as f:
            return pd.read_csv(f, encoding="utf-8-sig", **kw)
    return pd.read_csv(f"{TRAM_DIR}/{name}", encoding="utf-8-sig", **kw)


def tram_gtfs_available():
    if _get_tram_zip() is not None:
        return True
    return os.path.isdir(TRAM_DIR) and os.path.isfile(f"{TRAM_DIR}/stops.txt")


def _to_sec(t):
    h, m, s = str(t).split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def _expand_frequencies(base_offsets_by_trip, frequencies):
    """For each (trip_id, [start,end,headway]) window, generate one departure (seconds since
    midnight, at the FIRST stop) every `headway_secs` from start up to (not including) end —
    the standard GTFS frequencies.txt expansion. base_offsets_by_trip isn't needed here (the
    windows are already in first-stop time), kept as a parameter for symmetry/clarity."""
    departures_by_trip = collections.defaultdict(list)
    for row in frequencies.itertuples(index=False):
        trip_id = row.trip_id
        start = _to_sec(row.start_time)
        end = _to_sec(row.end_time)
        headway = int(row.headway_secs)
        if headway <= 0:
            continue
        t = start
        while t < end:
            departures_by_trip[trip_id].append(t)
            t += headway
    return departures_by_trip


def _active_services_by_weekday(calendar_df):
    """service_id -> set of weekday indices (0=Monday..6=Sunday) it runs on, ignoring
    start_date/end_date (see module docstring for why)."""
    services = {}
    for row in calendar_df.itertuples(index=False):
        days = set()
        for i, col in enumerate(WEEKDAY_COLS):
            if str(getattr(row, col)) == "1":
                days.add(i)
        services[row.service_id] = days
    return services


def build_tram_patterns(target_dates):
    """target_dates: list of 'YYYYMMDD' strings (same format gtfs_auto_update.py uses).
    Returns (stops_out, shapes_encoded, patterns_by_formatted_date) using the exact same shapes
    gtfs_auto_update.py already produces for buses, ready to be appended/merged in."""
    if not tram_gtfs_available():
        return [], {}, {}

    stops = _read("stops.txt", dtype={"stop_id": str})
    routes = _read("routes.txt", dtype={"route_id": str})
    trips = _read(
        "trips.txt",
        dtype={"route_id": str, "service_id": str, "trip_id": str, "shape_id": str},
    )
    calendar_df = _read(
        "calendar.txt", dtype={"service_id": str, "start_date": str, "end_date": str}
    )
    stop_times = _read(
        "stop_times.txt",
        dtype={"trip_id": str, "stop_id": str, "arrival_time": str, "departure_time": str, "stop_sequence": int},
    )
    frequencies = _read("frequencies.txt", dtype={"trip_id": str})
    try:
        shapes = _read("shapes.txt", dtype={"shape_id": str})
    except (FileNotFoundError, KeyError):
        shapes = None

    route_info = routes.set_index("route_id")[["route_short_name", "route_long_name"]].to_dict("index")
    trip_to_route = dict(zip(trips["trip_id"], trips["route_id"]))
    trip_to_headsign = dict(zip(trips["trip_id"], trips["trip_headsign"]))
    trip_to_shape = dict(zip(trips["trip_id"], trips["shape_id"])) if "shape_id" in trips else {}
    trip_to_service = dict(zip(trips["trip_id"], trips["service_id"]))

    # base stop sequence + relative offsets for each trip_id, from stop_times.txt
    rows_by_trip = collections.defaultdict(list)
    for row in stop_times.itertuples(index=False):
        rows_by_trip[row.trip_id].append((row.stop_sequence, row.stop_id, row.arrival_time))

    base_pattern_by_trip = {}
    for trip_id, rows in rows_by_trip.items():
        rows_sorted = sorted(rows, key=lambda r: r[0])
        stop_ids = [TRAM_PREFIX + str(r[1]) for r in rows_sorted]
        times_sec = [_to_sec(r[2]) for r in rows_sorted]
        first = times_sec[0]
        offsets = [t - first for t in times_sec]
        base_pattern_by_trip[trip_id] = (stop_ids, offsets)

    departures_by_trip = _expand_frequencies(base_pattern_by_trip, frequencies)

    services_by_weekday = _active_services_by_weekday(calendar_df)

    # -------- stops --------
    stop_routes = collections.defaultdict(set)
    for trip_id in departures_by_trip:
        route_id = trip_to_route.get(trip_id)
        short = route_info.get(route_id, {}).get("route_short_name", "")
        stop_ids, _ = base_pattern_by_trip.get(trip_id, ([], []))
        for sid in stop_ids:
            stop_routes[sid].add(short)

    stops_out = []
    for row in stops.itertuples(index=False):
        sid = TRAM_PREFIX + str(row.stop_id)
        lines = sorted(stop_routes.get(sid, []), key=lambda x: (len(str(x)), str(x)))
        stops_out.append({
            "id": sid,
            "name": "🚊 " + str(row.stop_name),  # tram icon prefix so it reads clearly in results/search
            "lat": round(float(row.stop_lat), 5),
            "lon": round(float(row.stop_lon), 5),
            "lines": lines[:6],
        })

    # -------- shapes (once, shared across dates) --------
    shapes_encoded = {}
    if shapes is not None:
        # reuse the same simplification/encoding approach as gtfs_auto_update.py so the tram
        # line renders on the map exactly like a bus line does
        from gtfs_auto_update import douglas_peucker, encode_polyline, SHAPE_SIMPLIFY_EPS
        needed_shape_ids = set(s for s in trip_to_shape.values() if s and str(s) != "nan")
        shapes_relevant = shapes[shapes["shape_id"].isin(needed_shape_ids)].sort_values(
            ["shape_id", "shape_pt_sequence"]
        )
        for sid, grp in shapes_relevant.groupby("shape_id"):
            coords = [(round(lat, 5), round(lon, 5)) for lat, lon in zip(grp["shape_pt_lat"], grp["shape_pt_lon"])]
            simplified = douglas_peucker(coords, SHAPE_SIMPLIFY_EPS)
            shapes_encoded[TRAM_PREFIX + sid] = encode_polyline(simplified)

    # -------- patterns per date --------
    patterns_by_date = {}
    for d in target_dates:
        weekday_idx = datetime.strptime(d, "%Y%m%d").weekday()
        active_services = {sid for sid, days in services_by_weekday.items() if weekday_idx in days}

        patterns = {}
        for trip_id, deps in departures_by_trip.items():
            service_id = trip_to_service.get(trip_id)
            if service_id not in active_services:
                continue
            stop_ids, offsets = base_pattern_by_trip.get(trip_id, ([], []))
            if len(stop_ids) < 2:
                continue
            route_id = trip_to_route.get(trip_id)
            key = (route_id, tuple(stop_ids))
            if key not in patterns:
                shape_id = trip_to_shape.get(trip_id)
                patterns[key] = {
                    "route_id": route_id, "stops": stop_ids, "offsets": offsets,
                    "headsign": trip_to_headsign.get(trip_id, ""), "departures": [],
                    "shape_id": (TRAM_PREFIX + shape_id) if shape_id and str(shape_id) != "nan" else None,
                }
            patterns[key]["departures"].extend(deps)

        out = []
        for key, p in patterns.items():
            info = route_info.get(p["route_id"], {})
            p["departures"] = sorted(set(p["departures"]))
            out.append({
                "routeId": TRAM_PREFIX + str(p["route_id"]),
                "short": info.get("route_short_name", ""),
                "color": TRAM_ROUTE_COLOR,
                "headsign": "🚊 " + str(p["headsign"]),
                "stops": p["stops"], "offsets": p["offsets"], "departures": p["departures"],
                "shapeId": p["shape_id"],
            })
        formatted_date = f"{d[:4]}-{d[4:6]}-{d[6:]}"
        patterns_by_date[formatted_date] = out

    return stops_out, shapes_encoded, patterns_by_date
