#!/usr/bin/env python3
"""
gtfs_auto_update.py — descarga el GTFS oficial de TITSA y regenera
stops-data.js y patterns-data.js con los datos más recientes, cubriendo
HOY + los próximos 2 días (para que la app permita elegir "hoy / mañana /
pasado mañana").

USO:
  pip install pandas requests --break-system-packages
  python3 gtfs_auto_update.py

Pensado para ejecutarse automáticamente una vez al día (ver
.github/workflows/update-gtfs.yml). Al terminar, sustituye stops-data.js y
patterns-data.js junto al index.html — solo hay que volver a desplegar/subir
la app para que se vea el cambio.
"""

import collections
import io
import json
import math
import zipfile
from datetime import date, datetime, timedelta

import pandas as pd
import requests

import tram_gtfs_merge

GTFS_URL = "http://www.titsa.com/Google_transit.zip"
OUTPUT_DIR = "."       # cambia esto si quieres guardar en otra carpeta
DAYS_AHEAD = 3         # hoy + este número de días - 1 (3 = hoy, mañana, pasado mañana)
SHAPE_SIMPLIFY_EPS = 0.00015  # ~15 m de tolerancia al simplificar la geometría de las líneas


def to_sec(t):
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def target_dates():
    today = date.today()
    return [(today + timedelta(days=i)).strftime("%Y%m%d") for i in range(DAYS_AHEAD)]


def perp_dist(pt, start, end):
    if start == end:
        return math.hypot(pt[0] - start[0], pt[1] - start[1])
    x0, y0 = pt
    x1, y1 = start
    x2, y2 = end
    num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    den = math.hypot(y2 - y1, x2 - x1)
    return num / den


def douglas_peucker(points, epsilon):
    if len(points) < 3:
        return points
    dmax, index = 0, 0
    for i in range(1, len(points) - 1):
        d = perp_dist(points[i], points[0], points[-1])
        if d > dmax:
            index, dmax = i, d
    if dmax > epsilon:
        left = douglas_peucker(points[: index + 1], epsilon)
        right = douglas_peucker(points[index:], epsilon)
        return left[:-1] + right
    return [points[0], points[-1]]


def encode_polyline(coords, precision=5):
    factor = 10 ** precision
    output = []
    prev_lat = prev_lon = 0
    for lat, lon in coords:
        lat_i, lon_i = round(lat * factor), round(lon * factor)
        for v in (lat_i - prev_lat, lon_i - prev_lon):
            v = ~(v << 1) if v < 0 else (v << 1)
            while v >= 0x20:
                output.append(chr((0x20 | (v & 0x1F)) + 63))
                v >>= 5
            output.append(chr(v + 63))
        prev_lat, prev_lon = lat_i, lon_i
    return "".join(output)


def main():
    print(f"Descargando GTFS desde {GTFS_URL} ...")
    resp = requests.get(GTFS_URL, timeout=120)
    resp.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(resp.content))

    def read(name, **kw):
        with z.open(name) as f:
            return pd.read_csv(f, encoding="utf-8-sig", **kw)

    stops = read("stops.txt", dtype={"stop_id": str})
    routes = read("routes.txt", dtype={"route_id": str})
    trips = read("trips.txt", dtype={"route_id": str, "service_id": str, "trip_id": str, "shape_id": str})
    cal_dates = read("calendar_dates.txt", dtype={"service_id": str, "date": str})
    try:
        cal_weekly = read(
            "calendar.txt",
            dtype={"service_id": str, "start_date": str, "end_date": str},
        )
    except KeyError:
        cal_weekly = None  # some feeds only use calendar_dates.txt — handle gracefully

    WEEKDAY_COLS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

    def active_services_for_date(d):
        active = set()
        if cal_weekly is not None:
            weekday_idx = datetime.strptime(d, "%Y%m%d").weekday()  # 0 = Monday
            col = WEEKDAY_COLS[weekday_idx]
            mask = (
                (cal_weekly[col].astype(str) == "1")
                & (cal_weekly["start_date"] <= d)
                & (cal_weekly["end_date"] >= d)
            )
            active |= set(cal_weekly.loc[mask, "service_id"])
        exceptions_today = cal_dates[cal_dates["date"] == d]
        added = set(exceptions_today[exceptions_today["exception_type"] == 1]["service_id"])
        removed = set(exceptions_today[exceptions_today["exception_type"] == 2]["service_id"])
        return (active | added) - removed

    dates = target_dates()
    active_by_date = {}
    for d in dates:
        active_by_date[d] = active_services_for_date(d)
        print(f"{d}: {len(active_by_date[d])} servicios activos")

    all_active = set()
    for s in active_by_date.values():
        all_active |= s

    trips_relevant = trips[trips["service_id"].isin(all_active)].copy()
    route_info = routes.set_index("route_id")[["route_short_name", "route_long_name", "route_color"]].to_dict("index")
    relevant_trip_ids = set(trips_relevant["trip_id"])
    trip_to_route = dict(zip(trips_relevant["trip_id"], trips_relevant["route_id"]))
    trip_to_headsign = dict(zip(trips_relevant["trip_id"], trips_relevant["trip_headsign"]))
    trip_to_shape = dict(zip(trips_relevant["trip_id"], trips_relevant["shape_id"]))
    trip_to_service = dict(zip(trips_relevant["trip_id"], trips_relevant["service_id"]))

    print("Procesando stop_times.txt (el fichero más grande, puede tardar un poco)...")
    rows_by_trip = collections.defaultdict(list)
    for chunk in pd.read_csv(
        z.open("stop_times.txt"),
        encoding="utf-8-sig",
        dtype={"trip_id": str, "stop_id": str, "arrival_time": str, "departure_time": str, "stop_sequence": int},
        chunksize=2_000_000,
    ):
        chunk = chunk[chunk["trip_id"].isin(relevant_trip_ids)]
        for row in chunk.itertuples(index=False):
            rows_by_trip[row.trip_id].append((row.stop_sequence, row.stop_id, row.departure_time))

    # -------- stops-data.js (unchanged: paradas + qué líneas paran en cada una) --------
    stop_routes = {}
    for trip_id, rows in rows_by_trip.items():
        route_id = trip_to_route.get(trip_id)
        short = route_info.get(route_id, {}).get("route_short_name", "")
        for _, stop_id, _ in rows:
            stop_routes.setdefault(stop_id, set()).add(short)

    stops_out = []
    for _, row in stops.iterrows():
        sid = row["stop_id"]
        lines = sorted(stop_routes.get(sid, []), key=lambda x: (len(str(x)), str(x)))
        stops_out.append({
            "id": sid,
            "name": row["stop_name"],
            "lat": round(float(row["stop_lat"]), 5),
            "lon": round(float(row["stop_lon"]), 5),
            "lines": lines[:6],
        })

    # -------- tram (Metropolitano de Tenerife) stops, merged in alongside the bus ones --------
    # See tram_gtfs_merge.py for why this is a separate, statically-bundled feed rather than a
    # second live download, and why every tram id gets a "TRAM-" prefix (guarantees it can never
    # collide with whatever ids TITSA's bus feed happens to use, now or in the future).
    tram_stops_out, tram_shapes_encoded, tram_patterns_by_date = tram_gtfs_merge.build_tram_patterns(dates)
    stops_out.extend(tram_stops_out)
    if tram_stops_out:
        print(f"+ {len(tram_stops_out)} paradas de tranvía añadidas (Metropolitano de Tenerife)")

    with open(f"{OUTPUT_DIR}/stops-data.js", "w", encoding="utf-8") as f:
        f.write("// Real TITSA stops data, auto-generated by gtfs_auto_update.py\n")
        f.write(f"// Generated: {datetime.now().isoformat()} — {len(stops_out)} stops\n")
        f.write("let REAL_STOPS_DB = ")
        json.dump(stops_out, f, ensure_ascii=False, separators=(",", ":"))
        f.write(";\n")
    print(f"stops-data.js actualizado ({len(stops_out)} paradas)")

    # -------- patterns per date + deduped shapes --------
    patterns_by_date = {}
    needed_shape_ids = set()

    for d in dates:
        active_services = active_by_date[d]
        patterns = {}
        for trip_id, service_id in trip_to_service.items():
            if service_id not in active_services:
                continue
            rows = rows_by_trip.get(trip_id)
            if not rows:
                continue
            rows_sorted = sorted(rows, key=lambda r: r[0])
            stop_ids = tuple(r[1] for r in rows_sorted)
            if len(stop_ids) < 2:
                continue
            times_sec = [to_sec(r[2]) for r in rows_sorted]
            route_id = trip_to_route.get(trip_id)
            key = (route_id, stop_ids)
            first_dep = times_sec[0]
            offsets = [t - first_dep for t in times_sec]
            if key not in patterns:
                patterns[key] = {
                    "route_id": route_id, "stops": list(stop_ids), "offsets": offsets,
                    "headsign": trip_to_headsign.get(trip_id, ""), "departures": [],
                    "shape_ids": collections.Counter(),
                }
            patterns[key]["departures"].append(first_dep)
            sid = trip_to_shape.get(trip_id)
            if sid and str(sid) != "nan":
                patterns[key]["shape_ids"][sid] += 1

        out = []
        for key, p in patterns.items():
            info = route_info.get(p["route_id"], {})
            p["departures"] = sorted(p["departures"])
            chosen_shape = p["shape_ids"].most_common(1)[0][0] if p["shape_ids"] else None
            if chosen_shape:
                needed_shape_ids.add(chosen_shape)
            out.append({
                "routeId": p["route_id"], "short": info.get("route_short_name", ""),
                "color": info.get("route_color", "75AD1C"), "headsign": p["headsign"],
                "stops": p["stops"], "offsets": p["offsets"], "departures": p["departures"],
                "shapeId": chosen_shape,
            })
        formatted_date = f"{d[:4]}-{d[4:6]}-{d[6:]}"
        patterns_by_date[formatted_date] = out
        print(f"{formatted_date}: {len(out)} patrones")

    # -------- shapes.txt: extract, simplify, encode once (shared across all dates) --------
    print(f"Extrayendo geometría real de {len(needed_shape_ids)} líneas distintas...")
    shapes_needed = read("shapes.txt", dtype={"shape_id": str})
    shapes_needed = shapes_needed[shapes_needed["shape_id"].isin(needed_shape_ids)].sort_values(
        ["shape_id", "shape_pt_sequence"]
    )
    shapes_encoded = {}
    for sid, grp in shapes_needed.groupby("shape_id"):
        coords = [(round(lat, 5), round(lon, 5)) for lat, lon in zip(grp["shape_pt_lat"], grp["shape_pt_lon"])]
        simplified = douglas_peucker(coords, SHAPE_SIMPLIFY_EPS)
        shapes_encoded[sid] = encode_polyline(simplified)

    # -------- merge tram (Metropolitano de Tenerife) patterns + shapes in per date --------
    for formatted_date, tram_patterns in tram_patterns_by_date.items():
        patterns_by_date.setdefault(formatted_date, []).extend(tram_patterns)
        if tram_patterns:
            print(f"{formatted_date}: +{len(tram_patterns)} patrones de tranvía")
    shapes_encoded.update(tram_shapes_encoded)

    with open(f"{OUTPUT_DIR}/patterns-data.js", "w", encoding="utf-8") as f:
        f.write("// Real TITSA schedule patterns for the next few days, auto-generated by gtfs_auto_update.py\n")
        f.write("// Each pattern references a shapeId; the road geometry lives once in REAL_SHAPES.\n")
        available_dates = list(patterns_by_date.keys())
        f.write("let AVAILABLE_DATES = ")
        json.dump(available_dates, f)
        f.write(";\n")
        f.write("let REAL_SHAPES = ")
        json.dump(shapes_encoded, f, ensure_ascii=False, separators=(",", ":"))
        f.write(";\n")
        f.write("let REAL_PATTERNS_BY_DATE = ")
        json.dump(patterns_by_date, f, ensure_ascii=False, separators=(",", ":"))
        f.write(";\n")

    print(f"patterns-data.js actualizado — fechas disponibles: {available_dates}")
    print("\n✅ Listo. Sube stops-data.js y patterns-data.js junto a tu index.html para publicar los datos nuevos.")


if __name__ == "__main__":
    main()
