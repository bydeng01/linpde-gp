r"""OpenAQ v3 monitor loader (real data).

OpenAQ v1/v2 were retired on 31 January 2025; this uses the v3 API at
https://api.openaq.org/v3/ , which requires a free API key (register at
https://explore.openaq.org and pass it in the ``X-API-Key`` header). The loader
pulls one pollutant (PM2.5 or NO2) from monitors inside a city bounding box for a
single time window and writes

    data/openaq_<city>_<pollutant>_<date>.csv

with columns (sensor_id, lon, lat, datetime, parameter, value, unit), the schema
``run.py --data`` consumes. The London Air Quality Network (londonair.org.uk) is
an acceptable alternative source with the same downstream format.

A live API pull is not itself reproducible (key + network + time window), so a
small committed sample CSV (real London monitor coordinates, synthetic one-hour
NO2 values) is shipped via ``--sample`` and is what ``reproduce.sh`` uses. The
bounding boxes are listed in ``CITY_BBOX`` below.

Usage:

    # Reproducible committed snapshot (no network):
    python fetch_data.py --sample

    # Live pull (needs OPENAQ_API_KEY in the environment):
    OPENAQ_API_KEY=... python fetch_data.py --city london --pollutant no2 \
        --datetime-from 2024-07-15T12:00:00Z --datetime-to 2024-07-15T13:00:00Z
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

# Default city / region bounding boxes (lon_min, lat_min, lon_max, lat_max).
CITY_BBOX = {
    "london": (-0.25, 51.42, 0.05, 51.56),
    "maryland": (-79.49, 37.91, -75.05, 39.72),  # whole state
    "baltimore": (-76.71, 39.20, -76.53, 39.37),  # city scale (steady-state defensible)
    "dc_metro": (-77.20, 38.79, -76.85, 39.00),  # Washington DC / MD suburbs
}

PARAM_ID = {"pm25": 2, "no2": 7}  # OpenAQ v3 parameter ids.

# Real central-London monitor coordinates (LAQN/AURN sites). The committed
# sample uses these locations with a one-hour NO2 snapshot (ug/m3); the values
# are a plausible mid-day pattern (kerbside high, park low), not a live pull.
_LONDON_SITES = [
    # (sensor_id, name, lon, lat, value_ug_m3)
    (101, "Marylebone Road", -0.15464, 51.52253, 86.0),
    (102, "Westminster", -0.13193, 51.49467, 61.0),
    (103, "Bloomsbury", -0.12588, 51.52229, 47.0),
    (104, "North Kensington", -0.21349, 51.52105, 33.0),
    (105, "City of London", -0.08319, 51.51384, 54.0),
    (106, "Southwark A2", -0.06597, 51.48050, 58.0),
    (107, "Camden Kerbside", -0.17500, 51.54420, 72.0),
    (108, "Hackney", -0.05600, 51.54900, 39.0),
    (109, "Tower Hamlets", -0.04200, 51.52200, 49.0),
    (110, "Greenwich Eltham", 0.02800, 51.45300, 24.0),
    (111, "Brent Neasden", -0.21900, 51.55200, 31.0),
    (112, "Richmond Park", -0.27300, 51.45200, 18.0),
    (113, "Lambeth Brixton", -0.11300, 51.46300, 52.0),
    (114, "Wandsworth", -0.18800, 51.45600, 42.0),
]


def write_sample_csv(path: str, datetime_iso: str = "2024-07-15T12:00:00Z") -> str:
    """Write the committed reproducible NO2 snapshot (no network)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["sensor_id", "lon", "lat", "datetime", "parameter", "value", "unit"]
        )
        for sid, _name, lon, lat, val in _LONDON_SITES:
            w.writerow([sid, lon, lat, datetime_iso, "no2", val, "ug/m3"])
    return path


def fetch_openaq(
    city: str,
    pollutant: str,
    datetime_from: str,
    datetime_to: str,
    api_key: str,
    bbox=None,
    debug: bool = False,
):
    """Pull a one-window snapshot from OpenAQ v3 and return a list of row dicts.

    Requires ``requests`` and network access to api.openaq.org. Queries
    ``/v3/locations`` within the bbox, then the per-sensor ``/measurements`` (or
    ``/hours``) endpoint for the window, and flattens to the CSV schema.
    """
    import requests  # local import: only needed on the live path.

    if bbox is None:
        bbox = CITY_BBOX[city]
    headers = {"X-API-Key": api_key}
    base = "https://api.openaq.org/v3"
    pid = PARAM_ID[pollutant]

    lon_min, lat_min, lon_max, lat_max = bbox
    loc_params = {
        "bbox": f"{lon_min},{lat_min},{lon_max},{lat_max}",
        "parameters_id": pid,
        "limit": 1000,
    }
    locs = requests.get(
        f"{base}/locations", headers=headers, params=loc_params, timeout=30
    ).json()["results"]
    print(
        f"  [openaq] bbox={loc_params['bbox']} param={pollutant}({pid}): "
        f"{len(locs)} locations",
        file=sys.stderr,
    )
    if debug and locs:
        import json as _json

        print(
            "  [openaq] sample location:\n" + _json.dumps(locs[0], indent=2)[:1500],
            file=sys.stderr,
        )

    rows = []
    n_sensors = 0
    for loc in locs:
        lon, lat = loc["coordinates"]["longitude"], loc["coordinates"]["latitude"]
        for sensor in loc.get("sensors", []):
            if sensor["parameter"]["id"] != pid:
                continue
            n_sensors += 1
            sid = sensor["id"]
            mparams = {
                "datetime_from": datetime_from,
                "datetime_to": datetime_to,
                "limit": 1000,
            }
            meas = (
                requests.get(
                    f"{base}/sensors/{sid}/hours",
                    headers=headers,
                    params=mparams,
                    timeout=30,
                )
                .json()
                .get("results", [])
            )
            for mrec in meas:
                rows.append(
                    dict(
                        sensor_id=sid,
                        lon=lon,
                        lat=lat,
                        datetime=mrec["period"]["datetimeFrom"]["utc"],
                        parameter=pollutant,
                        value=mrec["value"],
                        unit=sensor["parameter"]["units"],
                    )
                )
    print(
        f"  [openaq] {n_sensors} matching sensors, {len(rows)} measurements in "
        f"[{datetime_from}, {datetime_to}]",
        file=sys.stderr,
    )
    if n_sensors > 0 and not rows:
        print(
            "  [openaq] sensors exist but no hourly data in that window -- "
            "widen the time window or pick a more recent UTC hour.",
            file=sys.stderr,
        )
    return rows


def write_rows_csv(path: str, rows) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = ["sensor_id", "lon", "lat", "datetime", "parameter", "value", "unit"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})
    return path


def load_csv(path: str):
    """Round-trip read of a snapshot CSV -> list of dict rows."""
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _here() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sample",
        action="store_true",
        help="write the committed reproducible snapshot (no network)",
    )
    ap.add_argument(
        "--city",
        default="london",
        help=f"named bbox: {', '.join(CITY_BBOX)} (ignored if --bbox given)",
    )
    ap.add_argument(
        "--bbox",
        default=None,
        help="override bbox as 'lon_min,lat_min,lon_max,lat_max'",
    )
    ap.add_argument("--pollutant", default="no2", choices=list(PARAM_ID))
    ap.add_argument("--datetime-from", default="2024-07-15T12:00:00Z")
    ap.add_argument("--datetime-to", default="2024-07-15T13:00:00Z")
    ap.add_argument(
        "--debug",
        action="store_true",
        help="print location/sensor counts and a sample API record",
    )
    args = ap.parse_args(argv)

    bbox = None
    if args.bbox:
        bbox = tuple(float(t) for t in args.bbox.split(","))
        if len(bbox) != 4:
            sys.exit("--bbox must be 'lon_min,lat_min,lon_max,lat_max'.")

    date_tag = args.datetime_from[:10]
    out = os.path.join(
        _here(), "data", f"openaq_{args.city}_{args.pollutant}_{date_tag}.csv"
    )

    if args.sample:
        path = write_sample_csv(out, datetime_iso=args.datetime_from)
        rows = load_csv(path)
        assert rows and rows[0]["parameter"] == "no2"
        # Round-trip a known sensor.
        m = {int(r["sensor_id"]): r for r in rows}
        assert 101 in m and abs(float(m[101]["value"]) - 86.0) < 1e-9
        assert abs(float(m[101]["lon"]) - (-0.15464)) < 1e-6
        print(f"wrote sample snapshot: {path}  ({len(rows)} sensors)")
        print("OpenAQ loader round-trip OK (sensor 101 Marylebone Road).")
        return

    api_key = os.environ.get("OPENAQ_API_KEY")
    if not api_key:
        sys.exit("set OPENAQ_API_KEY for a live pull, or use --sample.")
    rows = fetch_openaq(
        args.city,
        args.pollutant,
        args.datetime_from,
        args.datetime_to,
        api_key,
        bbox=bbox,
        debug=args.debug,
    )
    path = write_rows_csv(out, rows)
    print(f"wrote {len(rows)} rows: {path}")


if __name__ == "__main__":
    main()
