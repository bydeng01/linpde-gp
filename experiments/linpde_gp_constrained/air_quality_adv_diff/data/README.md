# Air-quality monitor data

Hourly ground-monitor readings in the format read by `../run.py --data` and
`../sweep_real.py`: `sensor_id, lon, lat, datetime, parameter, value, unit`.

## OpenAQ snapshots

These four snapshots form the pooled real-data analysis; they are the entries
of `../snapshots.txt`.

| File | Pollutant | Date | Monitors |
|---|---|---|---|
| `openaq_maryland_pm25_2021-07-15.csv` | PM2.5 | 2021-07-15 | 22 |
| `openaq_maryland_pm25_2023-07-20.csv` | PM2.5 | 2023-07-20 | 31 |
| `openaq_nyc_pm25_2023-07-20.csv` | PM2.5 | 2023-07-20 | 12 |
| `openaq_maryland_no2_2024-06-20.csv` | NO2 | 2024-06-20 | 13 |

They were downloaded from the [OpenAQ](https://openaq.org) v3 API with
`../fetch_data.py`; re-downloading needs a free OpenAQ API key. OpenAQ
aggregates measurements from the original data providers, and the OpenAQ API
lists the provider and licence of each location. If you reuse these data,
credit both OpenAQ and the original providers as their licences require.
Measurements that reach OpenAQ through the US EPA AirNow program are
preliminary and have not been fully validated.

## Wind

The wind vectors in `../snapshots.txt` are daytime (09:00-17:00 local) means of
the hourly 10 m wind from the [Open-Meteo](https://open-meteo.com/) historical
weather API at the centre of each snapshot's bounding box, converted from the
meteorological "from" direction to an advection vector. Weather data by
[Open-Meteo.com](https://open-meteo.com/), licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Offline sample (synthetic values)

`openaq_london_no2_2024-07-15.csv` is written by `../fetch_data.py --sample`. It
uses real central-London station coordinates with synthetic NO2
concentrations, so that `../reproduce.sh` can exercise the `--data` path
without network access or an API key. It is not a real measurement and is not
part of the pooled analysis.
