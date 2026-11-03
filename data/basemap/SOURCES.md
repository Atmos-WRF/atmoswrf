# Sources of the viewer's geography

Everything here is in the public domain, so it can be shipped, changed and sold
without conditions.  `data/build_geodata.py` rebuilds each file from the downloads
listed below (see its docstring for the command).

## Map lines: `data/basemap/basemap.npz`

Every line of each layer, as integer 1e-5 degree lon/lat steps (numpy
`savez_compressed`).  Vertices closer than 0.1 km to the previous one are
dropped; the map thins to 0.35-0.8 km when it draws anyway.  5.4 MB, against
about 22 MB of shapefiles.

| Layer      | Source file                                  | Download |
|------------|----------------------------------------------|----------|
| `counties` | US Census 2023 cartographic boundaries, 1:5,000,000 counties (`cb_2023_us_county_5m`) | https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_5m.zip |
| `states`   | Natural Earth 10m admin-1 states and provinces lines (`ne_10m_admin_1_states_provinces_lines`) | https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_1_states_provinces_lines.zip |
| `borders`  | Natural Earth 10m admin-0 land boundaries, i.e. country borders (`ne_10m_admin_0_boundary_lines_land`) | https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_boundary_lines_land.zip |
| `coast`    | Natural Earth 10m coastline (`ne_10m_coastline`) | https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_coastline.zip |
| `lakes`    | Natural Earth 10m lakes (`ne_10m_lakes`)       | https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_lakes.zip |

- Natural Earth (https://www.naturalearthdata.com/): public domain, per its terms
  of use (https://www.naturalearthdata.com/about/terms-of-use/).
- US Census Bureau cartographic boundary files
  (https://www.census.gov/geographies/mapping-files/time-series/geo/cartographic-boundary.html):
  a work of the US Government, not subject to copyright (17 U.S.C. 105).

The 2026-10-20 build used these same files unmodified, taken from ArWen Desktop
1.0.10's `resources/assets/basemap` (whose READMEs give the URLs above), not
downloaded again.

## Radar sites: `data/radars.csv`

`icao, type, lat, lon, elev_m, state, name`; type 1 = WSR-88D (NEXRAD), 3 = TDWR.
163 WSR-88D and 47 TDWR sites.  Elevation is the station elevation in the list,
converted from feet.

- Source: NOAA NCEI Historical Observing Metadata Repository (HOMR), NEXRAD and
  TDWR station list, https://www.ncei.noaa.gov/access/homr/file/nexrad-stations.txt
  (file dated 2026-10-01, downloaded 2026-10-20).
- License: a work of the US Government (NOAA), not subject to copyright
  (17 U.S.C. 105).

## Cities: `data/cities.csv`

`name, state, lat, lon, pop`: every US incorporated place (city, town, village,
borough) at its Gazetteer internal point, with its Vintage 2025 population
estimate; 19,483 places, 4,901 of 5,000 or more.  Census designated places
(unincorporated communities such as Metairie, Hawaii apart from Honolulu, and
Puerto Rico) have no population estimate and are not included.  Names drop the
legal description ("Abbeville city" becomes "Abbeville", "Nashville-Davidson
metropolitan government (balance)" becomes "Nashville", "San Buenaventura
(Ventura)" becomes "Ventura").

- Locations: US Census Bureau 2025 Gazetteer, places,
  https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2025_Gazetteer/2025_Gaz_place_national.zip
  (downloaded 2026-10-20).
- Populations: US Census Bureau Vintage 2025 population estimates, city and town
  totals (`SUMLEV` 162 rows, `POPESTIMATE2025`),
  https://www2.census.gov/programs-surveys/popest/datasets/2020-2025/cities/totals/sub-est2025.csv
  (downloaded 2026-10-20).
- License: works of the US Government, not subject to copyright (17 U.S.C. 105).
