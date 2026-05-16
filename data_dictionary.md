# Data Dictionary: banana_panel.csv

| Variable        | Definition | Unit | Source | Transformation Notes |
|-----------------|------------|------|--------|----------------------|
| Province        | Philippine province name | Text | PSA | Mapped to ISO 3166-2:PH |
| Year            | Calendar year | Integer | PSA | 2010–2022 |
| lon             | Provincial centroid longitude | Decimal degrees | PSA/GeoNames | WGS84 |
| lat             | Provincial centroid latitude | Decimal degrees | PSA/GeoNames | WGS84 |
| SSE             | Source-Sink Efficiency (HI proxy) | Dimensionless | Derived | Yield / AGB_adj |
| TS              | Temperature Seasonality | °C | DOST-PAGASA | SD of monthly mean temp |
| CWDD            | Cumulative Water Deficit Days | Days yr⁻¹ | DOST-PAGASA + FAO-56 | Days below field capacity during flowering-to-bunch-initiation |
| CEI_LuzVis      | Cyclone Exposure Index (Luzon/Visayas) | Events yr⁻¹ | PAGASA | Cyclones within 200km, winds ≥62 km/h |
| CEI_Mind        | Cyclone Exposure Index (Mindanao) | Events yr⁻¹ | PAGASA | Same as above, Mindanao subset |
| DPI             | Disease Pressure Index | 0–1 | DA-NPQD (FOI) | Weighted composite (Foc-TR4 > BBTV > Sigatoka) |
| Fertiliser      | NPK application rate | kg ha⁻¹ | PSA | Provincial average |
| GAP             | Good Agricultural Practice adoption | % | PSA | Provincial coverage |
| PhenologyDelay  | Days delay in flowering/bunch initiation | Days | Derived proxy | Based on TS/CWDD stress accumulation |
| LeafArea        | Effective leaf area proxy | Dimensionless index | Derived proxy | Based on DPI/TS canopy stress indicators |

**Note:** SSE is a harvest-index proxy. AGB was climate-adjusted using FAO-56 stress multipliers to decouple from yield scaling. All variables are harmonised to ISO 19115 spatial standards.