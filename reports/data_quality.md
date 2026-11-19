# Data-quality report

- Raw rows: 737,453
- Date range: 2017-03-10 01:00:00 to 2017-09-09 23:00:00
- Hours in range: 4,415; with data: 4,097; missing: 318
- Unparseable values: 0

## Gaps (consecutive missing hours)

| start | end | hours |
|---|---|---|
| 2017-03-16 06:00:00 | 2017-03-29 11:00:00 | 318 |

## Hours with an unusual row count (expected 180)

| date | n_rows |
|---|---|
| 2017-03-10 01:00:00 | 174 |
| 2017-04-10 00:00:00 | 179 |

## Out-of-range raw values

None.

## Lab columns that vary within an hour

Hours where a lab value is not constant across its 20 s rows (sign of interpolation; we keep the first value).

- pct_iron_feed: 0 hours
- pct_silica_feed: 0 hours
- pct_iron_concentrate: 216 hours
- pct_silica_concentrate: 310 hours
