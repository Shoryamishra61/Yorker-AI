# Data Catalog

**Date:** April 24, 2026  
**Status:** Draft  

## Purpose

This catalog tracks every dataset needed for a credible IPL match prediction research project. Each source must have provenance, schema notes, refresh cadence, and modeling usage.

## Source Inventory

| Dataset | Current Status | Source | Modeling Use | Priority |
|---|---|---|---|---|
| Ball-by-ball IPL history | present | Cricsheet IPL JSON | deliveries, wickets, phases, player participation | P0 |
| Match metadata | present | Cricsheet JSON | teams, venue, toss, result, dates | P0 |
| Official schedule | present | IPLT20 schedule feed | upcoming fixtures and venues | P0 |
| Player registry | partial | Cricsheet people IDs | stable player identity | P0 |
| Playing XI | partial/inferred | scorecards / match JSON | lineup strength and availability | P1 |
| Squad lists | missing/partial | official IPL/team announcements | role and roster context | P1 |
| Auction data | missing/partial | BCCI/IPL auction releases | price, retention, roster changes | P1 |
| Venue metadata | partial | historical matches + manual metadata | venue priors, scoring, chase rates | P1 |
| Weather | missing | forecast/archive API | dew, humidity, rain, wind | P2 |
| Pitch reports | missing | official/broadcast/media text | pitch type and surface notes | P2 |
| Market odds | missing | legal historical odds provider | external baseline | P3 |

## Required Quality Checks

- match count by season
- duplicate match IDs
- duplicate fixture rows
- missing or unknown player IDs
- inconsistent team names
- inconsistent venue names
- missing toss fields
- missing innings data
- no-result and abandoned match handling
- super-over handling
- train/validation date ordering

## Data Lineage Target

Each processed field should trace back to:

1. raw file or API source
2. ingestion script
3. normalization rule
4. feature-generation function
5. availability time

## Availability Classes

| Class | Definition | Example |
|---|---|---|
| Pre-season | known before tournament starts | squads, auction price, historical venue priors |
| Match-eve | known before match day or hours before toss | fixture, venue, weather forecast, likely XI |
| Toss-confirmed | known after toss and team sheet | toss decision, confirmed XI |
| Post-match | known only after play | score, wickets, result, final player stats |

The main pre-match model must use only pre-season and match-eve features. A separate toss-confirmed model may use toss and confirmed XI.

## Next Additions

- Convert this draft into a generated report.
- Add row-count tables from actual processed datasets.
- Add source license notes.
- Add schema definitions for each processed table.
- Add missingness summaries by season.
