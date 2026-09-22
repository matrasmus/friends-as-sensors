# Friends as Sensors

Code and data preparation for the paper "Friends as Sensors. High-Sensitivity Monitoring of Misinformation and Platform Dynamics via the Friendship Paradox".

The paper studies whether users selected through the friendship paradox act as early and sensitive sensors for misinformation and platform level exposure dynamics on Twitter/X. It compares a sensor sample of 9,121 users against a random sample of 9,741 users over June to October 2024.

## Contents

- `reproduce_analysis_and_figures.ipynb` reproduces every analysis figure of the paper from the published datasets. Place the two dehydrated parquet files and the three lead time CSVs in `data/` and run it top to bottom. Figure 1 of the paper is a schematic illustration of the friendship paradox without any analysis behind it and is therefore not part of the notebook.
- `dehydrate.ipynb` builds the publishable datasets from the private raw parquets. It removes all text and profile fields, removes all URL and domain columns, and replaces every platform identifier with a salted SHA-256 hash. Each step is explained in the notebook. It only runs internally where the raw data lives.
- `reproduce_paper_figures.py` holds the statistical helpers the figure notebook imports.
- `DATASHEET.md` documents every column of the published parquet files.

## Why the data looks like this

The published datasets contain no tweet text, no user profiles, no URLs and no platform identifiers. Every design decision follows from one of three constraints.

**No re-identification.** Tweet texts, user names, profile descriptions and locations identify people, so they are removed entirely. All platform identifiers are replaced by salted SHA-256 hashes. The salt is stored privately with the raw data and is not part of this repository or the data publication, so the hashes are one way and the rows cannot be rehydrated through the X API. The hashes are consistent within and across both files, which keeps retweet chains and per user groupings intact for analysis.

**No reconstruction of the NewsGuard rating table.** NewsGuard scores are licensed and deterministic per domain. Publishing domains next to per tweet scores would let anyone rebuild the licensed domain table from a few million rows. The URL and domain columns are therefore removed and only the per tweet score values remain. URLs survive solely as salted hashes in `urls_hashed`, which lets anyone verify that two tweets share the same link, as the lead time analysis requires, without revealing which link it is.

**Data minimization.** API metadata and model scores that no analysis in the paper uses are dropped.

The result is that every number and every analysis figure of the paper can be recomputed from the published files alone. The one exception worth naming is the lead time pair matching, which can be recomputed via the hashed URLs, while the mapping of pairs to real links stays private.

## Author

Mathias Angermaier, University of Graz
