# Friends as Sensors

Code and data preparation for the paper "Friends as Sensors. High-Sensitivity Monitoring of Misinformation and Platform Dynamics via the Friendship Paradox".

The paper studies whether users selected through the friendship paradox act as early and sensitive sensors for misinformation and platform level exposure dynamics on Twitter/X. It compares a sensor sample of 9,121 users against a random sample of 9,741 users over June to October 2024.

## Contents

- `dehydrate.py` builds the publishable datasets from the private raw parquets. It removes all text and profile fields, removes all URL and domain columns, and replaces every platform identifier with a salted SHA-256 hash. See the module docstring for the reasoning behind each step.
- `DATASHEET.md` documents every column of the published parquet files.

## Data availability

The published datasets contain no tweet text, no user profiles, no URLs and no platform identifiers. Identifiers are one way hashes, so the rows cannot be rehydrated through the X API. NewsGuard scores appear only as per tweet values without the domains they belong to, so the proprietary NewsGuard domain table cannot be reconstructed. The salt used for hashing is stored privately with the raw data and is not part of this repository or the data publication.

## Author

Mathias Angermaier, University of Graz
