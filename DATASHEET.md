# Datasheet for the dehydrated datasets

Two files, one row per tweet.

- `sensor_dehydrated.parquet` covers 30,964,074 tweets from 9,121 sensor users collected June 1 to October 31, 2024.
- `random_dehydrated.parquet` covers 27,395,048 tweets from 9,741 randomly sampled users with timelines from January 1, 2015 onwards.

## Identifier columns

All identifiers are salted SHA-256 hashes truncated to 16 hex characters. The same input value always maps to the same hash within and across both files, so retweet chains and per user groupings are preserved. The salt is private, so the hashes cannot be reversed and the rows cannot be rehydrated through the X API.

| Column | Description |
|---|---|
| `TWEET_id_hashed` | Hash of the tweet id. |
| `USER_id_hashed` | Hash of the posting user id. |
| `referenced_tweet_id_hashed` | Hash of the referenced tweet id for retweets, quotes and replies. Empty otherwise. |
| `conversation_id_hashed` | Hash of the conversation root tweet id. |

## Tweet columns

| Column | Description |
|---|---|
| `referenced_tweet_type` | One of retweeted, quoted, replied_to, or empty for original tweets. |
| `TWEET_created_at` | Tweet timestamp as delivered by the API, UTC. |
| `lang` | Language code as delivered by the API. |
| `TWEET_like_count` | Like count at collection time. |
| `retweet_count` | Retweet count at collection time. |
| `quote_count` | Quote count at collection time. |
| `bookmark_count` | Bookmark count at collection time. |
| `reply_count` | Reply count at collection time. |
| `impression_count` | Impression count at collection time. |
| `newsguard_scores` | List of NewsGuard reliability scores, one per news link in the tweet. Empty when the tweet contains no rated link. The domains themselves are not included. |
| `newsguard_orientation` | List of NewsGuard political orientation labels, aligned with `newsguard_scores`. |

## User columns

Repeated on every row of the posting user.

| Column | Description |
|---|---|
| `USER_followers_count` | Follower count at collection time. |
| `USER_following_count` | Following count at collection time. |
| `USER_tweet_count` | Lifetime tweet count at collection time. |
| `USER_listed_count` | Listed count at collection time. |
| `USER_verified` | Verified flag at collection time. |
| `USER_created_at` | Account creation timestamp. |

## What was removed and why

Tweet text, user names, user descriptions, locations, profile links and mentions were removed because they identify people. URLs and domains were removed because together with the per tweet NewsGuard scores they would allow reconstruction of the licensed NewsGuard domain table. Remaining API metadata without use in the paper was removed for data minimization.
