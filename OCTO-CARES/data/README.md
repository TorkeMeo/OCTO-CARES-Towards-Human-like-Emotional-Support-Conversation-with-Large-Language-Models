# Data Preparation and Release Boundary

This package contains code, not the Reddit post corpus, ESCoT data, API response
caches, learned model weights, or experiment outputs. No license for those
third-party materials is granted by publishing this code. Obtain each dataset
through its authorized source and comply with its terms, including removal
requests and redistribution restrictions. A code license does not authorize
redistribution of dataset text.

Use a local, untracked directory such as `data/private/` for real data. Do not
upload raw emotional-support posts or generated logs merely to make the
repository executable. API annotation outputs can contain post text,
usernames, URLs, comments, and source paths; review them before any release.
All examples below are synthetic and are not records from the study.

## Corpus Counts

| Name | Role | Recorded count |
| --- | --- | --- |
| Original Reddit corpus | Qwen3.7-labeled corpus used to fit the eight adapters | 1600 |
| Fixed provenance partitions | Full post-level exports from that corpus | 1440 + 160 |
| Supplement source | Additional posts before complete-label filtering | 1060 |
| Complete supplement | Three-model majority labels with incomplete rows removed | 1053 |

The original 1600 and the 1053 supplement are different corpora. The original
1440/160 markers must remain in training files for provenance, even when a
separate downstream protocol intentionally fits all 1600 records.

The recorded original export was constructed from 1520 labeled records plus an
80-post pool with Qwen labels. The 160-record partition combines 80 sampled
from the 1520 and the separate 80; it is not a human-label test set. The
package omits unused manual-label templates and their historical scripts.
The supplement was annotated by three API models separately for each of eight
labels. The recorded filtering leaves 1053 of 1060 posts; new API runs may
produce a different completeness pattern.

## Raw Annotation Input

`annotation/label_posts_bailian.py` and
`annotation/label_supplement_three_models_bailian.py` read a JSON array or a
JSON object with a `records` array. Each item should contain a stable post ID,
the source title, and body under `source_post`. The body key is `selftext`:

```json
[
  {
    "source_post": {
      "id": "synthetic_post_001",
      "subreddit": "synthetic_community",
      "title": "A fictional request for support",
      "selftext": "This fabricated post exists only to demonstrate the schema."
    },
    "selected_comment": {
      "id": "synthetic_comment_001",
      "body": "A fabricated paired response for a retrieval demonstration."
    }
  }
]
```

The paired comment is optional for annotation and is not injected into label
requests. It may be required by later post-comment generation conditions.
Missing IDs, duplicate IDs, or empty post text are not interchangeable with
valid examples; validate input before relying on expected corpus counts.

## Labeled Training and Retrieval Input

The canonical label order is defined by `annotation/post_labels.py`. Every
complete record has eight binary values; `label_vector` uses this exact order:

```text
positive_informational_self_disclosure
negative_informational_self_disclosure
neutral_informational_self_disclosure
positive_emotional_self_disclosure
negative_emotional_self_disclosure
seek_emotional_support
seek_informational_support
seek_companionship
```

A full-record training/retrieval JSON item has this shape:

```json
{
  "post_id": "synthetic_post_001",
  "subreddit": "synthetic_community",
  "title": "A fictional request for support",
  "body": "This fabricated post exists only to demonstrate the schema.",
  "text": "A fictional request for support\n\nThis fabricated post exists only to demonstrate the schema.",
  "labels": {
    "positive_informational_self_disclosure": 0,
    "negative_informational_self_disclosure": 1,
    "neutral_informational_self_disclosure": 0,
    "positive_emotional_self_disclosure": 0,
    "negative_emotional_self_disclosure": 1,
    "seek_emotional_support": 1,
    "seek_informational_support": 0,
    "seek_companionship": 0
  },
  "label_vector": [0, 1, 0, 0, 1, 1, 0, 0],
  "status": "complete",
  "fixed_split": "train"
}
```

These label values are illustrative, not gold annotations. The stable trainer
accepts text from `text`, with a `title`/`body` fallback, and validates the
counts of both `fixed_split=train` and `fixed_split=test`. Its default expected
counts are 1440 and 160. Smaller synthetic tests require explicit expected
count arguments and enough examples of both binary classes.

The single-model annotation cache does not contain all full-record source
fields. Join it to the raw corpus using `post_id` before training or retrieval.
The supplement majority cache uses `majority_labels`; export it with
`annotation/export_supplement_majority_eval_schema.py` to obtain the compact
`labels`/`label_vector` schema. Do not replace missing labels with zero.

## Suggested Local Layout

```text
data/private/
  reddit1600_source.json
  reddit1600_train1440.json
  reddit1600_test160.json
  reddit1600_labeled.json
  reddit1060_source.json
  reddit1053_labeled.json
  escot_test.json
outputs/
  annotation/
  adapters/
  vectors/
  retrieval/
  generation/
  evaluation/
```

Paths are examples, not bundled artifacts. Use CLI arguments to point at your
files rather than copying private server paths into scripts.

## Reproduction Records

Retain a private manifest containing corpus counts, label order, stable IDs,
fixed split IDs, file checksums, annotation prompt versions, model identifiers,
and missing-label/exclusion reports. A sample size alone cannot reproduce the
same subset. Without the original split and selection artifacts, describe a
newly sampled corpus as a new run rather than the exact published experiment.

Before publishing any IDs, manifests, or annotation data, verify that their
release is permitted and that they do not compromise author anonymity or
participant privacy. Real data, credentials, caches, checkpoints, and logs
belong outside the anonymous code release by default.
