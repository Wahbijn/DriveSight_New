# Provenance check: `data/bdd100k/labels/100k/test`

**Question:** BDD100K's official detection challenge withholds test-set ground truth
(it's only scored via a hidden leaderboard), yet this repo has 20,000 fully-annotated
JSON files under `labels/100k/test`. Is this a bug, a mislabeled/duplicated folder, or
genuine data?

## Checks performed (read-only, this session)

1. **Schema/content check** — sampled 5 random files from `labels/100k/test`. Each has
   the same `{"name", "frames", "attributes"}` structure as `train`/`val`, with real
   `box2d` coordinates, per-object `occluded`/`truncated`/`trafficLightColor`
   attributes, and per-image `weather`/`scene`/`timeofday` attributes. Not placeholders,
   not empty, not malformed.

2. **Disjointness check** — compared the `name` stem of every file across all three
   label folders:

   | | train (70,000) | val (10,000) | test (20,000) |
   |---|---:|---:|---:|
   | overlap with train | — | 0 | 0 |
   | overlap with val | 0 | — | 0 |
   | overlap with test | 0 | 0 | — |

   Zero overlap in every pairwise comparison — `test` is not a renamed/duplicated copy
   of `train` or `val`.

3. **Statistical plausibility check** — sampled 1,000 files per split and tallied the
   `weather`/`timeofday` attribute distributions:

   | split | clear | overcast | rainy | snowy | partly cloudy | undefined | foggy |
   |---|---:|---:|---:|---:|---:|---:|---:|
   | train | 53.6% | 13.9% | 5.4% | 9.1% | 6.8% | 10.9% | 0.3% |
   | val | 54.6% | 12.7% | 5.9% | 8.7% | 7.1% | 10.8% | 0.2% |
   | test | 54.6% | 11.0% | 7.2% | 8.2% | 6.6% | 12.3% | 0.1% |

   Distributions are consistent across all three splits (within ordinary sampling
   noise) — `test` looks like a genuine independent sample from the same underlying
   corpus, not synthetic or cherry-picked data.

4. **File-timestamp check** — the clincher. Sampled mtimes (these are archive-extraction
   timestamps preserved from the original zip, not download dates, but they're
   internally consistent and informative):

   | path | sampled mtime range |
   |---|---|
   | `labels/100k/train` | 2018-03-13 06:32:34 – 06:35:09 |
   | `labels/100k/val` | 2018-03-13 06:47:59 – 06:48:19 |
   | `labels/100k/test` | 2018-03-13 **13:13:39 – 13:14:32** |
   | `images/10k/train` | 2018-04-05 19:07:44 – 19:07:53 |
   | `images/10k/val` | 2018-04-05 19:07:55 – 19:07:57 |
   | `images/10k/test` | 2018-04-05 19:07:53 – 19:07:55 |

   All three **label** splits share the same extraction day (2018-03-13); train/val are
   within the same 16-minute window and test follows a few hours later the same day —
   consistent with one coherent label package extracted in (at most) two sittings, not
   three separately-sourced files stitched together on different dates. All three
   **image** folders separately cluster on 2018-04-05, ~19:07 — a different, single
   extraction event, confirming (independently of the earlier matching-rate analysis)
   that the images (`10k` package) and labels (`100k` package) really are two distinct
   original BDD100K downloads, consistent with root cause #1 of the diagnosis.

## Conclusion

The test labels are very likely genuine, unmodified BDD100K annotations from the
dataset's **original 2018 public release** (the JSON schema — `frames`/`attributes`/
`box2d` with `occluded`/`truncated`/`trafficLightColor` — matches the 2018
`bdd100k_labels_release` format, not the later `det_20` format used for the CVPR2020
challenge). BDD100K's policy of withholding test-set ground truth for the public
leaderboard was introduced with that later (2020) challenge restructuring — it's
plausible the original 2018 release simply shipped all three partitions' annotations
before that policy existed.

**Practical takeaway:** you can trust `test` (830 images, after matching) as a genuine,
video-disjoint third holdout *internal to this project*. What you should **not** do is
treat any number computed on it as comparable to BDD100K's official hidden-leaderboard
test score — this is not that file, it just happens to use the same name.

**Residual uncertainty:** the exact original download URL/mirror could not be
determined from inside the repo (no manifest/README records it). If this matters for a
publication or compliance reason, it's worth tracking down explicitly — but it does not
block using the test split as an internal holdout.
