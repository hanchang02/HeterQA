# HeterQA Project Page

Static project page for the HeterQA benchmark. The page is designed for GitHub Pages or any plain static host and does not require a build step.

## Preview

```bash
python3 -m http.server 8090
```

Then open:

```text
http://127.0.0.1:8090/
```

If you serve this repository from a subdirectory, all links and assets remain relative.

## Contents

- `index.html`: single-page static site.
- `static/css/index.css`: layout and visual styling.
- `static/images/`: converted paper figures and favicon.

## Source Facts

The page uses the locked HeterQA paper facts:

- 857 QA pairs.
- Five evidence sources.
- Ten source-composition subsets.
- Average query length 22.2 words.
- Average verified answer set size 2.3 records.
- Shannon Entropy 6.712 and Type-Token Ratio 0.204.
- Best Recall@10 32.78.
- Best MRR@10 25.26.

The release description follows the dataset card: HeterQA publishes annotations, answer identifiers, qrels, schemas, and structured evidence summaries, but not original Yelp reviews, photos, full business records, business addresses, user data, prompt traces, or local paths.
