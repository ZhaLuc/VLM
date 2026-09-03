# Hidden-state video sourcing guide

The repository currently has **zero** clips that pass
`docs/HIDDEN_STATE_ELIGIBILITY.md`. The five Wikimedia / PeerJ cups-and-balls
files are transparent-cup stage clips with a late reveal. Keep them as
pilot/control footage. Do not force them into hidden-state gold.

This guide is for obtaining the **next** videos. It is not a request to
fabricate labels.

Target for a first held-out-only smoke baseline: **5 clips** that pass the
rubric (3 is the minimum if sourcing is slow; 15–25 remains the later Dataset B
goal in `magic-vlm-research-plan-v2.md` and is not the next step while the
count is zero).

## Suitable trick categories

Task structure, not a script to fake a dataset:

- object placed under one of several **opaque** cups
- object transferred between **closed** hands, then a question about which hand
- object placed inside one of several **opaque** boxes/containers
- shell-game / three-cup tracking with genuinely opaque cups
- card or small object concealed in a known location while later actions stay
  visible
- simple load/unload under an opaque cover, evaluation **before** any lift

Prefer one clear concealment per clip.

## Suitable visual structure

```
object location shown
        ↓
opaque cover / closed hand / closed box
        ↓
one or more visible actions that do not re-expose the answer
        ↓
stop (or crop) before the reveal
        ↓
ask which container/hand/location holds the object
```

Ideal clip: 6–20 seconds (acceptable 5–30). Static or barely moving camera.
Tabletop or close-up. Full object and hands in frame. Evaluation window ends
while the state is still hidden.

## Unsuitable structures

- transparent cups, glass, mesh, or any container you can see through
- the scored clip includes the cup-lift / hand-open reveal
- the answer is readable on the last frame
- titles, captions, or filenames that name the location
- wide stage shots where the load is off-camera with no prior cue
- jumpy edits, magic-special TV cutting, or obvious cheat zooms
- audio that announces the answer
- “what happened in this trick?” with no target state

The existing PeerJ supplements (s003–s008 on Commons) are the **transparent**
routine variants. Do not treat “Cups and Balls” + CC BY as sufficient.

The 2013 paper also crossed **opaque cups** with load/no-load and face/no-face.
Those experimental conditions are **not** in this repository and were **not**
found as separate Wikimedia files (Commons search 2026-09-03 returned only
s003–s008). If you later obtain opaque-cup files, inspect frames; do not
qualify them from the paper title.

## Camera setup

- one locked-off camera, eye-level or slightly above the table
- do not shoot through glass
- do not include a mirror, second angle, or reflection that shows the load
- do not crop so tightly that the cover is off-screen (that is uninformative,
  not hidden-state)
- lighting should not silhouette the object inside a “opaque” cup

## Object / container types

**Prefer:** opaque plastic/metal cups, opaque boxes, closed fists, opaque
cups with no see-through rim at the evaluation frame.

**Avoid:** clear acrylic cups (the current Wikimedia set), wine glasses,
see-through bags.

Use high-contrast objects (e.g. a dark ball on a light table) so tracking
before concealment is possible.

## Ideal duration

| Role | Length |
| ---- | ------ |
| First baseline clip | 6–20 s |
| Acceptable | 5–30 s |
| Too short | under ~4 s with no pre-concealment view |
| Too long | full routines; trim to one concealment |

If the source includes a reveal, **crop or set `temporal.end_s` before the
reveal**. Keep the untrimmed original for provenance.

## Licensing / provenance

### Preferred

Openly licensed research or public-domain footage with a clear page:

- Wikimedia Commons (verify the **file** license, not a category)
- PeerJ / PLOS / journal supplements with an explicit CC license
- Zenodo / Figshare / OSF deposits with CC BY / CC0
- footage you film yourself and own

Self-filmed Dataset B is the most reliable way to know the hidden state.

### Acceptable with review

- author-posted video whose page states a reuse license you can keep
- recordings you have **written permission** to use for this research
- university-owned lab recordings with documented rights

“I need it for research” does **not** make a copyrighted upload reusable.

### Avoid

- random YouTube / TikTok / Instagram performances
- TV recordings of Penn & Teller or other copyrighted acts
- files whose license cannot be named
- unlabeled screen recordings

## What to verify **before** downloading

1. License text on the source page (name + URL).
2. Preview: containers look **opaque**, not glass.
3. You can name a specific question and an independently known answer.
4. You can exclude the reveal from the evaluation window.
5. Camera does not leak the load.
6. Duration is usable (or you can trim).
7. You will store the original filename and URL.

Do not bulk-download. Inspect one candidate at a time.

## Metadata to preserve (do not invent)

For every file you keep:

- source URL
- title
- author / performer when known
- license (short name + license URL)
- date accessed
- original filename
- local filename
- conversion command if you transcode
- source notes (what the page actually says)
- ground-truth source (how **you** know the answer: you placed the ball;
  a documented protocol; a reveal that you **excluded** from the eval clip)

If a field is unknown, write `unknown` — do not guess.

Copy new files into `data/videos/` (gitignored). Add a provenance JSON under
`data/provenance/`. Do **not** write `ground_truth` into a research manifest
until the clip passes the rubric and you approve the question.

## Practical search queries

Preferred first:

```
site:commons.wikimedia.org cups and balls ogv CC BY -Penn
site:peerj.com supplemental movie magic trick
site:zenodo.org "cups and balls" CC-BY video
site:osf.io magic trick video CC-BY
```

Journal supplements:

```
"cups and balls" opaque supplemental video filetype:mp4
"three-shell game" research video Creative Commons
```

Self-film if search stays empty. That is expected: openly licensed **opaque**
concealment clips are uncommon. The current Commons PeerJ set is the
transparent demonstration series, not the hidden-state task.

## After a clip is on disk

1. Convert if needed (`scripts/convert_ogv_to_mp4.py`); keep the original.
2. Probe with `magic-vlm-sample-frames --load-frames --max-frames 8`.
3. Score A–F in the candidate inventory. Status starts as
   `QUALIFIES_WITH_HUMAN_REVIEW` at best.
4. Open `reports/hidden_state_candidates/index.html` after updating the
   inventory JSON.
5. Only then consider a pending annotation — still not gold.
