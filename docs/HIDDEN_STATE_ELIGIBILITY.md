# Hidden-state eligibility rubric

Research question: given a short magic or mentalism demonstration and a
targeted question, can a VLM infer a hidden state that cannot simply be read
from the final frame or an obvious visual reveal?

A valid example must use temporal visual information. The intended structure is:

```
Relevant state visible
        ↓
State becomes genuinely occluded/hidden
        ↓
Relevant action/event occurs
        ↓
Question asks for hidden state
        ↓
Answer must be inferred from the preceding visual sequence
```

This rubric is for **candidate qualification**. Passing it does not write
`ground_truth` into a research manifest. Gold labels require a later human
approval step.

Statuses used in the inventory:

| Status | Meaning |
| ------ | ------- |
| `QUALIFIES` | Local frames inspected; A–F all pass. Still do not write gold until approved. |
| `QUALIFIES_WITH_HUMAN_REVIEW` | Looks eligible, but occlusion, ground truth, or leakage needs a human decision. |
| `NOT_SUITABLE` | Inspected and fails at least one hard check. |
| `INSUFFICIENT_EVIDENCE` | Cannot decide without inspecting frames or a trustworthy source. |
| `MISSING_FROM_REPOSITORY` | Documented elsewhere; not on local disk. |

Do not award `QUALIFIES` from a filename, Commons caption, or paper title alone.

## Hard checks (A–F)

All six must pass for `QUALIFIES`.

### A. Occlusion

Is the queried object/state genuinely hidden at the evaluation point?

**Pass examples:** opaque cup; closed hand; opaque box; closed container;
object leaves the frame only if the task still makes the state inferable from
what was shown before.

**Fail examples:** transparent cup; object plainly visible; object visible in
the final evaluation frame; answer exposed by a reveal included before the
question point.

Check: pause at the intended evaluation time. If a naive viewer can read the
answer off that frame without the earlier sequence, fail.

### B. Temporal dependence

Would the question become substantially harder if the order of relevant events
were destroyed?

The clip should contain meaningful temporal information. Shuffle sensitivity
does not have to be proven at annotation time. Record whether temporal order
plausibly matters (`YES`, `WEAK`, or `NO`).

**Fail:** the answer is available in a single still (opening frame, final
frame, or a title card).

### C. Ground-truth establishment

Can the correct answer be independently established?

**Accept:** controlled experiment with known state; documented source
description that unambiguously establishes state; a temporary reveal elsewhere
in the source **if the evaluation clip itself does not include that reveal
before the question point**; researcher-controlled footage with trustworthy
provenance.

**Reject / flag:** guesswork; ambiguous performance footage; unclear
mechanism; unsupported claims from titles or captions.

Record the ground-truth source explicitly. If it is only a Commons caption,
that is not enough.

### D. No answer leakage

Fail if any of these expose the answer before evaluation:

- final reveal included in the scored clip
- transparent props
- object visible through a container
- captions/subtitles that state the answer
- obvious editing cues that mark the answer
- filename or title leakage
- metadata leakage
- camera angle that exposes the state

### E. Questionability

The clip must support a specific answerable question.

**Bad:** “What happened in this trick?”

**Better:** “Which cup contains the ball at the concealment point?” — only if
the state is genuinely hidden and the answer is independently known.

### F. Annotation reproducibility

A second researcher should be able to understand why the answer is considered
correct: which frames were used, what was occluded, and where the ground truth
came from.

## Control uses (not hidden-state gold)

Clips that fail A or D may still be kept as:

- `VISIBLE_EVENT_CONTROL` — a visible action or location can be named
- `TEMPORAL_CONTROL` — ordered events useful for shuffle/decode tooling
- `PIPELINE_SMOKE` — decode, sampling, preprocess checks
- `NOT_SUITABLE_FOR_HIDDEN_STATE` — must not enter the hidden-state gold set

Pilot/control footage stays in the repository. It is not relabeled as
`hidden_state` gold to make the dataset look complete.
