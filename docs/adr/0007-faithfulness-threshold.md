# 7. Faithfulness floor: keep 0.70, with the measurement that could have moved it

Date: 2026-10-09 · Status: Accepted

## Context

The CI eval gate fails when the mean lexical faithfulness proxy over the golden
set drops below `faithfulness_threshold = 0.70`. The value was chosen by hand in
v0.2. The roadmap item was to derive it from the proxy's agreement with an LLM
judge, which [`eval-calibration.md`](../eval-calibration.md) now measures: 100
held-out generations over 22 questions, scored by the proxy, by `qwen3:32b`
(twice) and by `gemma4:31b`, all frozen in `data/eval/judge-scores.json`.

The rule for moving the floor was written after the first judge (`qwen3:32b`)
had been scored and before the second judge and the repeat run were: change it
only if

1. the judge is reliable enough to calibrate against: verdict kappa between two
   different judge models of at least 0.60;
2. the cutoff that best reproduces the judge's verdict (maximum Cohen kappa) is
   stable: its question-bootstrap 95% interval excludes 0.70 for **both**
   judges; and
3. the calibrated quantity is the gated quantity.

## Measurements

| Criterion | Measured | Met? |
|---|---|---|
| 1. judge reliability | qwen3:32b vs gemma4:31b: same verdict 83/100 [0.74, 0.89], kappa 0.603 | barely (0.603 against 0.60) |
| 2. stable cutoff | kappa-best cutoff 0.158 [0.06, 0.60] against qwen3:32b, 1.000 [0.29, 1.00] against gemma4:31b | no: the two judges pick opposite ends, and gemma's interval contains 0.70 |
| 3. same quantity | calibration: per-item verdicts on fine-tuned model answers; gate: a mean over the golden set of extractive answers (currently 0.96) | no |

Additional facts:

- At 0.70 the proxy's verdict agrees with the judges on 66/100 (kappa 0.36) and
  71/100 (kappa 0.44). The best cutoff for each judge raises kappa to 0.43 and
  0.49, a gain chosen on the same sample it is reported on. Against `qwen3:32b`
  the interval of that cutoff excludes 0.70 from above; against `gemma4:31b`
  it contains 0.70 and its point estimate is 1.0. The judges disagree on the
  direction a change should take.
- At the level the gate operates on, the mean over a system's answers, every
  floor between 0.205 and 0.785 classifies the five measured systems the same
  way both judges do when a system counts as faithful at a mean judge score of
  0.75 or more (the two base arms below, the three LoRA arms above). 0.70
  is inside that range; so is every value from 0.21 to 0.78. Five systems
  cannot pick a point inside it.

## Decision

Keep `faithfulness_threshold = 0.70`. The calibration does not support a
different value: the per-item optimum is unstable across judges and is not the
gated quantity, and at the system level 0.70 is consistent with both judges
but not determined by them. The value stays hand-chosen inside a measured
range, and the docs say so.

## Consequences

- The README no longer calls the floor unmeasured; it states the range the
  judges support and that the point inside it is a choice.
- The gate keeps its role as a regression floor on extractive answers. It is not
  a faithfulness estimate for model output; for that, the proxy reaches a
  Spearman of about 0.49 with either judge, against 0.78 between the judges.
- To revisit: a larger set of systems or human labels for the item-level
  verdict, and a gate on model output instead of extractive answers. Rerunning
  the judges is `scripts/calibrate_judge.py --run <label> --model <model>`; the
  rule above applies unchanged.
