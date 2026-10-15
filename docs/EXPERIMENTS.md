# Remote experiment plan and result contract

Protocol draft dated 2026-10-15. All performance outcomes below are planned. Existing package smoke tests are software validation, not evidence of motion quality. Use the README for supported commands and the configuration files for exact defaults. A suggested experiment that lacks a CLI option is an extension task, not an implemented feature.

## First remote run

1. Verify the code revision, private archive checksum, extraction destination, dataset receipt, and environment lock/dependency report. Run the package's dataset validation and CPU smoke checks.
2. Inspect counts by original split and source ancestry. Confirm that new train/calibration/validation/test subdivisions come only from the permitted original training population and that original held-out data stays held out. The package test set is an internal holdout, not the original project's external or sealed panel. Reject a run if any ancestor crosses the intended split boundary.
3. Confirm channel order, 50 Hz timing, 25 observed frames and 100 future frames, local coordinate frame, goal provenance, masks, and train-only normalization. A two-second target contains 100 frames at 50 Hz; report the exact sample timestamps because inclusive endpoints can change apparent duration.
4. Run a small bounded pilot using a fresh output directory. Measure actual examples/second, peak VRAM, model parameters, checkpoint size, and wall time before scheduling the full comparison. Synthetic data only exercises the pipeline.
5. Train the first real-data scene-enabled and scene-disabled arms with matched data exposure, parameters where practical, optimizer budget, and checkpoint-selection rule. Record any unavoidable difference in parameters or compute.
6. Return the compact report plus receipts, configurations, metrics, and representative sampled futures. Do not send private raw motion in a public issue or commit generated checkpoints to Git.

The side project is CPU/GPU training without RL or simulator rollouts. The existing local Motion2Scene program's execution and sealed-evaluation gates remain separate.

## Minimum useful experiment matrix

Do not run every cell immediately. The first comparison is R0 versus R1; the next tests whether reconstruction pretraining earns its cost. Repeat promising comparisons with independent initialization seeds. A one-seed screen is a pilot, not evidence of a stable gain.

| ID | Arm | Question and fair comparison |
| --- | --- | --- |
| R0 | Future mean/log-variance prediction, scene disabled | Can history and goal already explain the recorded future? |
| R1 | Same predictor, scene enabled | Does scene information improve held-out prediction at the same budget? Report actual parameter difference if any. |
| R2 | Masked-history initialization → R1 fine-tuning | Does pretraining improve transfer? Include pretraining steps in total cost; also report matched downstream steps as a secondary comparison. |
| R3 | Joint masked-history + future prediction | Is staged initialization better than simpler joint training at equal total exposure? |
| G0 | Conditional Flow trained from scratch | Does a multimodal model improve held-out distributional quality over the future-mean baseline? Keep sample count/latency visible. |
| G1 | Masked-history initialization → conditional Flow | Does representation pretraining improve generation at equal total cost? |
| G2 | Joint masked-history + conditional Flow | Does the staged recipe earn its complexity? |
| U1 | Independent replicas of the chosen predictive arm | Does ensemble disagreement rank error beyond distance/speed/horizon baselines? Separate training-seed variation from predictive uncertainty. |

An objective weight of zero can implement some controls, but verify that disabled branches do not accidentally update the shared representation or change data sampling. The current `--init` loads the full same-architecture checkpoint with the same dataset manifest and its normalizer, then creates a new optimizer. It is a warm start, not an exact resume. Different-architecture or motion-only-to-paired-data transfer needs an explicit compatible-encoder loader; save its loaded/missing keys if added. Do not silently load incompatible normalizers or channel conventions.

Use a low-cost fixed pilot budget chosen after the throughput measurement; then freeze a common budget before comparing arms. Record both optimizer steps and actual valid frames/windows seen. “One epoch” is not equal cost if a data filter changes dataset size. Do not report an estimated GPU-hour budget as an observed runtime.

## Questions that change the research decision

**Q1: Does the model use the scene usefully?** Compare R0/R1 and the chosen generator's original, removed, and shuffled scene conditions. Coupled generator noise isolates the effect of changing context. Report both prediction error and output change. A larger response is not automatically good, and scene ablation can be out of the training distribution. Therefore combine this diagnostic with retrained scene-blind models and, later, valid irrelevant-obstacle or changed-scene/future pairs.

If R1 does not beat R0, inspect goal leakage, low scene diversity, and hindsight-compatible scene construction before enlarging the model. If R1 reacts strongly but accuracy does not improve, attention may be exploiting a shortcut or responding indiscriminately. If extra clean scene contrasts help while width does not, prioritize data acquisition.

**Q2: Does motion pretraining provide transferable structure?** Compare scratch, staged, and joint objectives. Evaluate both a frozen encoder plus a small common prediction head and full fine-tuning; these answer different questions. A full fine-tuning gain could reflect optimization rather than a linearly accessible representation. Useful probes include future root displacement and local motion change, derived from well-defined channels. Keep probe train/evaluation ancestry disjoint. Do not invent a “required skill” label from a compatible motion alone.

**Q3: Are generated futures plausible and appropriately varied?** Report deterministic error, sample-average error, and best-of-K error with K fixed. Best-of-K is an oracle diagnostic; it is not the performance of a deployed selector. Include diversity and basic temporal consistency so noise is not rewarded as useful multimodality. Euclidean error to one recorded future can punish another valid future, so any distributional claim needs that limitation and, ideally, qualified multi-future examples.

**Q4: Does uncertainty add decision-relevant information?** Measure whether variance or ensemble disagreement predicts held-out residual error after conditioning on horizon and simple geometry/motion baselines. Use calibration for interval scaling and evaluation for the final assessment. Compare global and horizon-specific coverage/width. Also inspect high-error low-uncertainty cases. A larger variance near obstacles, without improved ranking/calibration, is not a useful result.

**Q5: Can it return to ordinary motion after adaptation?** The extension dataset should preserve approach, obstacle interaction, and recovery. Measure behavior against speed- and turn-matched clear examples, with exact event-label definitions. Arm spread or shoulder swing can be descriptive proxies, but neither is a validated naturalness score. Do not globally penalize tucking or demand upright posture where geometry requires adaptation.

## Measurements and denominators

The MVP report should distinguish metrics it computes from research extensions. Missing metrics should be marked unavailable; do not substitute names for a different quantity.

| Measurement | Required interpretation |
| --- | --- |
| Masked-history reconstruction | Error only on masked valid coordinates; report the mask policy |
| Future prediction error | Normalized error for optimization plus native channel-group units for interpretation; horizon slices near 0.2, 0.5, 1.0, and 2.0 seconds where available |
| Gaussian NLL | Specify normalization and valid dimensions; a comparison of the assumed predictive density, not physical safety |
| Pointwise interval coverage and width | State nominal coverage, calibration population, channel/horizon aggregation, and count of examples/groups |
| Scene perturbation response | Same model, history, goal, and generator noise; report perturbation validity and output/error changes |
| Generation sample quality/diversity | Fixed K and function evaluations, ordinary sample and oracle best-of-K, speed/pose/temporal diagnostics available from the schema |
| Training/inference cost | Device, peak VRAM, wall time, examples/second, parameters, batch size, inference latency and sampler evaluations |
| FK clearance, slip/contact, joint-limit excess | Extensions only after tested body geometry and channel mappings; reference geometry remains a kinematic result |
| Executed task completion or collision-free success | Not measured in this offline package; requires later simulator evaluation with a fixed scorer |

Aggregate windows within ancestral groups before describing performance across independent groups, or use a documented hierarchical analysis. Report group counts, clip counts, window counts, and training seeds separately. Do not turn millions of overlapping frames into millions of independent trials. Pair arms on evaluation groups and sampling noise when meaningful. Bootstrap groups for exploratory intervals and show seed-level values; a broad claim may require more data than the pilot can supply.

## Attention and uncertainty visual report

Include a fixed, preselected panel plus separate worst cases; avoid selecting only attractive examples. For each clip, align these views on one physical timeline:

1. Observed history, recorded future, and several generated futures with a common camera and scale.
2. Scene-object attention by forecast time, labeled as attention weights.
3. Error and predicted residual interval width versus forecast horizon; use separate traces for sample spread and ensemble disagreement.
4. Original scene versus a qualified intervention, keeping generator noise fixed.
5. Provenance badge: observed/jointly generated/hindsight/unknown scene; reference-conditioned or deployable goal; train/calibration/evaluation.

The MVP may export arrays and summaries rather than this complete viewer. Building this aligned viewer is a proposed reporting extension. If scene-to-body relevance labels are absent, show that absence instead of depicting attention as measured human gaze. Unknown obstacle geometry is not a zero-distance obstacle or free space.

## Scale only after the small comparisons are interpretable

| Scaling axis | First measurement | Next action if useful |
| --- | --- | --- |
| Distinct motion ancestry | Learning curve by groups/hours, with fixed evaluation | Add new groups rather than denser overlapping windows |
| Decision coverage | Counts of clear, approach, passage, exit, recovery, turns | Acquire or curate missing transitions with provenance |
| Scene contrast quality | Same-history valid alternatives and verified irrelevant edits | Add targeted contrasts; do not randomly relabel futures |
| Horizon | Error/selectivity at 1 versus 2 seconds and matched latency | Add multiresolution prediction only if long context helps |
| Model size | Same data/exposure and measured cost | Increase capacity after a reproducible capacity limit |
| Uncertainty compute | Single model versus independently trained ensemble | Pay extra inference cost only if ranking/calibration improve |
| Motion-only corpus | Stage A transfer gain with licenses/splits respected | Expand pretraining without relabeling absent scenes as observed |

Event-balanced sampling, uncertainty-based acquisition, and contrastive pair losses are extensions. For acquisition, combine high residual/ensemble disagreement with diversity and failure categories; selecting only highest variance can overcollect annotation errors or intrinsically ambiguous examples. Keep evaluation data out of acquisition decisions.

## Report back

Return one result directory containing the following information, using the package's output filenames where available:

```text
run identity: code commit, dirty-state/source hash, config, seed, command, environment
data identity: manifest/archive checksums, ancestry counts, original and derived splits
resources: hardware, wall time, peak memory, throughput, number of examples seen
training: objective traces, checkpoint selection, loaded initialization keys
evaluation: per-arm/per-seed metrics and group-level or per-window tables
calibration: population, fitted scale, coverage and width, no evaluation fitting
samples: reproducible sample IDs/noise seeds and private arrays if authorized
conclusion: supported observation, unresolved issue, next discriminating experiment
```

Write failures and unavailable measurements next to favorable results. The expected first decision is whether scene conditioning and pretraining improve offline prediction/selectivity enough to justify the generative comparison. A null or inconclusive result favors the simpler model and a data audit. Physical collision avoidance, naturalness, and planner performance remain untested until the resulting model is evaluated in the original execution framework.
