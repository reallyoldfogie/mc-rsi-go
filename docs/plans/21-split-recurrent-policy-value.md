# 21 - Split recurrent policy and value branches

## Decision

The recurrent actor-critic will use independent policy and value trunks and
independent LSTM state. The existing shared-trunk recurrent model remains a
valid control and remains loadable, but new recurrent models will not share
representation parameters between the policy and critic.

The motivation is persistent optimization interference observed in the live
run: value-gradient norms were approximately three orders of magnitude larger
than policy-gradient norms, updates were repeatedly clipped, and student-vs-
teacher evaluation remained statistically flat after several days.

## Scope and compatibility

1. `actorcritic.RecurrentParams` gains `PolicyTrunk`, `ValueTrunk`,
   `PolicyLSTM`, and `ValueLSTM`. `Trunk` and `LSTM` remain compatibility
   aliases for the policy branch.
2. New recurrent constructors and warm starts allocate independent branches.
3. Collection, inference, chunk replay, and PPO updates carry independent
   policy/value hidden and cell states.
4. Recurrent checkpoint schema 2 writes both branches. Schema 1 checkpoints
   are accepted by duplicating the old shared trunk/LSTM into both branches.
5. Plain training is unchanged.

## Implementation sequence

- [x] Commit the pre-split recurrent lifecycle, recovery, instrumentation,
  dashboard, and compatibility work as baseline commit `cedca73`.
- [x] Add split recurrent model fields and independent initialization.
- [x] Route recurrent inference and actor state through both branches.
- [x] Record and replay both branch states in recurrent rollouts/chunks.
- [x] Route policy/value PPO losses to their respective parameter sets.
- [x] Add checkpoint schema 2 and schema 1 shared-model loading.
- [x] Add explicit unit tests proving policy updates do not change value
  parameters and value updates do not change policy parameters.
- [x] Add/extend metrics for separate branch parameter and gradient norms. The
  existing policy/value objective gradient metrics now map directly to the
  independent branches; the combined/applied metrics remain available for
  clipping diagnosis.
- [x] Run the full test suite and document results.
- [ ] Build and deploy only after review of the test results; compare against
  the shared-trunk control for at least 3-5 paired evaluations.

## Acceptance criteria

- A replayed rollout chunk reproduces collection-time policy and value outputs
  within floating-point tolerance when seeded with the recorded branch states.
- A policy-only objective leaves value trunk/LSTM matrices unchanged; a
  value-only objective leaves policy trunk/LSTM matrices unchanged.
- Old recurrent checkpoints load successfully and produce both branches by
  duplication.
- New checkpoints round-trip all policy/value trunk and LSTM matrices.
- Existing plain and recurrent lifecycle tests remain green.
- The live comparison reports policy/value loss, gradient norms, clipping,
  evaluation paired difference, and resource usage separately.

## Current status

The split is implemented through model construction, inference, rollout state
capture, chunk replay, PPO network construction, checkpoint encoding, and
branch-isolation tests. cRL actorcritic/PPO tests and the full trainer
repository test suite pass. The cRL changes are committed in its separate
local dependency repository; the trainer-side plan and configuration changes
are committed in this worktree. No live binary has been rebuilt or restarted
from this split yet.
