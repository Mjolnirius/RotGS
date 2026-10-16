# E09 — Transfer the R082 setup to more products

[All objectives](README.md) · [Project documentation](../README.md)

**Objective:** Does the selected R082 setup transfer to Ramseier, KaegiFret, and Weisswein?  
**Status at review:** Queue defined; consult logs for current outcomes  
**Primary run entries:** Not yet assigned in the run ledger  
**Reused references:** R082, R073, R054, R059, R065  
**Reviewed:** 2026-10-15, Europe/Zurich. Historical objectives were reconstructed from the run log and reports; they are editable descriptions of intent.

## Motivation and design

The current queue applies the selected R082 settings to Session 04 Ramseier, KaegiFret Top, and Weisswein. It uses fresh SAM31AC preparations, measured TAE angles, projected ±0.5° axes, and disabled local/global angle corrections. This extends the can-specific choice to other product captures.

## Runs and current state

The three queue output identities are:

- `4_Ramseier_30d_SAM31AC_TAE_R082_axis_projected_30k`
- `4_KaegiFret_Top_30d_SAM31AC_TAE_R082_axis_projected_30k`
- `4_Weisswein_30d_SAM31AC_TAE_R082_axis_projected_30k`

These are the updated queue targets. Three earlier outputs use the `SAM31_AUTO_TAE` preparation instead:

- `4_Ramseier_30d_SAM31_AUTO_TAE_R082_axis_projected_30k`
- `4_KaegiFret_Top_30d_SAM31_AUTO_TAE_R082_axis_projected_30k`
- `4_Weisswein_30d_SAM31_AUTO_TAE_R082_axis_projected_30k`

All three earlier output folders contain saved checkpoints through 30k. Their saved configurations name the earlier AUTO datasets; they are distinct from the new SAM31AC queue targets. The R001–R085 ledger snapshot does not assign these product-transfer attempts IDs. Register them from actual launch/configuration evidence; a saved endpoint alone does not establish process exit status. Current outcomes belong in the queue logs and run ledger rather than a fixed status claim here.

## Evidence and follow-up

- [Training queue](../../jobs/train_queue.sh)
- [Training logs](../../logs/train_queue/session_04_r082_products)
- [Preprocessing queue](../../jobs/preprocess_queue.sh)
- [Preprocessing logs](../../logs/preprocess_queue/session_04_r082)
- [Selected can-axis study](E07_axis.md)

Compare within the same product/capture using a documented target. Earlier Session 03 and SAM31 runs provide context but differ in capture/preparation. Do not present a cross-product leaderboard as an isolated axis effect.

## Your notes

Add observations, changed objectives, and decisions here. Preserve these notes when updating the study.
