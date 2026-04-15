# Advanced Baselines

This folder contains research-oriented comparator baselines that are heavier,
more assumption-sensitive, and not recommended as the primary baselines for the
main paper narrative.

Included here:
- `WhittleIndexScheduler`
- `CDKFGradientScheduler`
- `MIQPOptimalScheduler`
- `DRLScheduler`

They are still re-exported through `src/baselines/baselines.py` for backwards
compatibility, so existing imports continue to work.
