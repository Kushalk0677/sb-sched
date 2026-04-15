# Framing and Limitations

This note is intended to help position the repository accurately in GitHub, academic writing, demos, and project reports.

## Correct framing

The strongest defensible framing for this repository is:

> A covariance-bounded adaptive update scheduling framework for Kalman-style state estimators, studied in controlled multi-rate estimation testbeds.

SB-Sched is fundamentally a **scheduler**. It reasons about when to consume low-rate measurements in order to keep estimator uncertainty within a budget while reducing sensing cost.

## What the code does support

The repository supports claims about:

- uncertainty-aware update scheduling
- measurement-efficiency versus estimation-quality tradeoffs
- behaviour under outages, latency, jitter, and hard sensing budgets
- transition behaviour across different motion regimes
- Pareto-optimal operating points under competing quality and cost metrics

## What the code does not support strongly enough

The repository should not be framed as a full end-to-end robotics fusion stack.

### EuRoC and TUM-VI

In the current implementation:

- low-rate `camera` updates are oracle-style pseudo-measurements derived from interpolated ground truth
- the experiments are designed to isolate scheduler behaviour from perception-front-end noise
- the setup is therefore best read as a controlled estimation/scheduling testbed rather than a full visual-inertial odometry benchmark

### KITTI Raw

The KITTI path is more physically grounded, but it should still be described carefully.

It is safer to say that the KITTI experiments validate scheduling behaviour and estimator consistency in a realistic data setting than to claim superiority over production navigation or SLAM pipelines.

## Why this framing is still valuable

These limitations do not invalidate the work. They define its scope.

A repository can still make a meaningful contribution if it cleanly isolates one important systems problem. In this case, the problem is:

> How should an estimator allocate scarce measurement opportunities while keeping uncertainty under control?

That is a relevant and practically important question in robotics, autonomous systems, edge sensing, and embedded estimation.

## How to talk about the results

Good phrasing:

- “adaptive update scheduling”
- “covariance-bounded sensing”
- “resource-constrained estimation”
- “controlled multi-rate estimation testbed”
- “Pareto tradeoff between sensing rate and quality loss”

Avoid over-claiming phrasing such as:

- “state-of-the-art sensor fusion”
- “best end-to-end navigation accuracy”
- “full VIO benchmark”
- “provably optimal against all fixed-rate baselines”

## Best-supported experimental story

The most defensible narrative emerging from the experiments is:

1. SB-Sched aggressively reduces effective sensing rate.
2. Under some stressful conditions, that lower sensing rate increases RMSE and violation metrics.
3. Under tight but feasible sensing budgets, SB-Sched can allocate updates more effectively than several comparison baselines.
4. Across newer experiment families, SB-Sched frequently lies on the Pareto frontier, meaning it is not dominated once sensing cost is taken seriously.

That is already a strong systems contribution.

## Suggested one-sentence summary

If you need one line for GitHub, a paper draft, or a portfolio, use this:

> SB-Sched is a covariance-aware adaptive update scheduler for Kalman-style estimators that studies how to trade measurement cost against uncertainty and estimation quality in controlled multi-rate settings.
