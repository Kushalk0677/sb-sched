"""Compatibility exports for baseline schedulers.

Main, paper-safe baselines live in ``baselines.main``.
This module re-exports them to preserve the original import path:
    from baselines.baselines import ...

Advanced baselines (WhittleIndexScheduler, CDKFGradientScheduler,
MIQPOptimalScheduler, DRLScheduler) have been removed from the paper
comparison set and are no longer exported here.
"""

from .main import (
    BaseScheduler,
    FixedHighScheduler,
    FixedLowScheduler,
    FixedMatchedScheduler,
    HeuristicAdaptiveScheduler,
    AoIMinScheduler,
    PeriodicOptimalScheduler,
    EventTriggeredScheduler,
    DelayAwarePolicyScheduler,
    ClairvoyantLookaheadScheduler,
    VarianceThresholdScheduler,
)

__all__ = [
    "BaseScheduler",
    "FixedHighScheduler",
    "FixedLowScheduler",
    "FixedMatchedScheduler",
    "HeuristicAdaptiveScheduler",
    "AoIMinScheduler",
    "PeriodicOptimalScheduler",
    "EventTriggeredScheduler",
    "DelayAwarePolicyScheduler",
    "ClairvoyantLookaheadScheduler",
    "VarianceThresholdScheduler",
]
