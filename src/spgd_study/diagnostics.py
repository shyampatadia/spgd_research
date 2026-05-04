"""Diagnostics for measuring saddle / plateau behaviour during training.

A *stagnation episode* is a maximal run of consecutive optimiser steps where
||grad f||_2 < eps. Counting episodes (rather than just total stagnant steps)
distinguishes "stuck once for a long time" from "intermittently slow."

Escape time = number of consecutive stagnant steps within the longest episode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class StagnationTracker:
    """Online detector of stagnation episodes from a stream of gradient norms.

    Usage:
        tracker = StagnationTracker(eps=1e-4)
        for step in range(n_steps):
            ...
            tracker.update(grad_norm)
        summary = tracker.summary()
    """

    eps: float = 1e-4
    n_episodes: int = 0
    total_stagnant_steps: int = 0
    longest_episode: int = 0
    _current_run: int = 0
    _in_episode: bool = False
    grad_norm_history: List[float] = field(default_factory=list)

    def update(self, grad_norm_value: float) -> None:
        self.grad_norm_history.append(grad_norm_value)
        if grad_norm_value < self.eps:
            self.total_stagnant_steps += 1
            self._current_run += 1
            if not self._in_episode:
                self.n_episodes += 1
                self._in_episode = True
            if self._current_run > self.longest_episode:
                self.longest_episode = self._current_run
        else:
            self._current_run = 0
            self._in_episode = False

    def summary(self) -> dict:
        return dict(
            eps=self.eps,
            n_episodes=self.n_episodes,
            total_stagnant_steps=self.total_stagnant_steps,
            longest_episode=self.longest_episode,
        )
