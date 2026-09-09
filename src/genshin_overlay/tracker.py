from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from genshin_overlay.models import (
    CooldownState,
    CooldownView,
    ScreenObservation,
    TrackerDecision,
)


@dataclass(slots=True)
class _Pending:
    seconds: float
    observed_at: float
    confirmations: int
    portrait: np.ndarray


@dataclass(slots=True)
class _Tracked:
    party_slot: int
    skill: str
    remaining_at_observation: float
    observed_at: float
    total_seconds: float
    portrait: np.ndarray

    def remaining(self, now: float) -> float:
        return max(0.0, self.remaining_at_observation - (now - self.observed_at))


class CooldownTracker:
    """Convert noisy same-frame observations into persistent cooldown state."""

    def __init__(self, confirmations: int = 2, consistency_tolerance: float = 0.45):
        self.confirmations = confirmations
        self.consistency_tolerance = consistency_tolerance
        self._pending: dict[tuple[int, str], _Pending] = {}
        self._tracked: dict[tuple[int, str], _Tracked] = {}
        self.last_decisions: tuple[TrackerDecision, ...] = ()

    def update(
        self,
        frame: np.ndarray,
        observation: ScreenObservation,
        now: float,
    ) -> tuple[CooldownView, ...]:
        decisions: list[TrackerDecision] = []
        self._expire(now)
        party = observation.party
        if party is None:
            self.last_decisions = ()
            return self.snapshot(now)

        portrait = party.portrait_rect.crop(frame)
        if portrait.size == 0:
            self.last_decisions = ()
            return self.snapshot(now)
        portrait = portrait.copy()
        portrait.setflags(write=False)

        for skill, reading in (("E", observation.e), ("Q", observation.q)):
            if (
                reading.state is not CooldownState.COOLDOWN
                or reading.seconds is None
                or reading.seconds <= 0
                or reading.confidence < 0.6
            ):
                decisions.append(
                    TrackerDecision(
                        party.active_slot,
                        skill,
                        "ignored",
                        f"state={reading.state.value} confidence={reading.confidence:.3f}",
                        reading.seconds,
                    )
                )
                continue
            key = (party.active_slot, skill)
            tracked = self._tracked.get(key)
            if tracked is not None:
                expected = tracked.remaining(now)
                if abs(reading.seconds - expected) <= self.consistency_tolerance:
                    tracked.remaining_at_observation = reading.seconds
                    tracked.observed_at = now
                    tracked.portrait = portrait
                    decisions.append(
                        TrackerDecision(
                            party.active_slot,
                            skill,
                            "resynced",
                            "reading matches local countdown",
                            reading.seconds,
                            expected,
                        )
                    )
                elif expected <= 0.5:
                    self._tracked[key] = self._new_tracked(
                        party.active_slot, skill, reading.seconds, now, portrait
                    )
                    decisions.append(
                        TrackerDecision(
                            party.active_slot,
                            skill,
                            "restarted",
                            "previous cooldown reached zero",
                            reading.seconds,
                            expected,
                        )
                    )
                else:
                    decisions.append(
                        TrackerDecision(
                            party.active_slot,
                            skill,
                            "rejected",
                            "reading differs from local countdown",
                            reading.seconds,
                            expected,
                        )
                    )
                continue

            pending = self._pending.get(key)
            if pending is None:
                self._pending[key] = _Pending(reading.seconds, now, 1, portrait)
                decisions.append(
                    TrackerDecision(
                        party.active_slot,
                        skill,
                        "pending",
                        f"confirmation 1/{self.confirmations}",
                        reading.seconds,
                    )
                )
                continue
            expected = max(0.0, pending.seconds - (now - pending.observed_at))
            if abs(reading.seconds - expected) <= self.consistency_tolerance:
                pending.seconds = reading.seconds
                pending.observed_at = now
                pending.confirmations += 1
                pending.portrait = portrait
            else:
                self._pending[key] = _Pending(reading.seconds, now, 1, portrait)
                decisions.append(
                    TrackerDecision(
                        party.active_slot,
                        skill,
                        "rejected",
                        "candidate sequence is inconsistent; restarted confirmation",
                        reading.seconds,
                        expected,
                    )
                )
                continue
            if pending.confirmations >= self.confirmations:
                self._tracked[key] = self._new_tracked(
                    party.active_slot, skill, pending.seconds, now, pending.portrait
                )
                del self._pending[key]
                decisions.append(
                    TrackerDecision(
                        party.active_slot,
                        skill,
                        "confirmed",
                        f"confirmation {pending.confirmations}/{self.confirmations}",
                        pending.seconds,
                        expected,
                    )
                )
            else:
                decisions.append(
                    TrackerDecision(
                        party.active_slot,
                        skill,
                        "pending",
                        f"confirmation {pending.confirmations}/{self.confirmations}",
                        pending.seconds,
                        expected,
                    )
                )

        self.last_decisions = tuple(decisions)
        return self.snapshot(now)

    def snapshot(self, now: float) -> tuple[CooldownView, ...]:
        self._expire(now)
        views = [
            CooldownView(
                party_slot=item.party_slot,
                skill=item.skill,
                remaining_seconds=item.remaining(now),
                total_seconds=item.total_seconds,
                portrait_bgr=item.portrait,
            )
            for item in self._tracked.values()
        ]
        return tuple(sorted(views, key=lambda item: (item.party_slot, item.skill)))

    def _expire(self, now: float) -> None:
        expired = [key for key, item in self._tracked.items() if item.remaining(now) <= 0]
        for key in expired:
            del self._tracked[key]
        stale = [key for key, item in self._pending.items() if now - item.observed_at > 1.0]
        for key in stale:
            del self._pending[key]

    @staticmethod
    def _new_tracked(
        party_slot: int,
        skill: str,
        seconds: float,
        now: float,
        portrait: np.ndarray,
    ) -> _Tracked:
        return _Tracked(party_slot, skill, seconds, now, seconds, portrait)
