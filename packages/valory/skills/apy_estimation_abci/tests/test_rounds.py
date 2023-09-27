# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2021-2023 Valory AG
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------

"""Test the rounds of the skill."""


from typing import Dict, FrozenSet, Optional, Tuple
from unittest import mock

import pytest

from packages.valory.skills.abstract_round_abci.base import AbciAppDB, CollectionRound
from packages.valory.skills.abstract_round_abci.test_tools.rounds import (
    BaseCollectSameUntilThresholdRoundTest,
)
from packages.valory.skills.apy_estimation_abci.payloads import (
    EstimatePayload,
    FetchingPayload,
    OptimizationPayload,
    PreprocessPayload,
    RandomnessPayload,
)
from packages.valory.skills.apy_estimation_abci.payloads import (
    TestingPayload as _TestingPayload,
)
from packages.valory.skills.apy_estimation_abci.payloads import (
    TrainingPayload,
    TransformationPayload,
)
from packages.valory.skills.apy_estimation_abci.rounds import (
    CollectHistoryRound,
    EstimateRound,
    Event,
    OptimizeRound,
    PreprocessRound,
    RandomnessRound,
    SynchronizedData,
)
from packages.valory.skills.apy_estimation_abci.rounds import TestRound as _TestRound
from packages.valory.skills.apy_estimation_abci.rounds import TrainRound, TransformRound


MAX_PARTICIPANTS: int = 4
RANDOMNESS: int = 1785937125
INVALID_RANDOMNESS = None


def get_participants() -> FrozenSet[str]:
    """Participants"""
    return frozenset([f"agent_{i}" for i in range(MAX_PARTICIPANTS)])


def get_participant_to_fetching(
    participants: FrozenSet[str], history: Optional[str]
) -> Dict[str, FetchingPayload]:
    """participant_to_fetching"""
    return {
        participant: FetchingPayload(sender=participant, history=history)
        for participant in participants
    }


def get_participant_to_randomness(
    participants: FrozenSet[str],
) -> Dict[str, RandomnessPayload]:
    """participant_to_randomness"""
    return {
        participant: RandomnessPayload(
            sender=participant,
            randomness=RANDOMNESS,
        )
        for participant in participants
    }


def get_participant_to_invalid_randomness(
    participants: FrozenSet[str],
) -> Dict[str, RandomnessPayload]:
    """Invalid participant_to_randomness"""
    return {
        participant: RandomnessPayload(
            sender=participant,
            randomness=INVALID_RANDOMNESS,
        )
        for participant in participants
    }


def get_transformation_payload(
    participants: FrozenSet[str],
    transformation_hash: Optional[str],
    latest_observation_hist_hash: Optional[str],
    latest_transformation_period: Optional[int],
) -> Dict[str, TransformationPayload]:
    """Get transformation payloads."""
    return {
        participant: TransformationPayload(
            participant,
            transformation_hash,
            latest_observation_hist_hash,
            latest_transformation_period,
        )
        for participant in participants
    }


def get_participant_to_preprocess_payload(
    participants: FrozenSet[str],
    train_hash: Optional[str],
    test_hash: Optional[str],
) -> Dict[str, PreprocessPayload]:
    """Get preprocess payload."""
    if any(hash_ is None for hash_ in (train_hash, test_hash)):
        train_test_hash = None
    else:
        train_test_hash = str(train_hash) + str(test_hash)
    return {
        participant: PreprocessPayload(participant, train_test_hash)
        for participant in participants
    }


def get_participant_to_optimize_payload(
    participants: FrozenSet[str],
) -> Dict[str, OptimizationPayload]:
    """Get optimization payload."""
    return {
        participant: OptimizationPayload(participant, "best_params_hash")  # type: ignore
        for participant in participants
    }


def get_participant_to_train_payload(
    participants: FrozenSet[str],
    models_hash: Optional[str],
) -> Dict[str, TrainingPayload]:
    """Get training payload."""
    return {
        participant: TrainingPayload(participant, models_hash)
        for participant in participants
    }


def get_participant_to_test_payload(
    participants: FrozenSet[str],
) -> Dict[str, _TestingPayload]:
    """Get testing payload."""
    return {
        participant: _TestingPayload(participant, "report_hash")
        for participant in participants
    }


def get_participant_to_estimate_payload(
    participants: FrozenSet[str],
    estimations_hash: Optional[str],
    n_estimations: Optional[int] = 1,
) -> Dict[str, EstimatePayload]:
    """Get estimate payload."""
    return {
        participant: EstimatePayload(participant, n_estimations, estimations_hash)
        for participant in participants
    }


class TestCollectHistoryRound(BaseCollectSameUntilThresholdRoundTest):
    """Test `CollectHistoryRound`."""

    _synchronized_data_class = SynchronizedData
    _event_class = Event

    @pytest.mark.parametrize(
        "most_voted_payload, expected_event",
        (("x0", Event.DONE), (None, Event.FILE_ERROR), ("", Event.NETWORK_ERROR)),
    )
    def test_run(
        self,
        most_voted_payload: Optional[str],
        expected_event: Event,
    ) -> None:
        """Runs test."""
        test_round = CollectHistoryRound(self.synchronized_data, mock.MagicMock())
        self._complete_run(
            self._test_round(
                test_round=test_round,
                round_payloads=get_participant_to_fetching(
                    self.participants, most_voted_payload
                ),
                synchronized_data_update_fn=lambda _synchronized_data, _: _synchronized_data,
                synchronized_data_attr_checks=[],
                most_voted_payload=most_voted_payload,
                exit_event=expected_event,
            )
        )

    def test_no_majority_event(self) -> None:
        """Test the no-majority event."""
        test_round = CollectHistoryRound(self.synchronized_data, mock.MagicMock())
        self._test_no_majority_event(test_round)


class TestTransformRound(BaseCollectSameUntilThresholdRoundTest):
    """Test `TransformRound`."""

    _synchronized_data_class = SynchronizedData
    _event_class = Event

    @pytest.mark.parametrize(
        "transformed_history_hash, latest_observation_hist_hash, latest_transformation_period, expected_event",
        (("x0", "x1", 10, Event.DONE), (None, None, None, Event.FILE_ERROR)),
    )
    def test_run(
        self,
        transformed_history_hash: Optional[str],
        latest_observation_hist_hash: Optional[str],
        latest_transformation_period: Optional[int],
        expected_event: Event,
    ) -> None:
        """Runs test."""
        initial_latest_transformation_period = 2
        initial_period_count = 2

        test_round = TransformRound(
            self.synchronized_data.update(
                latest_transformation_period=initial_latest_transformation_period,
                period_count=initial_period_count,
            ),
            mock.MagicMock(),
        )
        self._complete_run(
            self._test_round(
                test_round=test_round,
                round_payloads=get_transformation_payload(
                    self.participants,
                    transformed_history_hash,
                    latest_observation_hist_hash,
                    latest_transformation_period,
                ),
                synchronized_data_update_fn=lambda _synchronized_data, _: _synchronized_data.update(
                    latest_transformation_period=latest_transformation_period
                    or initial_latest_transformation_period,
                ),
                synchronized_data_attr_checks=[
                    lambda _synchronized_data: _synchronized_data.latest_transformation_period,
                ],
                most_voted_payload=transformed_history_hash,
                exit_event=expected_event,
            )
        )

    def test_no_majority_event(self) -> None:
        """Test the no-majority event."""
        test_round = TransformRound(self.synchronized_data, mock.MagicMock())
        self._test_no_majority_event(test_round)


class TestPreprocessRound(BaseCollectSameUntilThresholdRoundTest):
    """Test `PreprocessRound`."""

    _synchronized_data_class = SynchronizedData
    _event_class = Event

    @pytest.mark.parametrize(
        "most_voted_payloads, expected_event",
        ((("train_hash", "test_hash"), Event.DONE), ((None, None), Event.FILE_ERROR)),
    )
    def test_run(
        self,
        most_voted_payloads: Tuple[Optional[str]],
        expected_event: Event,
    ) -> None:
        """Runs test."""

        test_round = PreprocessRound(self.synchronized_data, mock.MagicMock())
        self._complete_run(
            self._test_round(
                test_round=test_round,
                round_payloads=get_participant_to_preprocess_payload(  # type: ignore
                    self.participants, *most_voted_payloads
                ),
                synchronized_data_update_fn=lambda _synchronized_data, _: _synchronized_data,
                synchronized_data_attr_checks=[],
                most_voted_payload="train_hashtest_hash"
                if not any(payload is None for payload in most_voted_payloads)
                else None,
                exit_event=expected_event,
            )
        )

    def test_no_majority_event(self) -> None:
        """Test the no-majority event."""
        test_round = PreprocessRound(self.synchronized_data, mock.MagicMock())
        self._test_no_majority_event(test_round)


class TestRandomnessRound(BaseCollectSameUntilThresholdRoundTest):
    """Test RandomnessRound."""

    _synchronized_data_class = SynchronizedData
    _event_class = Event

    def test_run(
        self,
    ) -> None:
        """Run tests."""

        test_round = RandomnessRound(self.synchronized_data, mock.MagicMock())
        self._complete_run(
            self._test_round(
                test_round=test_round,
                round_payloads=get_participant_to_randomness(self.participants),
                synchronized_data_update_fn=lambda synchronized_data, _: synchronized_data,
                synchronized_data_attr_checks=[],
                most_voted_payload=RANDOMNESS,
                exit_event=Event.DONE,
            )
        )

    def test_invalid_randomness(self) -> None:
        """Test the no-majority event."""
        test_round = RandomnessRound(self.synchronized_data, mock.MagicMock())
        self._complete_run(
            self._test_round(
                test_round=test_round,
                round_payloads=get_participant_to_invalid_randomness(self.participants),
                synchronized_data_update_fn=lambda synchronized_data, _: synchronized_data,
                synchronized_data_attr_checks=[],
                most_voted_payload=INVALID_RANDOMNESS,
                exit_event=Event.RANDOMNESS_INVALID,
            )
        )

    def test_no_majority_event(self) -> None:
        """Test the no-majority event."""
        test_round = RandomnessRound(self.synchronized_data, mock.MagicMock())
        self._test_no_majority_event(test_round)


class TestOptimizeRound(BaseCollectSameUntilThresholdRoundTest):
    """Test `OptimizeRound`."""

    _synchronized_data_class = SynchronizedData
    _event_class = Event

    def test_run(self) -> None:
        """Runs test."""

        test_round = OptimizeRound(self.synchronized_data, mock.MagicMock())
        self._complete_run(
            self._test_round(
                test_round=test_round,
                round_payloads=get_participant_to_optimize_payload(self.participants),
                synchronized_data_update_fn=lambda _synchronized_data, _: _synchronized_data,
                synchronized_data_attr_checks=[],
                most_voted_payload="best_params_hash",
                exit_event=Event.DONE,
            )
        )

    def test_no_majority_event(self) -> None:
        """Test the no-majority event."""
        test_round = OptimizeRound(self.synchronized_data, mock.MagicMock())
        self._test_no_majority_event(test_round)


class TestTrainRound(BaseCollectSameUntilThresholdRoundTest):
    """Test `TrainRound`."""

    _synchronized_data_class = SynchronizedData
    _event_class = Event

    @pytest.mark.parametrize(
        "full_training, most_voted_payload, expected_event",
        (
            (True, "x0", Event.FULLY_TRAINED),
            (False, "x0", Event.DONE),
            (True, None, Event.FILE_ERROR),
        ),
    )
    def test_run(
        self,
        full_training: bool,
        most_voted_payload: Optional[str],
        expected_event: Event,
    ) -> None:
        """Runs test."""

        test_round = TrainRound(
            self.synchronized_data.update(full_training=full_training), mock.MagicMock()
        )
        self._complete_run(
            self._test_round(
                test_round=test_round,
                round_payloads=get_participant_to_train_payload(
                    self.participants, most_voted_payload
                ),
                synchronized_data_update_fn=lambda _synchronized_data, _: _synchronized_data,
                synchronized_data_attr_checks=[],
                most_voted_payload=most_voted_payload,
                exit_event=expected_event,
            )
        )

    def test_no_majority_event(self) -> None:
        """Test the no-majority event."""
        test_round = TrainRound(self.synchronized_data, mock.MagicMock())
        self._test_no_majority_event(test_round)


class TestTestRound(BaseCollectSameUntilThresholdRoundTest):
    """Test `TestRound`."""

    _synchronized_data_class = SynchronizedData
    _event_class = Event

    def test_run(self) -> None:
        """Runs test."""

        test_round = _TestRound(
            self.synchronized_data.update(full_training=False), mock.MagicMock()
        )
        self._complete_run(
            self._test_round(
                test_round=test_round,
                round_payloads=get_participant_to_test_payload(self.participants),
                synchronized_data_update_fn=lambda _synchronized_data, _: _synchronized_data.update(
                    full_training=True
                ),
                synchronized_data_attr_checks=[
                    lambda _synchronized_data: bool(_synchronized_data.full_training)
                ],
                most_voted_payload="report_hash",
                exit_event=Event.DONE,
            )
        )

    def test_no_majority_event(self) -> None:
        """Test the no-majority event."""
        test_round = _TestRound(self.synchronized_data, mock.MagicMock())
        self._test_no_majority_event(test_round)


class TestEstimateRound(BaseCollectSameUntilThresholdRoundTest):
    """Test `EstimateRound`."""

    _synchronized_data_class = SynchronizedData
    _event_class = Event

    @pytest.mark.parametrize(
        "n_estimations, estimate_hash, expected_event",
        (
            (1, "test_hash", Event.DONE),
            (None, None, Event.FILE_ERROR),
        ),
    )
    def test_estimation_cycle_run(
        self,
        n_estimations: Optional[int],
        estimate_hash: Optional[str],
        expected_event: Event,
    ) -> None:
        """Runs test."""
        test_round = EstimateRound(
            self.synchronized_data.update(n_estimations=0), mock.MagicMock()
        )
        self._complete_run(
            self._test_round(
                test_round=test_round,
                round_payloads=get_participant_to_estimate_payload(
                    self.participants,
                    estimate_hash,
                    n_estimations,
                ),
                synchronized_data_update_fn=lambda _synchronized_data, _: _synchronized_data.update(
                    n_estimations=n_estimations,
                ),
                synchronized_data_attr_checks=[lambda _: n_estimations],
                most_voted_payload=n_estimations,
                exit_event=expected_event,
            )
        )

    def test_no_majority_event(self) -> None:
        """Test the no-majority event."""
        test_round = EstimateRound(self.synchronized_data, mock.MagicMock())
        self._test_no_majority_event(test_round)


def test_period() -> None:
    """Test SynchronizedData."""

    participants = get_participants()
    setup_params: Dict = {}
    most_voted_randomness = 1
    estimates_hash = "test_hash"
    full_training = False
    n_estimations = 1
    participant_to_estimate = get_participant_to_estimate_payload(
        frozenset({"test_agent"}), "test_hash"
    )

    synchronized_data = SynchronizedData(
        db=AbciAppDB(
            setup_data=AbciAppDB.data_to_lists(
                dict(
                    participants=tuple(participants),
                    setup_params=setup_params,
                    most_voted_randomness=most_voted_randomness,
                    estimates_hash=estimates_hash,
                    full_training=full_training,
                    n_estimations=n_estimations,
                    participant_to_estimate=CollectionRound.serialize_collection(
                        participant_to_estimate
                    ),
                )
            ),
        )
    )

    assert synchronized_data.participants == participants
    assert synchronized_data.period_count == 0
    assert synchronized_data.most_voted_randomness == most_voted_randomness
    assert synchronized_data.estimates_hash == estimates_hash
    assert synchronized_data.full_training == full_training
    assert synchronized_data.n_estimations == n_estimations
    assert synchronized_data.is_most_voted_estimate_set is not None
    assert synchronized_data.participant_to_estimate == participant_to_estimate
