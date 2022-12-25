# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2021-2022 Valory AG
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

"""This module contains the rounds for the APY estimation ABCI application."""
from abc import ABC
from enum import Enum
from typing import Dict, List, Mapping, Optional, Set, Tuple, Type, cast

from packages.valory.skills.abstract_round_abci.base import (
    AbciApp,
    AbciAppTransitionFunction,
    AbstractRound,
    AppState,
    BaseSynchronizedData,
    CollectSameUntilThresholdRound,
    DegenerateRound,
    TransactionType,
    VotingRound,
    get_name,
)
from packages.valory.skills.apy_estimation_abci.payloads import (
    BatchPreparationPayload,
    EmitPayload,
    EstimatePayload,
    FetchingPayload,
    ModelStrategyPayload,
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
    UpdatePayload,
)
from packages.valory.skills.apy_estimation_abci.tools.general import filter_out_numbers


class Event(Enum):
    """Event enumeration for the APY estimation demo."""

    DONE = "done"
    NEGATIVE = "negative"
    NONE = "none"
    ROUND_TIMEOUT = "round_timeout"
    NO_MAJORITY = "no_majority"
    RESET_TIMEOUT = "reset_timeout"
    FULLY_TRAINED = "fully_trained"
    RANDOMNESS_INVALID = "randomness_invalid"
    FILE_ERROR = "file_error"
    NETWORK_ERROR = "network_error"


class SynchronizedData(BaseSynchronizedData):
    """Class to represent the synchronized data.

    This data is replicated by the tendermint application.
    """

    @property
    def history_hash(self) -> str:
        """Get the most voted history hash."""
        return cast(str, self.db.get_strict("history_hash"))

    @property
    def batch_hash(self) -> str:
        """Get the most voted batch hash."""
        return cast(str, self.db.get_strict("batch_hash"))

    @property
    def transformed_history_hash(self) -> str:
        """Get the most voted transformed history hash."""
        return cast(str, self.db.get_strict("transformed_history_hash"))

    @property
    def latest_observation_hist_hash(self) -> str:
        """Get the latest observation's history hash."""
        return cast(str, self.db.get_strict("latest_observation_hist_hash"))

    @property
    def latest_transformation_period(self) -> int:
        """Get the latest period for which a transformation took place."""
        return cast(int, self.db.get_strict("latest_transformation_period"))

    @property
    def most_voted_split(self) -> str:
        """Get the most voted split."""
        return cast(str, self.db.get_strict("most_voted_split"))

    # TODO: remove - not used?
    @property
    def train_hash(self) -> str:
        """Get the most voted train hash."""
        return self.most_voted_split[0 : int(len(self.most_voted_split) / 2)]

    # TODO: remove - not used?
    @property
    def test_hash(self) -> str:
        """Get the most voted test hash."""
        return self.most_voted_split[int(len(self.most_voted_split) / 2) :]

    @property
    def params_hash(self) -> str:
        """Get the params_hash."""
        return cast(str, self.db.get_strict("params_hash"))

    @property
    def models_hash(self) -> str:
        """Get the models_hash."""
        return cast(str, self.db.get_strict("models_hash"))

    @property
    def estimates_hash(self) -> str:
        """Get the estimates_hash."""
        return cast(str, self.db.get_strict("estimates_hash"))

    @property
    def is_most_voted_estimate_set(self) -> bool:
        """Check if estimates_hash is set."""
        return self.db.get("estimates_hash", None) is not None

    @property
    def full_training(self) -> bool:
        """Get the full_training flag."""
        return cast(bool, self.db.get("full_training", False))

    @property
    def n_estimations(self) -> int:
        """Get the n_estimations."""
        return cast(int, self.db.get("n_estimations", 0))

    @property
    def participant_to_estimate(self) -> Mapping[str, EstimatePayload]:
        """Get the `participant_to_estimate`."""
        return cast(
            Mapping[str, EstimatePayload],
            self.db.get_strict("participant_to_estimate"),
        )

    @property
    def participant_to_strategy_votes(self) -> Mapping[str, ModelStrategyPayload]:
        """Get the participant_to_strategy_votes."""
        return cast(
            Mapping[str, ModelStrategyPayload],
            self.db.get_strict("participant_to_strategy_votes"),
        )

    @property
    def participant_to_history(self) -> Mapping[str, FetchingPayload]:
        """Get the participant_to_history."""
        return cast(
            Mapping[str, FetchingPayload], self.db.get_strict("participant_to_history")
        )

    @property
    def participant_to_full_training(self) -> Mapping[str, _TestingPayload]:
        """Get the participant_to_full_training."""
        return cast(
            Mapping[str, _TestingPayload],
            self.db.get_strict("participant_to_full_training"),
        )

    @property
    def participant_to_update(self) -> Mapping[str, UpdatePayload]:
        """Get the participant_to_update."""
        return cast(
            Mapping[str, UpdatePayload], self.db.get_strict("participant_to_update")
        )

    @property
    def participant_to_batch(self) -> Mapping[str, BatchPreparationPayload]:
        """Get the participant_to_batch."""
        return cast(
            Mapping[str, BatchPreparationPayload],
            self.db.get_strict("participant_to_batch"),
        )

    @property
    def participant_to_transform(self) -> Mapping[str, TransformationPayload]:
        """Get the participant_to_transform."""
        return cast(
            Mapping[str, TransformationPayload],
            self.db.get_strict("participant_to_transform"),
        )

    @property
    def participant_to_preprocessing(self) -> Mapping[str, PreprocessPayload]:
        """Get the participant_to_preprocessing."""
        return cast(
            Mapping[str, PreprocessPayload],
            self.db.get_strict("participant_to_preprocessing"),
        )

    @property
    def participant_to_params(self) -> Mapping[str, OptimizationPayload]:
        """Get the participant_to_params."""
        return cast(
            Mapping[str, OptimizationPayload],
            self.db.get_strict("participant_to_params"),
        )

    @property
    def participant_to_training(self) -> Mapping[str, TrainingPayload]:
        """Get the participant_to_training."""
        return cast(
            Mapping[str, TrainingPayload], self.db.get_strict("participant_to_training")
        )

    @property
    def participant_to_batch_preparation(self) -> Mapping[str, BatchPreparationPayload]:
        """Get the participant_to_batch_preparation."""
        return cast(
            Mapping[str, BatchPreparationPayload],
            self.db.get_strict("participant_to_batch_preparation"),
        )

    @property
    def participant_to_emit(self) -> Mapping[str, EmitPayload]:
        """Get the participant_to_emit."""
        return cast(
            Mapping[str, EmitPayload], self.db.get_strict("participant_to_emit")
        )

    @property
    def most_voted_emission_period(self) -> int:
        """Get the most_voted_emission_period."""
        return cast(int, self.db.get_strict("most_voted_emission_period"))


class APYEstimationAbstractRound(AbstractRound[Event, TransactionType], ABC):
    """Abstract round for the APY estimation skill."""

    @property
    def synchronized_data(self) -> SynchronizedData:
        """Return the synchronized data."""
        return cast(SynchronizedData, super().synchronized_data)

    def _return_no_majority_event(self) -> Tuple[SynchronizedData, Event]:
        """
        Trigger the NO_MAJORITY event.

        :return: a new synchronized data and a NO_MAJORITY event
        """
        return self.synchronized_data, Event.NO_MAJORITY


class ModelStrategyRound(VotingRound, APYEstimationAbstractRound):
    """A round that represents the model's strategy selection"""

    allowed_tx_type = ModelStrategyPayload.transaction_type
    done_event = Event.DONE
    negative_event = Event.NEGATIVE
    none_event = Event.NONE
    no_majority_event = Event.NO_MAJORITY
    collection_key = get_name(SynchronizedData.participant_to_strategy_votes)
    synchronized_data_class = SynchronizedData


class CollectHistoryRound(CollectSameUntilThresholdRound, APYEstimationAbstractRound):
    """A round in which agents collect historical data"""

    allowed_tx_type = FetchingPayload.transaction_type
    payload_attribute = get_name(FetchingPayload.history)
    collection_key = get_name(SynchronizedData.participant_to_history)
    selection_key = get_name(SynchronizedData.history_hash)
    synchronized_data_class = SynchronizedData

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Event]]:
        """Process the end of the block."""
        if self.threshold_reached:
            if self.most_voted_payload is None:
                return self.synchronized_data, Event.FILE_ERROR

            if self.most_voted_payload == "":
                return self.synchronized_data, Event.NETWORK_ERROR

            update_kwargs = {
                "synchronized_data_class": self.synchronized_data_class,
                self.collection_key: self.collection,
                self.selection_key: self.most_voted_payload,
            }

            synchronized_data = self.synchronized_data.update(**update_kwargs)
            return synchronized_data, Event.DONE

        if not self.is_majority_possible(
            self.collection, self.synchronized_data.nb_participants
        ):
            return self._return_no_majority_event()

        return None


class CollectLatestHistoryBatchRound(CollectHistoryRound):
    """A round in which agents collect the latest data batch"""

    collection_key = get_name(SynchronizedData.participant_to_batch)
    selection_key = get_name(SynchronizedData.batch_hash)
    synchronized_data_class = SynchronizedData


class TransformRound(CollectSameUntilThresholdRound, APYEstimationAbstractRound):
    """A round in which agents transform data"""

    allowed_tx_type = TransformationPayload.transaction_type
    payload_attribute = get_name(TransformationPayload.transformed_history_hash)
    synchronized_data_class = SynchronizedData

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Event]]:
        """Process the end of the block."""
        if self.threshold_reached and self.most_voted_payload is None:
            return self.synchronized_data, Event.FILE_ERROR

        if self.threshold_reached:
            synchronized_data = self.synchronized_data.update(
                synchronized_data_class=self.synchronized_data_class,
                **{
                    get_name(
                        SynchronizedData.participant_to_transform
                    ): self.collection,
                    get_name(
                        SynchronizedData.transformed_history_hash
                    ): self.most_voted_payload,
                    get_name(SynchronizedData.latest_observation_hist_hash): cast(
                        TransformationPayload, list(self.collection.values())[0]
                    ).latest_observation_hist_hash,
                    get_name(
                        SynchronizedData.latest_transformation_period
                    ): self.synchronized_data.period_count,
                },
            )
            return synchronized_data, Event.DONE

        if not self.is_majority_possible(
            self.collection, self.synchronized_data.nb_participants
        ):
            return self._return_no_majority_event()

        return None


class PreprocessRound(CollectSameUntilThresholdRound, APYEstimationAbstractRound):
    """A round in which the agents preprocess the data"""

    allowed_tx_type = PreprocessPayload.transaction_type
    payload_attribute = get_name(PreprocessPayload.train_test_hash)
    synchronized_data_class = SynchronizedData

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Event]]:
        """Process the end of the block."""
        if self.threshold_reached:
            if self.most_voted_payload is None:
                return self.synchronized_data, Event.FILE_ERROR

            synchronized_data = self.synchronized_data.update(
                synchronized_data_class=self.synchronized_data_class,
                **{
                    get_name(
                        SynchronizedData.participant_to_preprocessing
                    ): self.collection,
                    get_name(
                        SynchronizedData.most_voted_split
                    ): self.most_voted_payload,
                },
            )
            return synchronized_data, Event.DONE

        if not self.is_majority_possible(
            self.collection, self.synchronized_data.nb_participants
        ):
            return self._return_no_majority_event()

        return None


class PrepareBatchRound(CollectSameUntilThresholdRound):
    """A round in which agents prepare a batch of data"""

    allowed_tx_type = BatchPreparationPayload.transaction_type
    payload_attribute = get_name(BatchPreparationPayload.prepared_batch)
    synchronized_data_class = SynchronizedData
    done_event = Event.DONE
    no_majority_event = Event.NO_MAJORITY
    none_event = Event.FILE_ERROR
    collection_key = get_name(SynchronizedData.participant_to_batch_preparation)
    selection_key = get_name(SynchronizedData.latest_observation_hist_hash)


class RandomnessRound(CollectSameUntilThresholdRound, APYEstimationAbstractRound):
    """A round in which a random number is retrieved

    This number is obtained from a distributed randomness beacon. The agents
    need to reach consensus on this number and subsequently use it to seed
    any random number generators.
    """

    allowed_tx_type = RandomnessPayload.transaction_type
    payload_attribute = get_name(RandomnessPayload.randomness)
    synchronized_data_class = SynchronizedData

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Event]]:
        """Process the end of the block."""
        if self.threshold_reached:
            filtered_randomness = filter_out_numbers(self.most_voted_payload)

            if filtered_randomness is None:
                return self.synchronized_data, Event.RANDOMNESS_INVALID

            synchronized_data = self.synchronized_data.update(
                synchronized_data_class=self.synchronized_data_class,
                **{
                    get_name(
                        SynchronizedData.participant_to_randomness
                    ): self.collection,
                    get_name(
                        SynchronizedData.most_voted_randomness
                    ): filtered_randomness,
                },
            )
            return synchronized_data, Event.DONE

        if not self.is_majority_possible(
            self.collection, self.synchronized_data.nb_participants
        ):
            return self._return_no_majority_event()

        return None


class OptimizeRound(CollectSameUntilThresholdRound):
    """A round in which agents agree on the optimal hyperparameters"""

    allowed_tx_type = OptimizationPayload.transaction_type
    payload_attribute = get_name(OptimizationPayload.best_params)
    synchronized_data_class = SynchronizedData
    done_event = Event.DONE
    no_majority_event = Event.NO_MAJORITY
    none_event = Event.FILE_ERROR
    collection_key = get_name(SynchronizedData.participant_to_params)
    selection_key = get_name(SynchronizedData.params_hash)


class TrainRound(CollectSameUntilThresholdRound, APYEstimationAbstractRound):
    """A round in which agents train a model"""

    allowed_tx_type = TrainingPayload.transaction_type
    payload_attribute = get_name(TrainingPayload.models_hash)
    synchronized_data_class = SynchronizedData

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Event]]:
        """Process the end of the block."""
        if self.threshold_reached:
            if self.most_voted_payload is None:
                return self.synchronized_data, Event.FILE_ERROR

            update_params = dict(
                synchronized_data_class=self.synchronized_data_class,
                **{
                    get_name(SynchronizedData.participant_to_training): self.collection,
                    get_name(SynchronizedData.models_hash): self.most_voted_payload,
                },
            )

            if self.synchronized_data.full_training:
                update_params.update(
                    {
                        get_name(SynchronizedData.full_training): True,
                    }
                )
                synchronized_data = self.synchronized_data.update(
                    **update_params,
                )
                return synchronized_data, Event.FULLY_TRAINED

            synchronized_data = self.synchronized_data.update(**update_params)
            return synchronized_data, Event.DONE

        if not self.is_majority_possible(
            self.collection, self.synchronized_data.nb_participants
        ):
            return self._return_no_majority_event()

        return None


class TestRound(CollectSameUntilThresholdRound):
    """A round in which agents test a model"""

    allowed_tx_type = _TestingPayload.transaction_type
    payload_attribute = get_name(_TestingPayload.report_hash)
    synchronized_data_class = SynchronizedData
    done_event = Event.DONE
    no_majority_event = Event.NO_MAJORITY
    none_event = Event.FILE_ERROR
    collection_key = get_name(SynchronizedData.participant_to_full_training)
    selection_key = get_name(SynchronizedData.full_training)


class UpdateForecasterRound(CollectSameUntilThresholdRound):
    """A round in which agents update the forecasting model"""

    allowed_tx_type = UpdatePayload.transaction_type
    payload_attribute = get_name(UpdatePayload.updated_models_hash)
    synchronized_data_class = SynchronizedData
    done_event = Event.DONE
    no_majority_event = Event.NO_MAJORITY
    none_event = Event.FILE_ERROR
    collection_key = get_name(SynchronizedData.participant_to_update)
    selection_key = get_name(SynchronizedData.models_hash)


class EstimateRound(CollectSameUntilThresholdRound, APYEstimationAbstractRound):
    """A round in which agents make predictions using a model"""

    allowed_tx_type = EstimatePayload.transaction_type
    payload_attribute = get_name(EstimatePayload.estimations_hash)
    synchronized_data_class = SynchronizedData

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Event]]:
        """Process the end of the block."""
        if self.threshold_reached:
            if self.most_voted_payload is None:
                return self.synchronized_data, Event.FILE_ERROR

            synchronized_data = self.synchronized_data.update(
                synchronized_data_class=self.synchronized_data_class,
                **{
                    get_name(SynchronizedData.participant_to_estimate): self.collection,
                    get_name(SynchronizedData.n_estimations): cast(
                        SynchronizedData, self.synchronized_data
                    ).n_estimations
                    + 1,
                    get_name(SynchronizedData.estimates_hash): self.most_voted_payload,
                },
            )

            return synchronized_data, Event.DONE

        if not self.is_majority_possible(
            self.collection, self.synchronized_data.nb_participants
        ):
            return self._return_no_majority_event()

        return None


class EmitRound(CollectSameUntilThresholdRound, APYEstimationAbstractRound):
    """A round that represents the emission of the estimates to the backend"""

    allowed_tx_type = EmitPayload.transaction_type
    payload_attribute = get_name(EmitPayload.period_count)
    synchronized_data_class = SynchronizedData
    done_event = Event.DONE
    no_majority_event = Event.NO_MAJORITY
    collection_key = get_name(SynchronizedData.participant_to_emit)
    selection_key = get_name(SynchronizedData.most_voted_emission_period)


class FinishedAPYEstimationRound(DegenerateRound, ABC):
    """A round that represents APY estimation has finished"""


class FailedAPYRound(DegenerateRound, ABC):
    """A round that represents that the period failed"""


class APYEstimationAbciApp(AbciApp[Event]):  # pylint: disable=too-few-public-methods
    """APYEstimationAbciApp

    Initial round: ModelStrategyRound

    Initial states: {ModelStrategyRound}

    Transition states:
        0. ModelStrategyRound
            - done: 1.
            - negative: 10.
            - none: 0.
            - round timeout: 0.
            - no majority: 0.
        1. CollectHistoryRound
            - done: 2.
            - no majority: 1.
            - round timeout: 1.
            - file error: 14.
            - network error: 14.
        2. TransformRound
            - done: 3.
            - no majority: 2.
            - round timeout: 2.
            - file error: 14.
        3. PreprocessRound
            - done: 4.
            - no majority: 3.
            - round timeout: 3.
            - file error: 14.
        4. RandomnessRound
            - done: 5.
            - randomness invalid: 4.
            - no majority: 4.
            - round timeout: 4.
        5. OptimizeRound
            - done: 6.
            - no majority: 5.
            - round timeout: 5.
            - file error: 14.
        6. TrainRound
            - fully trained: 8.
            - done: 7.
            - no majority: 6.
            - round timeout: 6.
            - file error: 14.
        7. TestRound
            - done: 6.
            - no majority: 7.
            - round timeout: 7.
            - file error: 14.
        8. EstimateRound
            - done: 9.
            - round timeout: 8.
            - no majority: 8.
            - file error: 14.
        9. EmitRound
            - done: 13.
            - round timeout: 9.
            - no majority: 9.
        10. CollectLatestHistoryBatchRound
            - done: 11.
            - round timeout: 10.
            - no majority: 10.
            - file error: 14.
            - network error: 14.
        11. PrepareBatchRound
            - done: 12.
            - round timeout: 11.
            - no majority: 11.
            - file error: 14.
        12. UpdateForecasterRound
            - done: 8.
            - round timeout: 12.
            - no majority: 12.
            - file error: 14.
        13. FinishedAPYEstimationRound
        14. FailedAPYRound

    Final states: {FailedAPYRound, FinishedAPYEstimationRound}

    Timeouts:
        round timeout: 30.0
        reset timeout: 30.0
    """

    initial_round_cls: Type[AbstractRound] = ModelStrategyRound
    transition_function: AbciAppTransitionFunction = {
        ModelStrategyRound: {
            Event.DONE: CollectHistoryRound,
            Event.NEGATIVE: CollectLatestHistoryBatchRound,
            Event.NONE: ModelStrategyRound,  # NOTE: unreachable
            Event.ROUND_TIMEOUT: ModelStrategyRound,
            Event.NO_MAJORITY: ModelStrategyRound,
        },
        CollectHistoryRound: {
            Event.DONE: TransformRound,
            Event.NO_MAJORITY: CollectHistoryRound,
            Event.ROUND_TIMEOUT: CollectHistoryRound,
            Event.FILE_ERROR: FailedAPYRound,
            Event.NETWORK_ERROR: FailedAPYRound,
        },
        TransformRound: {
            Event.DONE: PreprocessRound,
            Event.NO_MAJORITY: TransformRound,
            Event.ROUND_TIMEOUT: TransformRound,
            Event.FILE_ERROR: FailedAPYRound,
        },
        PreprocessRound: {
            Event.DONE: RandomnessRound,
            Event.NO_MAJORITY: PreprocessRound,
            Event.ROUND_TIMEOUT: PreprocessRound,
            Event.FILE_ERROR: FailedAPYRound,
        },
        RandomnessRound: {
            Event.DONE: OptimizeRound,
            Event.RANDOMNESS_INVALID: RandomnessRound,
            Event.NO_MAJORITY: RandomnessRound,
            Event.ROUND_TIMEOUT: RandomnessRound,
        },
        OptimizeRound: {
            Event.DONE: TrainRound,
            Event.NO_MAJORITY: OptimizeRound,
            Event.ROUND_TIMEOUT: OptimizeRound,
            Event.FILE_ERROR: FailedAPYRound,
        },
        TrainRound: {
            Event.FULLY_TRAINED: EstimateRound,
            Event.DONE: TestRound,
            Event.NO_MAJORITY: TrainRound,
            Event.ROUND_TIMEOUT: TrainRound,
            Event.FILE_ERROR: FailedAPYRound,
        },
        TestRound: {
            Event.DONE: TrainRound,
            Event.NO_MAJORITY: TestRound,
            Event.ROUND_TIMEOUT: TestRound,
            Event.FILE_ERROR: FailedAPYRound,
        },
        EstimateRound: {
            Event.DONE: EmitRound,
            Event.ROUND_TIMEOUT: EstimateRound,
            Event.NO_MAJORITY: EstimateRound,
            Event.FILE_ERROR: FailedAPYRound,
        },
        EmitRound: {
            Event.DONE: FinishedAPYEstimationRound,
            Event.ROUND_TIMEOUT: EmitRound,
            Event.NO_MAJORITY: EmitRound,
        },
        CollectLatestHistoryBatchRound: {
            Event.DONE: PrepareBatchRound,
            Event.ROUND_TIMEOUT: CollectLatestHistoryBatchRound,
            Event.NO_MAJORITY: CollectLatestHistoryBatchRound,
            Event.FILE_ERROR: FailedAPYRound,
            Event.NETWORK_ERROR: FailedAPYRound,
        },
        PrepareBatchRound: {
            Event.DONE: UpdateForecasterRound,
            Event.ROUND_TIMEOUT: PrepareBatchRound,
            Event.NO_MAJORITY: PrepareBatchRound,
            Event.FILE_ERROR: FailedAPYRound,
        },
        UpdateForecasterRound: {
            Event.DONE: EstimateRound,
            Event.ROUND_TIMEOUT: UpdateForecasterRound,
            Event.NO_MAJORITY: UpdateForecasterRound,
            Event.FILE_ERROR: FailedAPYRound,
        },
        FinishedAPYEstimationRound: {},
        FailedAPYRound: {},
    }
    cross_period_persisted_keys = [
        get_name(SynchronizedData.full_training),
        get_name(SynchronizedData.n_estimations),
        get_name(SynchronizedData.models_hash),
        get_name(SynchronizedData.latest_transformation_period),
        get_name(SynchronizedData.transformed_history_hash),
        get_name(SynchronizedData.latest_observation_hist_hash),
    ]
    final_states: Set[AppState] = {FinishedAPYEstimationRound, FailedAPYRound}
    event_to_timeout: Dict[Event, float] = {
        Event.ROUND_TIMEOUT: 30.0,
        Event.RESET_TIMEOUT: 30.0,
    }
    db_pre_conditions: Dict[AppState, List[str]] = {ModelStrategyRound: []}
    db_post_conditions: Dict[AppState, List[str]] = {
        FinishedAPYEstimationRound: [],
        FailedAPYRound: [],
    }
