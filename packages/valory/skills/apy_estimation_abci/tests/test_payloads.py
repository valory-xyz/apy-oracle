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

"""Test the payloads.py module of the skill."""

# pylint: skip-file

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


class TestPayloads:
    """Test for `Payloads`."""

    @staticmethod
    def test_model_strategy_payload() -> None:
        """Test `ModelStrategyPayload`"""
        payload = ModelStrategyPayload(sender="sender", vote=True)

        assert payload.vote is True
        assert payload.data == {"vote": True}

    @staticmethod
    def test_fetching_payload() -> None:
        """Test `FetchingPayload`"""
        payload = FetchingPayload(
            sender="sender",
            history="x0",
        )

        assert payload.history == "x0"
        assert payload.data == {"history": "x0"}

    @staticmethod
    def test_transformation_payload() -> None:
        """Test `TransformationPayload`"""
        payload = TransformationPayload(
            sender="sender",
            transformed_history_hash="x0",
            latest_observation_hist_hash="x1",
        )

        assert payload.transformed_history_hash == "x0"
        assert payload.latest_observation_hist_hash == "x1"
        assert payload.data == {
            "transformed_history_hash": "x0",
            "latest_observation_hist_hash": "x1",
        }

    @staticmethod
    def test_preprocess_payload() -> None:
        """Test `PreprocessPayload`"""
        payload = PreprocessPayload(sender="sender", train_test_hash="x0")
        assert payload.train_test_hash == "x0"
        assert payload.data == {
            "train_test_hash": "x0",
        }

    @staticmethod
    def test_randomness_payload() -> None:
        """Test `RandomnessPayload`"""

        payload = RandomnessPayload(sender="sender", round_id=1, randomness="1")

        assert payload.round_id == 1
        assert payload.randomness == "1"
        assert payload.data == {"round_id": 1, "randomness": "1"}

    @staticmethod
    def test_batch_preparation_payload() -> None:
        """Test `BatchPreparationPayload`"""
        payload = BatchPreparationPayload(
            sender="sender",
            prepared_batch="x0",
        )

        assert payload.prepared_batch == "x0"
        assert payload.data == {"prepared_batch": "x0"}

    @staticmethod
    def test_optimization_payload() -> None:
        """Test `OptimizationPayload`"""
        payload = OptimizationPayload(
            sender="sender",
            best_params="x0",
        )

        assert payload.best_params == "x0"
        assert payload.data == {"best_params": "x0"}

    @staticmethod
    def test_training_payload() -> None:
        """Test `TrainingPayload`"""
        payload = TrainingPayload(sender="sender", models_hash="x0")

        assert payload.models_hash == "x0"
        assert payload.data == {"models_hash": "x0"}

    @staticmethod
    def test_testing_payload() -> None:
        """Test `TestingPayload`"""
        payload = _TestingPayload(sender="sender", report_hash="x0")

        assert payload.report_hash == "x0"
        assert payload.data == {"report_hash": "x0"}

    @staticmethod
    def test_update_payload() -> None:
        """Test `UpdatePayload`"""
        payload = UpdatePayload(sender="sender", updated_models_hash="x0")

        assert payload.updated_models_hash == "x0"
        assert payload.data == {"updated_models_hash": "x0"}

    @staticmethod
    def test_estimate_payload() -> None:
        """Test `EstimatePayload`"""
        payload = EstimatePayload(sender="sender", estimations_hash="test_hash")

        assert payload.estimations_hash == "test_hash"
        assert payload.data == {"estimations_hash": "test_hash"}

    @staticmethod
    def test_emit_payload() -> None:
        """Test `EmitPayload`"""
        payload = EmitPayload(sender="sender", period_count=0)

        assert payload.period_count == 0
        assert payload.data == {"period_count": 0}
