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

"""This module contains the transaction payloads for the APY estimation app."""

from dataclasses import dataclass
from typing import Optional

from packages.valory.skills.abstract_round_abci.base import (
    BaseTxPayload as BaseAPYPayload,
)


@dataclass(frozen=True)
class ModelStrategyPayload(BaseAPYPayload):
    """Represent a transaction payload of type 'validate'."""

    vote: Optional[bool] = None


@dataclass(frozen=True)
class RandomnessPayload(BaseAPYPayload):
    """Represent a transaction payload of type 'randomness'."""

    randomness: Optional[int]


@dataclass(frozen=True)
class FetchingPayload(BaseAPYPayload):
    """Represent a transaction payload of type 'fetching'."""

    history: Optional[str]


@dataclass(frozen=True)
class TransformationPayload(BaseAPYPayload):
    """Represent a transaction payload of type 'transformation'."""

    transformed_history_hash: Optional[str]
    latest_observation_hist_hash: Optional[str]
    latest_transformation_period: Optional[int]


@dataclass(frozen=True)
class PreprocessPayload(BaseAPYPayload):
    """Represent a transaction payload of type 'preprocess'."""

    train_test_hash: Optional[str]


@dataclass(frozen=True)
class BatchPreparationPayload(BaseAPYPayload):
    """Represent a transaction payload of type 'batch_preparation'."""

    prepared_batch: Optional[str]


@dataclass(frozen=True)
class OptimizationPayload(BaseAPYPayload):
    """Represent a transaction payload of type 'optimization'."""

    best_params: Optional[str]


@dataclass(frozen=True)
class TrainingPayload(BaseAPYPayload):
    """Represent a transaction payload of type 'training'."""

    models_hash: Optional[str]


@dataclass(frozen=True)
class TestingPayload(BaseAPYPayload):
    """Represent a transaction payload of type 'testing'."""

    report_hash: Optional[str]


@dataclass(frozen=True)
class UpdatePayload(BaseAPYPayload):
    """Represent a transaction payload of type 'update'."""

    updated_models_hash: Optional[str]


@dataclass(frozen=True)
class EstimatePayload(BaseAPYPayload):
    """Represent a transaction payload of type 'estimate'."""

    n_estimations: Optional[int]
    estimations_hash: Optional[str]


@dataclass(frozen=True)
class EmitPayload(BaseAPYPayload):
    """Represent a transaction payload of type 'emit'."""

    period_count: int
