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

"""Test the models.py module of the skill."""

# pylint: skip-file

from unittest import mock

import pytest

from packages.valory.skills.apy_estimation_abci.rounds import Event
from packages.valory.skills.apy_estimation_chained_abci.composition import (
    APYEstimationChainedAbciApp,
)
from packages.valory.skills.apy_estimation_chained_abci.models import (
    MARGIN,
    SharedState,
)
from packages.valory.skills.reset_pause_abci.rounds import Event as ResetPauseEvent


@pytest.fixture
def shared_state() -> SharedState:
    """Initialize a test shared state."""
    return SharedState(name="", skill_context=mock.MagicMock())


class TestSharedState:
    """Test SharedState(Model) class."""

    def test_setup(
        self,
        shared_state: SharedState,
    ) -> None:
        """Test setup."""
        shared_state.context.params.setup_params = {"test": []}
        shared_state.setup()
        assert (
            APYEstimationChainedAbciApp.event_to_timeout[Event.ROUND_TIMEOUT]
            == shared_state.context.params.round_timeout_seconds
        )
        assert (
            APYEstimationChainedAbciApp.event_to_timeout[
                ResetPauseEvent.RESET_AND_PAUSE_TIMEOUT
            ]
            == shared_state.context.params.reset_pause_duration + MARGIN
        )
