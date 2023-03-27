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

"""Custom objects for the APY estimation ABCI application."""


from packages.valory.skills.abstract_round_abci.models import (
    BenchmarkTool as BaseBenchmarkTool,
)
from packages.valory.skills.abstract_round_abci.models import Requests as BaseRequests
from packages.valory.skills.abstract_round_abci.models import (
    SharedState as BaseSharedState,
)
from packages.valory.skills.apy_estimation_abci.models import (
    APYParams as APYEstimationAPYParams,
)
from packages.valory.skills.apy_estimation_abci.models import (
    ETHSubgraph as APYEstimationETHSubgraph,
)
from packages.valory.skills.apy_estimation_abci.models import (
    FantomSubgraph as APYEstimationFantomSubgraph,
)
from packages.valory.skills.apy_estimation_abci.models import (
    RandomnessApi as APYEstimationRandomnessApi,
)
from packages.valory.skills.apy_estimation_abci.models import (
    ServerApi as APYEstimationServerApi,
)
from packages.valory.skills.apy_estimation_abci.models import (
    SpookySwapSubgraph as APYEstimationSpookySwapSubgraph,
)
from packages.valory.skills.apy_estimation_abci.models import (
    UniswapSubgraph as APYEstimationUniswapSubgraph,
)
from packages.valory.skills.apy_estimation_abci.rounds import Event
from packages.valory.skills.apy_estimation_chained_abci.composition import (
    APYEstimationChainedAbciApp,
)
from packages.valory.skills.reset_pause_abci.rounds import Event as ResetPauseEvent


Requests = BaseRequests
BenchmarkTool = BaseBenchmarkTool
RandomnessApi = APYEstimationRandomnessApi
FantomSubgraph = APYEstimationFantomSubgraph
ETHSubgraph = APYEstimationETHSubgraph
UniswapSubgraph = APYEstimationUniswapSubgraph
SpookySwapSubgraph = APYEstimationSpookySwapSubgraph
ServerApi = APYEstimationServerApi
APYParams = APYEstimationAPYParams

MARGIN = 5


class SharedState(BaseSharedState):
    """Keep the current shared state of the skill."""

    abci_app_cls = APYEstimationChainedAbciApp

    def setup(self) -> None:
        """Set up."""
        super().setup()

        event_to_timeout_overrides = {
            Event.ROUND_TIMEOUT: self.context.params.round_timeout_seconds,
            ResetPauseEvent.RESET_AND_PAUSE_TIMEOUT: self.context.params.reset_pause_duration
            + MARGIN,
        }

        for event, override in event_to_timeout_overrides.items():
            APYEstimationChainedAbciApp.event_to_timeout[event] = override
