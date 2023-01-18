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

"""Integration tests for the valory/apy_estimation_abci skill."""


from pathlib import Path

import pytest
from aea.configurations.data_types import PublicId
from aea_test_autonomy.base_test_classes.agents import (
    BaseTestEnd2EndExecution,
    RoundChecks,
)
from aea_test_autonomy.configurations import KEY_PAIRS
from aea_test_autonomy.fixture_helpers import (  # noqa: F401
    UseACNNode,
    UseGnosisSafeHardHatNet,
    abci_host,
    abci_port,
    acn_config,
    acn_node,
    flask_tendermint,
    gnosis_safe_hardhat_scope_function,
    hardhat_addr,
    hardhat_port,
    key_pairs,
    tendermint_port,
)

from packages.valory.skills.abstract_round_abci.tests.test_io.test_ipfs import (  # noqa: F401
    ipfs_daemon,
)
from packages.valory.skills.apy_estimation_abci.rounds import (
    CollectHistoryRound,
    CollectLatestHistoryBatchRound,
    EmitRound,
    EstimateRound,
    ModelStrategyRound,
    OptimizeRound,
    PrepareBatchRound,
    PreprocessRound,
    RandomnessRound,
    TestRound,
    TrainRound,
    TransformRound,
    UpdateForecasterRound,
)
from packages.valory.skills.reset_pause_abci.rounds import ResetAndPauseRound


HAPPY_PATH = (
    RoundChecks(ModelStrategyRound.auto_round_id()),
    RoundChecks(CollectHistoryRound.auto_round_id()),
    RoundChecks(TransformRound.auto_round_id()),
    RoundChecks(PreprocessRound.auto_round_id()),
    RoundChecks(RandomnessRound.auto_round_id()),
    RoundChecks(OptimizeRound.auto_round_id()),
    RoundChecks(TrainRound.auto_round_id()),
    RoundChecks(TrainRound.auto_round_id(), success_event="FULLY_TRAINED"),
    RoundChecks(TestRound.auto_round_id()),
    RoundChecks(EstimateRound.auto_round_id(), n_periods=2),
    RoundChecks(EmitRound.auto_round_id(), n_periods=2),
    RoundChecks(ResetAndPauseRound.auto_round_id(), n_periods=2),
    RoundChecks(
        ModelStrategyRound.auto_round_id(), n_periods=2, success_event="NEGATIVE"
    ),
    RoundChecks(CollectLatestHistoryBatchRound.auto_round_id(), n_periods=2),
    RoundChecks(PrepareBatchRound.auto_round_id(), n_periods=2),
    RoundChecks(UpdateForecasterRound.auto_round_id(), n_periods=2),
)


@pytest.mark.usefixtures("ipfs_daemon")
class BaseTestABCIAPYEstimationSkillNormalExecution(
    BaseTestEnd2EndExecution, UseACNNode, UseGnosisSafeHardHatNet
):
    """Base class for the APY estimation e2e tests."""

    agent_package = "valory/apy_estimation:0.1.0"
    skill_package = "valory/apy_estimation_chained_abci:0.1.0"
    happy_path = HAPPY_PATH
    ROUND_TIMEOUT_SECONDS = 480
    wait_to_finish = 480
    __args_prefix = f"vendor.valory.skills.{PublicId.from_str(skill_package).name}.models.params.args"
    extra_configs = [
        {
            "dotted_path": f"{__args_prefix}.ipfs_domain_name",
            "value": "/dns/localhost/tcp/5001/http",
        },
        {
            "dotted_path": f"{__args_prefix}.optimizer.timeout",
            "value": 1,
        },
    ]
    package_registry_src_rel = Path(__file__).parents[4]
    key_pairs_override = KEY_PAIRS[:4]

    def prepare_and_launch(self, nb_nodes: int) -> None:
        """Prepare and launch the agents."""
        self.key_pairs = self.key_pairs_override
        super().prepare_and_launch(nb_nodes)


@pytest.mark.parametrize("nb_nodes", (1,))
class TestABCIAPYEstimationSingleAgent(BaseTestABCIAPYEstimationSkillNormalExecution):
    """Test the ABCI apy_estimation_abci skill with only one agent."""

    key_pairs_override = [KEY_PAIRS[4]]


@pytest.mark.parametrize("nb_nodes", (2,))
class TestABCIAPYEstimationTwoAgents(BaseTestABCIAPYEstimationSkillNormalExecution):
    """Test the ABCI apy_estimation_abci skill with two agents."""

    key_pairs_override = KEY_PAIRS[5:7]


@pytest.mark.parametrize("nb_nodes", (4,))
class TestABCIAPYEstimationFourAgents(BaseTestABCIAPYEstimationSkillNormalExecution):
    """Test the ABCI apy_estimation_abci skill with four agents."""
