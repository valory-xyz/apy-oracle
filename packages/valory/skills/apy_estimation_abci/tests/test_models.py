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

import re
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type, TypedDict, Union
from unittest.mock import MagicMock

import hypothesis.strategies as st
import pytest
from hypothesis import database, given, settings

from packages.valory.skills.abstract_round_abci.models import ApiSpecs
from packages.valory.skills.apy_estimation_abci.models import (
    APYParams,
    SharedState,
    SpookySwapSubgraph,
    SubgraphsMixin,
)
from packages.valory.skills.apy_estimation_abci.rounds import APYEstimationAbciApp


APYParamsArgsType = Tuple[str]


class APYParamsKwargsType(TypedDict):
    """Typed dict for the APY kwargs."""

    skill_context: MagicMock
    tendermint_url: str
    tendermint_com_url: str
    tendermint_check_sleep_delay: int
    tendermint_max_retries: int
    reset_tendermint_after: int
    ipfs_domain_name: str
    max_healthcheck: int
    round_timeout_seconds: float
    sleep_time: int
    retry_attempts: int
    retry_timeout: int
    request_timeout: float
    request_retry_delay: float
    reset_pause_duration: int
    drand_public_key: str
    history_interval_in_unix: int
    n_observations: int
    optimizer: Dict[str, Union[None, str, int]]
    testing: Dict[str, int]
    estimation: Dict[str, int]
    n_estimations_before_retrain: int
    pair_ids: Dict[str, List[str]]
    service_id: str
    service_registry_address: str
    keeper_timeout: float
    cleanup_history_depth: int
    backwards_compatible: bool
    decimals: int
    genesis_config: dict
    broadcast_to_server: bool
    cleanup_history_depth_current: Optional[int]
    tx_timeout: float
    max_attempts: int
    on_chain_service_id: int
    share_tm_config_on_startup: bool
    tendermint_p2p_url: str
    setup: Dict[str, Any]
    history_end: Optional[int]
    use_termination: bool


APY_PARAMS_ARGS = ("test",)
APY_PARAMS_KWARGS = APYParamsKwargsType(
    skill_context=MagicMock(skill_id="test"),
    tendermint_url="test",
    tendermint_com_url="test",
    tendermint_check_sleep_delay=0,
    tendermint_max_retries=0,
    reset_tendermint_after=0,
    ipfs_domain_name="test",
    max_healthcheck=0,
    round_timeout_seconds=0.1,
    sleep_time=0,
    retry_attempts=0,
    retry_timeout=0,
    request_timeout=0.1,
    request_retry_delay=0.1,
    reset_pause_duration=10,
    drand_public_key="test",
    history_interval_in_unix=86400,
    n_observations=10,
    optimizer={"timeout": 0, "window_size": 0},
    testing={"test": 0},
    estimation={"test": 0},
    n_estimations_before_retrain=1,
    pair_ids={"test": ["not_supported"], "spooky_subgraph": ["supported"]},
    service_id="apy_estimation",
    service_registry_address="0xa51c1fc2f0d1a1b8494ed1fe312d7c3a78ed91c0",
    keeper_timeout=30.0,
    cleanup_history_depth=0,
    backwards_compatible=False,
    decimals=5,
    genesis_config={
        "genesis_time": "0",
        "chain_id": "chain",
        "consensus_params": {
            "block": {"max_bytes": "0", "max_gas": "0", "time_iota_ms": "0"},
            "evidence": {
                "max_age_num_blocks": "0",
                "max_age_duration": "0",
                "max_bytes": "0",
            },
            "validator": {"pub_key_types": ["test"]},
            "version": {},
        },
        "voting_power": "0",
    },
    broadcast_to_server=False,
    cleanup_history_depth_current=None,
    tx_timeout=0.1,
    max_attempts=0,
    on_chain_service_id=0,
    share_tm_config_on_startup=False,
    tendermint_p2p_url="test",
    setup={"test": [0]},
    history_end=0,
    use_termination=False,
    service_endpoint_base="https://dummy_service.autonolas.tech/",
)


class TestSharedState:
    """Test SharedState(Model) class."""

    def test_setup(
        self,
        shared_state: SharedState,
    ) -> None:
        """Test setup."""
        shared_state.context.params.setup_params = {"test": []}
        shared_state.setup()
        assert shared_state.abci_app_cls == APYEstimationAbciApp


class TestAPYParams:
    """Test `APYParams`"""

    @staticmethod
    @given(n_observations=st.integers(), interval=st.integers())
    @settings(deadline=None, database=database.InMemoryExampleDatabase())
    def test_ts_length(n_observations: int, interval: int) -> None:
        """Test `ts_length` property."""
        args = APY_PARAMS_ARGS
        # TypedDict can’t be used for specifying the type of a **kwargs argument: https://peps.python.org/pep-0589/
        kwargs: dict = deepcopy(APY_PARAMS_KWARGS)  # type: ignore
        kwargs["n_observations"] = n_observations
        kwargs["history_interval_in_unix"] = interval
        params = APYParams(*args, **kwargs)

        expected = n_observations * interval
        assert params.ts_length == expected

    @staticmethod
    @pytest.mark.parametrize("param_value", (None, "not_an_int", 0))
    def test__validate_params(param_value: Union[None, str, int]) -> None:
        """Test `__validate_params`."""
        args = APY_PARAMS_ARGS
        # TypedDict can’t be used for specifying the type of a **kwargs argument: https://peps.python.org/pep-0589/
        kwargs: dict = deepcopy(APY_PARAMS_KWARGS)  # type: ignore
        kwargs["optimizer"]["timeout"] = param_value
        kwargs["optimizer"]["window_size"] = param_value

        if param_value is not None and not isinstance(param_value, int):
            with pytest.raises(ValueError):
                APYParams(*args, **kwargs)

            kwargs["optimizer"]["timeout"] = "None"  # type: ignore
            # set the voting power again because the previous `APYParams` call has popped it.
            kwargs["genesis_config"]["voting_power"] = "0"  # type: ignore

            with pytest.raises(ValueError):
                APYParams(*args, **kwargs)

            return

        apy_params = APYParams(*args, **kwargs)
        assert apy_params.optimizer_params["timeout"] is param_value
        assert apy_params.optimizer_params["window_size"] is param_value


class TestSubgraphsMixin:
    """Test `SubgraphsMixin`."""

    class DummyMixinUsage(APYParams, SubgraphsMixin):
        """Dummy class that utilizes the `SubgraphsMixin`."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Initialize `APYEstimationBaseBehaviour`."""
            super().__init__(*args, **kwargs)
            subgraph_args = ["spooky_name"]
            subgraph_kwargs = {
                "skill_context": MagicMock(),
                "url": "url",
                "api_id": "spooky_api_id",
                "method": "method",
                "bundle_id": 0,
                "chain_subgraph": "chain_subgraph",
                "headers": OrderedDict(),
                "parameters": OrderedDict(),
            }
            self.context.test = None
            self.context.spooky_subgraph = SpookySwapSubgraph(
                *subgraph_args, **subgraph_kwargs
            )
            subgraph_args[0] = "chain_name"
            subgraph_kwargs["api_id"] = "chain_api_id"
            self.context.chain_subgraph = ApiSpecs(*subgraph_args, **subgraph_kwargs)
            self.context.params = MagicMock(pair_ids=self.pair_ids)
            SubgraphsMixin.__init__(self)

    dummy_mixin_usage: DummyMixinUsage

    @classmethod
    def setup_class(cls) -> None:
        """Initialize a `TestSubgraphsMixin`."""
        # TypedDict can’t be used for specifying the type of a **kwargs argument: https://peps.python.org/pep-0589/
        kwargs: dict = deepcopy(APY_PARAMS_KWARGS)  # type: ignore
        del kwargs["pair_ids"]["test"]
        cls.dummy_mixin_usage = TestSubgraphsMixin.DummyMixinUsage(
            *APY_PARAMS_ARGS, **kwargs
        )

    @staticmethod
    def test_incorrect_initialization() -> None:
        """Test `SubgraphsMixin`'s `__init__` when not subclassed properly."""
        with pytest.raises(
            AttributeError,
            match=re.escape(
                "`SubgraphsMixin` is missing attribute(s): "
                "['context', 'context.params', 'context.params.pair_ids']."
            ),
        ):
            SubgraphsMixin()

    @staticmethod
    def test_initialization_unsupported_subgraph() -> None:
        """Test `SubgraphsMixin`'s `__init__` when subclassed properly, but an unsupported subgraph is given."""
        # TypedDict can’t be used for specifying the type of a **kwargs argument: https://peps.python.org/pep-0589/
        kwargs: dict = deepcopy(APY_PARAMS_KWARGS)  # type: ignore

        with pytest.raises(
            ValueError,
            match=re.escape(
                "Subgraph(s) {'test'} not recognized. "
                "Please specify them in the `skill.yaml` config file and `models.py`."
            ),
        ):
            TestSubgraphsMixin.DummyMixinUsage(*APY_PARAMS_ARGS, **kwargs)

    @pytest.mark.parametrize(
        "subgraph, name, id_",
        (
            ("spooky_subgraph", "spooky_name", "spooky_api_id"),
            ("chain_subgraph", "chain_name", "chain_api_id"),
        ),
    )
    def test_initialization(self, subgraph: str, name: str, id_: str) -> None:
        """Test `SubgraphsMixin`'s `__init__` when subclassed properly, and only unsupported subgraphs are given."""
        assert subgraph in self.dummy_mixin_usage._utilized_subgraphs.keys()
        subgraph_instance = getattr(self.dummy_mixin_usage.context, subgraph)
        assert isinstance(subgraph_instance, ApiSpecs)
        assert subgraph_instance.name == name
        assert subgraph_instance.api_id == id_

    def test_utilized_subgraphs(self) -> None:
        """Test `utilized_subgraphs` property."""
        assert [
            subgraph.name for subgraph in self.dummy_mixin_usage.utilized_subgraphs
        ] == ["spooky_name", "chain_name"]

    @pytest.mark.parametrize(
        "subgraph, name, type_, id_",
        (
            ("spooky_subgraph", "spooky_name", SpookySwapSubgraph, "spooky_api_id"),
            ("chain_subgraph", "chain_name", ApiSpecs, "chain_api_id"),
        ),
    )
    def test_get_subgraph(
        self, subgraph: str, name: str, type_: Type[ApiSpecs], id_: str
    ) -> None:
        """Test `get_subgraph` method."""
        subgraph_instance = self.dummy_mixin_usage.get_subgraph(subgraph)
        assert subgraph_instance == getattr(self.dummy_mixin_usage.context, subgraph)
        assert type(subgraph_instance) == type_
        assert subgraph_instance.name == name
        assert subgraph_instance.api_id == id_
