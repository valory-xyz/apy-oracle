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


from typing import Any, Dict, List, Optional, Set, Union, ValuesView, cast

from aea.skills.base import SkillContext

from packages.valory.skills.abstract_round_abci.models import ApiSpecs, BaseParams
from packages.valory.skills.abstract_round_abci.models import (
    BenchmarkTool as BaseBenchmarkTool,
)
from packages.valory.skills.abstract_round_abci.models import Requests as BaseRequests
from packages.valory.skills.abstract_round_abci.models import (
    SharedState as BaseSharedState,
)
from packages.valory.skills.apy_estimation_abci.rounds import APYEstimationAbciApp
from packages.valory.skills.apy_estimation_abci.tools.general import UNITS_TO_UNIX


# A tolerance in seconds.
# It is *not* acceptable to calculate the APY value if the diff between two timestamps is not in 24h +- tolerance
APY_TOLERANCE = 0.5 * UNITS_TO_UNIX["hour"]
DAY_IN_UNIX = UNITS_TO_UNIX["day"]


Requests = BaseRequests
BenchmarkTool = BaseBenchmarkTool


class SharedState(BaseSharedState):
    """Keep the current shared state of the skill."""

    abci_app_cls = APYEstimationAbciApp


class RandomnessApi(ApiSpecs):
    """A model that wraps ApiSpecs for randomness api specifications."""


class ServerApi(ApiSpecs):
    """A model for oracle web server api specs."""


class FantomSubgraph(ApiSpecs):
    """A model that wraps ApiSpecs for Fantom subgraph specifications."""


class ETHSubgraph(ApiSpecs):
    """A model that wraps ApiSpecs for ETH subgraph specifications."""


class DEXSubgraph(ApiSpecs):
    """A model that wraps ApiSpecs for DEX subgraph specifications."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize DEX Subgraph."""
        self.bundle_id: int = self._ensure("bundle_id", kwargs, int)
        self.chain_subgraph_name: str = self._ensure("chain_subgraph", kwargs, str)
        super().__init__(*args, **kwargs)


class UniswapSubgraph(DEXSubgraph):
    """A model that wraps DEXSubgraph for Uniswap subgraph specifications."""


class SpookySwapSubgraph(DEXSubgraph):
    """A model that wraps DEXSubgraph for SpookySwap subgraph specifications."""


PairIdsType = Dict[str, List[str]]
ValidatedSubgraphType = Union[DEXSubgraph, ApiSpecs]
ValidatedSubgraphsType = ValuesView[ValidatedSubgraphType]
ValidatedSubgraphsMappingType = Dict[str, ValidatedSubgraphType]
UnvalidatedSubgraphType = Optional[ValidatedSubgraphType]
UnvalidatedSubgraphsMappingType = Dict[str, UnvalidatedSubgraphType]


class SubgraphsMixin:
    """A mixin to handle the subgraphs' information."""

    _necessary_attributes = {"context.params.pair_ids"}
    _utilized_subgraphs: UnvalidatedSubgraphsMappingType
    context: SkillContext

    def __init__(self) -> None:
        """Initialize the mixin object."""
        self._check_attributes()
        utilized_dex_names = set(self.context.params.pair_ids.keys())
        utilized_dex_subgraphs = self._get_subgraphs_mapping(utilized_dex_names)
        utilized_block_names = {
            dex.chain_subgraph_name
            for dex in utilized_dex_subgraphs.values()
            if dex is not None
        }
        utilized_block_subgraphs = self._get_subgraphs_mapping(utilized_block_names)
        self.__dict__["_utilized_subgraphs"] = {
            **utilized_dex_subgraphs,
            **utilized_block_subgraphs,
        }
        self._validate_utilized_subgraphs()

    def _check_attributes(self) -> None:
        """Checks that the Mixin is subclassed by a class which has the necessary attributes."""
        missing_attrs = []
        for attr in self._necessary_attributes:
            part_checked_so_far, path_so_far = self, ""
            for part in attr.split("."):
                try:
                    path_so_far += f"{part}"
                    part_checked_so_far = getattr(part_checked_so_far, part)
                except AttributeError:
                    missing_attrs.append(path_so_far)
                finally:
                    path_so_far += "."

        if missing_attrs:
            raise AttributeError(
                f"`SubgraphsMixin` is missing attribute(s): {missing_attrs}."
            )

    def _validate_utilized_subgraphs(self) -> None:
        """Check that the utilized subgraphs are valid, i.e., they are defined in the `skill.yaml` config file."""
        unknown_subgraphs = {
            name
            for name, subgraph in self._utilized_subgraphs.items()
            if subgraph is None
        }
        if unknown_subgraphs:
            raise ValueError(
                f"Subgraph(s) {unknown_subgraphs} not recognized. "
                "Please specify them in the `skill.yaml` config file and `models.py`."
            )

    def _get_subgraphs_mapping(
        self, names: Set[str]
    ) -> UnvalidatedSubgraphsMappingType:
        """Get subgraphs mapped to their names."""
        return {name: self._try_get_subgraph(name) for name in names}

    def _try_get_subgraph(self, name: str) -> UnvalidatedSubgraphType:
        """Try to get a subgraph by its name. If it does not exist, return `None`"""
        return getattr(self.context, name, None)

    def get_subgraph(self, name: str) -> ValidatedSubgraphType:
        """Get a subgraph by its name. If it does not exist, an `AttributeError` is raised."""
        return getattr(self.context, name)

    @property
    def utilized_subgraphs(self) -> ValidatedSubgraphsType:
        """Get the utilized Subgraphs."""
        return cast(ValidatedSubgraphsMappingType, self._utilized_subgraphs).values()


class APYParams(BaseParams):  # pylint: disable=too-many-instance-attributes
    """Parameters."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the parameters object."""
        # end can be `None`; this means that the current time will be used
        # It is set in the behaviour using the last synced timestamp among the agents
        self.end: Optional[int] = self._ensure("history_end", kwargs, Optional[int])
        self.interval: int = self._ensure("history_interval_in_unix", kwargs, int)
        self.interval_not_acceptable = not (
            DAY_IN_UNIX - APY_TOLERANCE <= self.interval <= DAY_IN_UNIX + APY_TOLERANCE
        )
        self.n_observations: int = self._ensure("n_observations", kwargs, int)
        self.optimizer_params: Dict[
            str, Union[None, bool, int, float, str]
        ] = self._ensure(
            "optimizer", kwargs, Dict[str, Union[None, bool, int, float, str]]
        )
        self.testing: Dict[str, int] = self._ensure("testing", kwargs, Dict[str, int])
        self.estimation: Dict[str, int] = self._ensure(
            "estimation", kwargs, Dict[str, int]
        )
        self.n_estimations_before_retrain: int = self._ensure(
            "n_estimations_before_retrain", kwargs, int
        )
        self.pair_ids: PairIdsType = self._ensure("pair_ids", kwargs, PairIdsType)
        self.ipfs_domain_name: str = self._ensure("ipfs_domain_name", kwargs, str)
        self.is_broadcasting_to_server: bool = self._ensure(
            "broadcast_to_server", kwargs, bool
        )
        self.decimals: int = self._ensure("decimals", kwargs, int)
        super().__init__(*args, **kwargs)

        self.__validate_params()

    @property
    def ts_length(self) -> int:
        """The length of the timeseries in seconds."""
        return self.n_observations * self.interval

    def __validate_params(self) -> None:
        """Validate the given parameters."""
        # Eventually, we should probably validate all the parameters. E.g., `ts_length` should be < `end`
        for param_name in ("timeout", "window_size"):
            param_val = self.optimizer_params.get(param_name)
            if param_val is not None and not isinstance(param_val, int):
                raise ValueError(
                    f"Optimizer's parameter `{param_name}` can be either of type `int` or `None`. "
                    f"{type(param_val)} was given."
                )
            # if the value did not exist in the config, then we set it to the default (None) returned from `.get` method
            self.optimizer_params[param_name] = param_val
