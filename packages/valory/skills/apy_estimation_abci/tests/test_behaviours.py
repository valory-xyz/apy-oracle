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

"""Tests for valory/apy_estimation_abci skill's behaviours."""

# pylint: skip-file

import binascii
import itertools
import json
import logging
import os
import shutil
import time
from builtins import dict
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from enum import Enum
from itertools import product
from math import ceil
from multiprocessing.pool import AsyncResult
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
)
from unittest import mock
from uuid import uuid4

import numpy as np
import optuna
import pandas as pd
import pytest
from _pytest.logging import LogCaptureFixture
from _pytest.monkeypatch import MonkeyPatch
from aea.skills.tasks import TaskManager
from hypothesis import HealthCheck, assume, database, given, settings
from hypothesis import strategies as st

from packages.valory.skills.abstract_round_abci.base import AbciAppDB
from packages.valory.skills.abstract_round_abci.behaviour_utils import BaseBehaviour
from packages.valory.skills.abstract_round_abci.models import ApiSpecs
from packages.valory.skills.abstract_round_abci.test_tools.base import (
    FSMBehaviourBaseCase,
)
from packages.valory.skills.apy_estimation_abci import tasks
from packages.valory.skills.apy_estimation_abci.behaviours import (
    APYEstimationBaseBehaviour,
    EmitEstimatesBehaviour,
    EstimateBehaviour,
    EstimatorRoundBehaviour,
    FetchBatchBehaviour,
    FetchBehaviour,
    ModelStrategyBehaviour,
    OptimizeBehaviour,
    PrepareBatchBehaviour,
    PreprocessBehaviour,
    RandomnessBehaviour,
)
from packages.valory.skills.apy_estimation_abci.behaviours import (
    TestBehaviour as _TestBehaviour,
)
from packages.valory.skills.apy_estimation_abci.behaviours import (
    TrainBehaviour,
    TransformBehaviour,
    UpdateForecasterBehaviour,
)
from packages.valory.skills.apy_estimation_abci.ml.forecasting import (
    PoolIdToForecasterType,
    PoolIdToTestReportType,
)
from packages.valory.skills.apy_estimation_abci.ml.optimization import (
    PoolToHyperParamsWithStatusType,
)
from packages.valory.skills.apy_estimation_abci.ml.preprocessing import (
    prepare_pair_data,
)
from packages.valory.skills.apy_estimation_abci.models import (
    DAY_IN_UNIX,
    SubgraphsMixin,
)
from packages.valory.skills.apy_estimation_abci.rounds import Event, SynchronizedData
from packages.valory.skills.apy_estimation_abci.tests.conftest import DummyPipeline
from packages.valory.skills.apy_estimation_abci.tools.etl import ResponseItemType
from packages.valory.skills.apy_estimation_abci.tools.general import UNITS_TO_UNIX
from packages.valory.skills.apy_estimation_abci.tools.queries import SAFE_BLOCK_TIME


PACKAGE_DIR = Path(__file__).parents[1]
SLEEP_TIME_TWEAK = 0.01
N_OBSERVATIONS = 10
HISTORY_INTERVAL = 86400
HISTORY_END = 1655136875


@pytest.fixture(scope="session", autouse=True)
def hypothesis_cleanup() -> Generator:
    """Fixture to remove hypothesis directory after tests."""
    yield
    hypothesis_dir = PACKAGE_DIR / ".hypothesis"
    if hypothesis_dir.exists():
        with suppress(OSError, PermissionError):
            shutil.rmtree(hypothesis_dir)


class DummyAsyncResult(object):
    """Dummy class for AsyncResult."""

    def __init__(
        self,
        task_result: Any,
        ready: bool = True,
    ) -> None:
        """Initialize class."""

        self.id = uuid4()
        self._ready = ready
        self._task_result = task_result

    def ready(
        self,
    ) -> bool:
        """Returns bool"""
        return self._ready

    def get(
        self,
    ) -> Any:
        """Returns task result."""
        return self._task_result


def wrap_dummy_ipfs_operation(return_value: Any) -> Callable:
    """Wrap dummy_get_from_ipfs."""

    def dummy_ipfs_operation(*args: Any, **kwargs: Any) -> Generator[None, None, Any]:
        """A mock for an IPFS operation."""
        yield
        return return_value

    return dummy_ipfs_operation


class APYEstimationFSMBehaviourBaseCase(FSMBehaviourBaseCase):
    """Base case for testing APYEstimation FSMBehaviour."""

    path_to_skill = PACKAGE_DIR

    behaviour: EstimatorRoundBehaviour
    behaviour_class: Type[APYEstimationBaseBehaviour]
    next_behaviour_class: Type[APYEstimationBaseBehaviour]
    synchronized_data: SynchronizedData

    @classmethod
    def setup_class(cls, **kwargs: Any) -> None:
        """Set up the test class."""
        super().setup_class(
            param_overrides={"ipfs_domain_name": "/dns/localhost/tcp/5001/http"}
        )

    def setup(self, **kwargs: Any) -> None:
        """Set up the test method."""
        super().setup()
        assert self.behaviour.current_behaviour is not None
        self.behaviour.current_behaviour.batch = False
        self.behaviour.current_behaviour.params.__dict__[
            "n_observations"
        ] = N_OBSERVATIONS
        self.behaviour.current_behaviour.params.__dict__["interval"] = HISTORY_INTERVAL
        self.behaviour.current_behaviour.params.__dict__["end"] = HISTORY_END
        self.behaviour.current_behaviour.params.__dict__[
            "interval_not_acceptable"
        ] = False

        for api in (
            "randomness_api",
            "ethereum_subgraph",
            "spooky_subgraph",
            "uniswap_subgraph",
        ):
            api_instance = getattr(self.behaviour.current_behaviour.context, api)
            api_instance.retries_info.backoff_factor = SLEEP_TIME_TWEAK

        self.synchronized_data = SynchronizedData(
            AbciAppDB(
                setup_data={"full_training": [False]},
            )
        )

    def end_round(self, done_event: Enum = Event.DONE) -> None:
        """Ends round early to cover `wait_for_end` generator."""
        super().end_round(done_event)


class TestModelStrategyBehaviour(APYEstimationFSMBehaviourBaseCase):
    """Test `ModelStrategyBehaviour`"""

    behaviour_class = ModelStrategyBehaviour

    fresh_msg = "Creating a fresh forecasting model."
    cycle_msg = "Estimation will happen again using the same model, after updating it with the latest data."

    @pytest.mark.parametrize(
        "period_count, n_estimations_before_retrain, log_message, event, next_behaviour_id",
        (
            (
                1,
                1,
                fresh_msg,
                Event.DONE,
                FetchBehaviour.auto_behaviour_id(),
            ),
            (
                0,
                2,
                fresh_msg,
                Event.DONE,
                FetchBehaviour.auto_behaviour_id(),
            ),
            (
                1,
                2,
                cycle_msg,
                Event.NEGATIVE,
                FetchBatchBehaviour.auto_behaviour_id(),
            ),
        ),
    )
    def test_strategy_behaviour(
        self,
        caplog: LogCaptureFixture,
        period_count: int,
        n_estimations_before_retrain: int,
        log_message: str,
        event: Event,
        next_behaviour_id: str,
    ) -> None:
        """Test the behaviour."""
        db = AbciAppDB({})

        db._data = {period_count: {}}

        self.fast_forward_to_behaviour(
            behaviour=self.behaviour,
            behaviour_id=self.behaviour_class.auto_behaviour_id(),
            synchronized_data=SynchronizedData(db),
        )
        behaviour = cast(ModelStrategyBehaviour, self.behaviour.current_behaviour)
        assert behaviour.behaviour_id == self.behaviour_class.auto_behaviour_id()

        behaviour.params.__dict__[
            "n_estimations_before_retrain"
        ] = n_estimations_before_retrain

        with caplog.at_level(
            logging.INFO,
            logger="aea.test_agent_name.packages.valory.skills.apy_estimation_abci",
        ):
            behaviour.act_wrapper()
        assert log_message in caplog.text

        self.mock_a2a_transaction()
        self._test_done_flag_set()
        self.end_round(event)

        behaviour = cast(BaseBehaviour, self.behaviour.current_behaviour)
        assert behaviour.behaviour_id == next_behaviour_id


class TestFetchProgress:
    """Test `FetchBehaviour`'s `Progress`."""

    @pytest.mark.parametrize(
        "current_timestamp, current_dex_name, expected",
        (
            (None, None, False),
            (None, "test", False),
            (0, None, False),
            (0, "test", True),
        ),
    )
    def test_can_continue(
        self,
        current_timestamp: Optional[int],
        current_dex_name: Optional[str],
        expected: bool,
    ) -> None:
        """Test `can_continue` property."""
        progress = FetchBehaviour.Progress(
            current_timestamp=current_timestamp, current_dex_name=current_dex_name
        )
        assert progress.can_continue is expected


class TestFetchAndBatchBehaviours(APYEstimationFSMBehaviourBaseCase):
    """Test FetchBehaviour and FetchBatchBehaviour."""

    behaviour_class = FetchBehaviour
    next_behaviour_class = TransformBehaviour

    @pytest.mark.parametrize(
        "progress_initialized, current_dex_name, expected_pair_ids",
        (
            (False, "spooky_subgraph", []),
            (True, "spooky_subgraph", ["0xec454eda10accdd66209c57af8c12924556f3abd"]),
            (True, "uniswap_subgraph", ["0x00004ee988665cdda9a1080d5792cecd16dc1220"]),
        ),
    )
    def test_current_pair_ids(
        self,
        progress_initialized: bool,
        current_dex_name: str,
        expected_pair_ids: List[str],
        pairs_ids: Dict[str, List[str]],
    ) -> None:
        """Test `current_pair_ids`."""
        self.fast_forward_to_behaviour(
            self.behaviour,
            self.behaviour_class.auto_behaviour_id(),
            self.synchronized_data,
        )
        behaviour = cast(FetchBehaviour, self.behaviour.current_behaviour)
        behaviour._progress.initialized = progress_initialized
        behaviour._progress.current_dex_name = current_dex_name
        behaviour.params.__dict__["pair_ids"] = pairs_ids

        assert expected_pair_ids == behaviour.current_pair_ids
        self.end_round()

    @pytest.mark.parametrize(
        "progress_initialized, current_dex_name, expected",
        (
            (False, "test", "default"),
            (False, "other", "default"),
            (True, "test", "test"),
            (True, "other", "other"),
        ),
    )
    @mock.patch.object(
        SubgraphsMixin, "get_subgraph", return_value="get_subgraph_output"
    )
    def test_current_dex(
        self,
        get_subgraph_mock: mock.Mock,
        progress_initialized: bool,
        current_dex_name: str,
        expected: str,
    ) -> None:
        """Test `current_dex`."""
        self.fast_forward_to_behaviour(
            self.behaviour,
            self.behaviour_class.auto_behaviour_id(),
            self.synchronized_data,
        )
        behaviour = cast(FetchBehaviour, self.behaviour.current_behaviour)
        behaviour._progress.initialized = progress_initialized
        behaviour._progress.current_dex_name = current_dex_name

        actual = behaviour.current_dex
        get_subgraph_mock.assert_called_once_with(expected)
        assert actual == "get_subgraph_output"
        self.end_round()

    def test_current_chain(self) -> None:
        """Test `current_chain`."""
        self.fast_forward_to_behaviour(
            self.behaviour,
            self.behaviour_class.auto_behaviour_id(),
            self.synchronized_data,
        )
        behaviour = cast(FetchBehaviour, self.behaviour.current_behaviour)
        behaviour.context.default = mock.MagicMock(
            chain_subgraph_name="chain_subgraph_name"
        )
        behaviour.context.chain_subgraph_name = mock.MagicMock(
            api_id="test_current_chain"
        )

        assert (
            behaviour.current_chain.api_id == "test_current_chain"
        ), "current chain is incorrect"
        self.end_round()

    @pytest.mark.parametrize(
        "current_dex_name", ("uniswap_subgraph", "spooky_subgraph")
    )
    @pytest.mark.parametrize(
        "progress_initialized, pairs_hist, n_fetched, expected",
        (
            (False, [], 0, 0),
            (False, ["test"], 100, 0),
            (True, ["test"], 1, 0),
            (True, [i for i in range(100)], 10, 90),
            (True, [i for i in range(123)], 67, 56),
        ),
    )
    def test_currently_downloaded(
        self,
        current_dex_name: str,
        progress_initialized: bool,
        pairs_hist: ResponseItemType,
        n_fetched: int,
        expected: int,
        pairs_ids: Dict[str, List[str]],
    ) -> None:
        """Test `currently_downloaded`."""
        self.fast_forward_to_behaviour(
            self.behaviour,
            self.behaviour_class.auto_behaviour_id(),
            self.synchronized_data,
        )
        behaviour = cast(FetchBehaviour, self.behaviour.current_behaviour)
        behaviour.params.__dict__["pair_ids"] = pairs_ids
        behaviour._pairs_hist = pairs_hist
        behaviour._progress.initialized = progress_initialized
        behaviour._progress.current_dex_name = current_dex_name
        behaviour._progress.n_fetched = n_fetched

        assert behaviour.currently_downloaded == expected
        self.end_round()

    @given(st.lists(st.integers()))
    @settings(deadline=None, database=database.InMemoryExampleDatabase())
    def test_total_downloaded(
        self,
        pairs_hist: List,
    ) -> None:
        """Test `total_downloaded`."""
        self.fast_forward_to_behaviour(
            self.behaviour,
            self.behaviour_class.auto_behaviour_id(),
            self.synchronized_data,
        )
        behaviour = cast(FetchBehaviour, self.behaviour.current_behaviour)
        behaviour._pairs_hist = pairs_hist
        #
        assert behaviour.total_downloaded == len(pairs_hist)
        self.end_round()

    @given(st.lists(st.booleans(), max_size=50))
    @settings(deadline=None, database=database.InMemoryExampleDatabase())
    def test_retries_exceeded(
        self,
        is_exceeded_per_subgraph: List[bool],
    ) -> None:
        """Test `retries_exceeded`."""
        self.fast_forward_to_behaviour(
            self.behaviour,
            self.behaviour_class.auto_behaviour_id(),
            self.synchronized_data,
        )
        behaviour = cast(FetchBehaviour, self.behaviour.current_behaviour)
        # set up dummy subgraphs with their `is_retries_exceeded` returning whatever hypothesis selected to
        for i in range(len(is_exceeded_per_subgraph)):
            subgraph = mock.MagicMock()
            subgraph.is_retries_exceeded.return_value = is_exceeded_per_subgraph[i]
            behaviour._utilized_subgraphs[f"test{i}"] = subgraph
        #
        # we expect `retries_exceeded` to return True if any of the subgraphs' `is_retries_exceeded` returns `True`
        expected = any(is_exceeded_per_subgraph)
        assert behaviour.retries_exceeded == expected
        behaviour._utilized_subgraphs = {}
        self.end_round()

    @pytest.mark.parametrize(
        "interval_not_acceptable, expected", ((True, DAY_IN_UNIX), (False, 0))
    )
    def test_shift(
        self,
        interval_not_acceptable: bool,
        expected: int,
    ) -> None:
        """Test `retries_exceeded`."""
        self.fast_forward_to_behaviour(
            self.behaviour,
            self.behaviour_class.auto_behaviour_id(),
            self.synchronized_data,
        )
        behaviour = cast(FetchBehaviour, self.behaviour.current_behaviour)
        behaviour.params.__dict__["interval_not_acceptable"] = interval_not_acceptable
        assert behaviour.shift == expected
        self.end_round()

    @pytest.mark.parametrize("batch_flag", (True, False))
    def test_setup(self, monkeypatch: MonkeyPatch, batch_flag: bool) -> None:
        """Test behaviour setup."""
        self.skill.skill_context.state.round_sequence.abci_app._last_timestamp = (
            datetime.now()
        )

        self.fast_forward_to_behaviour(
            self.behaviour,
            self.behaviour_class.auto_behaviour_id(),
            self.synchronized_data,
        )
        cast(FetchBehaviour, self.behaviour.current_behaviour).batch = batch_flag

        monkeypatch.setattr(os.path, "join", lambda *_: "")
        cast(APYEstimationBaseBehaviour, self.behaviour.current_behaviour).setup()

    @staticmethod
    def mocked_handle_response_wrapper(
        res: Optional[Any],
    ) -> Callable[[Any, Any], Generator[None, None, Optional[Any]]]:
        """A wrapper to a mocked version of the `_handle_response` method, which returns the given `fetched_pairs`."""

        def mocked_handle_response(
            *_: Any, **__: Any
        ) -> Generator[None, None, Optional[Any]]:
            """A mocked version of the `_handle_response` method, which returns the given `fetched_pairs`."""
            yield
            return res

        return mocked_handle_response

    @pytest.mark.parametrize(
        "fetched_pairs, expected",
        (
            (None, False),
            ([{"id": "test"}, {"id": "test"}], False),
            (
                [{"id": "test"}, {"id": "0x00004ee988665cdda9a1080d5792cecd16dc1220"}],
                False,
            ),
            (
                [{"id": "0xec454eda10accdd66209c57af8c12924556f3abd"}, {"id": "test"}],
                False,
            ),
            (
                [
                    {"id": "0xec454eda10accdd66209c57af8c12924556f3abd"},
                    {"id": "0x00004ee988665cdda9a1080d5792cecd16dc1220"},
                ],
                True,
            ),
        ),
    )
    def test_check_given_pairs(
        self,
        fetched_pairs: Optional[List[Dict[str, str]]],
        expected: bool,
        pairs_ids: Dict[str, List[str]],
    ) -> None:
        """Test `_check_given_pairs` method."""
        self.fast_forward_to_behaviour(
            self.behaviour,
            self.behaviour_class.auto_behaviour_id(),
            self.synchronized_data,
        )
        behaviour = cast(FetchBehaviour, self.behaviour.current_behaviour)
        behaviour.params.__dict__["pair_ids"] = pairs_ids
        # we do this because of https://github.com/valory-xyz/open-autonomy/pull/646
        behaviour.get_subgraph = mock.MagicMock()  # type: ignore
        # we do this because of https://github.com/valory-xyz/open-autonomy/pull/646
        behaviour.get_http_response = mock.MagicMock()  # type: ignore
        # we do this because of https://github.com/valory-xyz/open-autonomy/pull/646
        behaviour._handle_response = TestFetchAndBatchBehaviours.mocked_handle_response_wrapper(  # type: ignore
            fetched_pairs
        )  # type: ignore

        gen = behaviour._check_given_pairs()
        gen.send(None)
        gen.send(None)

        try:
            gen.send(None)
        except StopIteration:
            assert behaviour._pairs_exist is expected
        else:
            raise AssertionError(
                "Test did not finish as expected. `_check_given_pairs` should have reached to its end."
            )
        self.end_round()

    @pytest.mark.parametrize("batch", (True, False))
    @pytest.mark.parametrize("interval_not_acceptable", (True, False))
    @given(
        # set max data points to 1000
        st.integers(min_value=0, max_value=1000),
        # set max interval to a day
        st.integers(min_value=1, max_value=60 * 60 * 24),
        st.integers(min_value=1),
    )
    @settings(
        deadline=None,
        database=database.InMemoryExampleDatabase(),
        suppress_health_check=[HealthCheck.filter_too_much],
    )
    def test_reset_timestamps_iterator(
        self,
        batch: bool,
        interval_not_acceptable: bool,
        n_observations: int,
        interval: int,
        end: int,
    ) -> None:
        """Test `_reset_timestamps_iterator` method."""
        # filter out end values that will result in negative `start`
        assume(end >= n_observations * interval)

        self.fast_forward_to_behaviour(
            self.behaviour,
            self.behaviour_class.auto_behaviour_id(),
            self.synchronized_data,
        )
        behaviour = cast(FetchBehaviour, self.behaviour.current_behaviour)
        behaviour.batch = batch
        behaviour.params.__dict__["n_observations"] = n_observations
        behaviour.params.__dict__["interval"] = interval
        behaviour.params.__dict__["interval_not_acceptable"] = interval_not_acceptable
        behaviour._end_timestamp = end

        behaviour._reset_timestamps_iterator()

        start = end - n_observations * interval
        if batch and interval_not_acceptable:
            expected = [(end - DAY_IN_UNIX, True), (end, False)]
        elif batch:
            expected = [(end, False)]
        elif interval_not_acceptable:
            timestamps = tuple(range(start, end, interval))
            expected = [
                (timestamp, flag)
                for value in timestamps
                for timestamp, flag in ((value - behaviour.shift, True), (value, False))
            ]
        else:
            timestamps_iterator = range(start, end, interval)
            expected = list(itertools.product(timestamps_iterator, (False,)))
        assert behaviour._progress.timestamps_iterator is not None
        assert list(behaviour._progress.timestamps_iterator) == expected
        self.end_round()

    @given(
        st.booleans(),
        st.booleans(),
        st.booleans(),
        st.booleans(),
        st.integers(max_value=50),
        st.integers(),
        st.booleans(),
        st.text(min_size=1),
        st.tuples(st.integers(), st.booleans()),
        st.lists(st.text(min_size=1)),
    )
    @settings(deadline=None, database=database.InMemoryExampleDatabase())
    def test_set_current_progress(
        self,
        pairs_ids: Dict[str, List[str]],
        apy_shift: bool,
        interval_not_acceptable: bool,
        retries_exceeded: bool,
        call_failed: bool,
        currently_downloaded: int,
        target_per_pool: int,
        batch: bool,
        expected_dex_name: str,
        expected_timestamp: Tuple[int, bool],
        pairs_hist: ResponseItemType,
    ) -> None:
        """Test `_set_current_progress`."""
        self.fast_forward_to_behaviour(
            self.behaviour,
            self.behaviour_class.auto_behaviour_id(),
            self.synchronized_data,
        )
        behaviour = cast(FetchBehaviour, self.behaviour.current_behaviour)
        behaviour._progress.apy_shift = apy_shift
        behaviour.params.__dict__["interval_not_acceptable"] = interval_not_acceptable
        behaviour._utilized_subgraphs["test"] = ApiSpecs(
            name="",
            skill_context=mock.MagicMock(),
            url="test",
            api_id="test",
            method="test",
            headers=OrderedDict(),
            parameters=OrderedDict(),
        )
        assert behaviour._utilized_subgraphs["test"] is not None
        behaviour.batch = batch

        # start of setting the `currently_downloaded`
        # we cannot simply mock because of https://github.com/valory-xyz/open-autonomy/pull/646
        behaviour.params.__dict__["pair_ids"] = pairs_ids
        behaviour._progress.current_dex_name = "uniswap_subgraph"
        behaviour._pairs_hist = [{"test": "test"} for _ in range(currently_downloaded)]
        # end of setting the `currently_downloaded`

        behaviour._target_per_pool = target_per_pool
        behaviour._pairs_hist = pairs_hist
        behaviour._progress.call_failed = call_failed
        behaviour._progress.dex_names_iterator = (
            iter((expected_dex_name,)) if expected_dex_name is not None else iter(())
        )
        behaviour._progress.timestamps_iterator = (
            iter((expected_timestamp,)) if expected_timestamp is not None else iter(())
        )
        # we do this because of https://github.com/valory-xyz/open-autonomy/pull/646
        behaviour._reset_timestamps_iterator = mock.MagicMock()  # type: ignore
        # we do this because of https://github.com/valory-xyz/open-autonomy/pull/646
        behaviour.clean_up = mock.MagicMock()  # type: ignore

        if retries_exceeded:
            # exceed the retries before calling the method
            for _ in range(
                behaviour._utilized_subgraphs["test"].retries_info.retries + 1
            ):
                behaviour._utilized_subgraphs["test"].increment_retries()
            # call the tested method
            behaviour._set_current_progress()
            # assert its results
            assert behaviour._progress.current_timestamp is None
            assert behaviour._progress.current_dex_name is None
            behaviour.clean_up.assert_called_once()

        elif not call_failed:
            # call the tested method
            behaviour._set_current_progress()

            if (
                currently_downloaded == 0
                or currently_downloaded == target_per_pool
                or batch
            ) and not apy_shift:
                # assert its results
                assert behaviour._progress.current_dex_name == expected_dex_name
                behaviour._reset_timestamps_iterator.assert_called_once()
                assert (
                    behaviour._progress.n_fetched == ceil(len(pairs_hist) / 2)
                    if interval_not_acceptable
                    else 1
                )

            assert behaviour._progress.current_timestamp == expected_timestamp[0]
            assert behaviour._progress.apy_shift == expected_timestamp[1]

        else:
            # call the tested method
            behaviour._set_current_progress()

        assert behaviour._progress.initialized
        self.end_round()

    def test_handle_response(self, caplog: LogCaptureFixture) -> None:
        """Test `handle_response`."""
        self.fast_forward_to_behaviour(
            self.behaviour,
            self.behaviour_class.auto_behaviour_id(),
            self.synchronized_data,
        )

        # test with empty response.
        specs = ApiSpecs(
            url="test",
            api_id="test",
            method="GET",
            headers=OrderedDict(),
            parameters=OrderedDict(),
            name="test",
            skill_context=self.behaviour.context,
            backoff_factor=SLEEP_TIME_TWEAK,
        )

        with caplog.at_level(
            logging.ERROR,
            logger="aea.test_agent_name.packages.valory.skills.apy_estimation_abci",
        ):
            handling_generator = cast(
                FetchBehaviour, self.behaviour.current_behaviour
            )._handle_response(None, "test_context", ("", 0), specs)
            next(handling_generator)
            time.sleep(specs.retries_info.suggested_sleep_time + 0.01)

            try:
                next(handling_generator)
            except StopIteration as res:
                assert res.value is None

            assert (
                "[test_agent_name] Could not get test_context from test" in caplog.text
            )
            assert specs.retries_info.retries_attempted == 1

        caplog.clear()
        with caplog.at_level(
            logging.INFO,
            logger="aea.test_agent_name.packages.valory.skills.apy_estimation_abci",
        ):
            handling_generator = cast(
                FetchBehaviour, self.behaviour.current_behaviour
            )._handle_response({"test": [4, 5]}, "test", ("test", 0), specs)
            try:
                next(handling_generator)
            except StopIteration as res:
                assert res.value == 4
            assert "[test_agent_name] Retrieved test: 4." in caplog.text
            assert specs.retries_info.retries_attempted == 0

        self.end_round()

    @mock.patch.object(
        BaseBehaviour,
        "send_to_ipfs",
        side_effect=wrap_dummy_ipfs_operation("hash"),
    )
    @pytest.mark.parametrize(
        "interval, interval_not_acceptable", ((UNITS_TO_UNIX["hour"], True),)
    )
    def test_fetch_behaviour(
        self,
        _send_to_ipfs: mock._patch,
        block_from_timestamp_q: str,
        timestamp_gte: str,
        timestamp_lte: str,
        uni_eth_price_usd_q: str,
        spooky_eth_price_usd_q: str,
        spooky_pairs_q: str,
        uni_pairs_q: str,
        pairs_ids: Dict[str, List[str]],
        pool_fields: Tuple[str, ...],
        interval_not_acceptable: bool,
        interval: int,
    ) -> None:
        """Run tests."""
        self.fast_forward_to_behaviour(
            self.behaviour, FetchBehaviour.auto_behaviour_id(), self.synchronized_data
        )
        behaviour = cast(FetchBehaviour, self.behaviour.current_behaviour)
        behaviour.params.__dict__["interval"] = interval
        behaviour.params.__dict__["pair_ids"] = pairs_ids
        behaviour.params.__dict__["interval_not_acceptable"] = interval_not_acceptable
        # we do this because of https://github.com/valory-xyz/open-autonomy/pull/646
        behaviour._check_given_pairs = mock.MagicMock()  # type: ignore
        behaviour._pairs_exist = True
        n_iterations = N_OBSERVATIONS
        n_iterations *= 2 if interval_not_acceptable else 1

        # for every subgraph and every iteration that will be performed, we test fetching a single batch
        for subgraph_name, block_number, query in (
            ("uniswap_subgraph", "15178691", uni_eth_price_usd_q),
            ("spooky_subgraph", "52230630", spooky_eth_price_usd_q),
        ):
            for _ in range(n_iterations):
                behaviour.act_wrapper()
                subgraph = getattr(behaviour.context, subgraph_name)
                request_kwargs: Dict[str, Union[str, bytes]] = dict(
                    method="POST",
                    url=subgraph.url,
                    headers="Content-Type: application/json\r\n",
                    version="",
                )
                response_kwargs = dict(
                    version="",
                    status_code=200,
                    status_text="",
                    headers="",
                )

                # block request.
                assert behaviour._progress.current_timestamp is not None
                block_query = block_from_timestamp_q.replace(
                    timestamp_gte, str(behaviour._progress.current_timestamp)
                )
                block_query = block_query.replace(
                    timestamp_lte,
                    str(behaviour._progress.current_timestamp + SAFE_BLOCK_TIME),
                )
                block_subgraph = getattr(
                    behaviour.context, subgraph.chain_subgraph_name
                )
                request_kwargs["url"] = block_subgraph.url
                request_kwargs["body"] = json.dumps({"query": block_query}).encode(
                    "utf-8"
                )
                res = {"data": {"blocks": [{"timestamp": "1", "number": block_number}]}}
                response_kwargs["body"] = json.dumps(res).encode("utf-8")
                self.mock_http_request(request_kwargs, response_kwargs)

                # ETH price request.
                request_kwargs["url"] = subgraph.url
                request_kwargs["body"] = json.dumps({"query": query}).encode("utf-8")
                res = {"data": {"bundles": [{"ethPrice": "0.8973548"}]}}
                response_kwargs["body"] = json.dumps(res).encode("utf-8")
                behaviour.act_wrapper()
                self.mock_http_request(request_kwargs, response_kwargs)

                # top pairs data.
                pairs_q = (
                    uni_pairs_q
                    if subgraph_name == "uniswap_subgraph"
                    else spooky_pairs_q
                )
                request_kwargs["body"] = json.dumps({"query": pairs_q}).encode("utf-8")
                res = {
                    "data": {"pairs": [{field: "dummy_value" for field in pool_fields}]}
                }
                response_kwargs["body"] = json.dumps(res).encode("utf-8")
                behaviour.act_wrapper()
                self.mock_http_request(request_kwargs, response_kwargs)

        assert all(
            int(str(t2["forTimestamp"])) - int(str(t1["forTimestamp"])) == DAY_IN_UNIX
            for t1, t2 in zip(behaviour._pairs_hist[::2], behaviour._pairs_hist[1::2])
        )

        behaviour.act_wrapper()
        behaviour.act_wrapper()
        self.mock_a2a_transaction()
        self._test_done_flag_set()
        self.end_round()
        behaviour = cast(BaseBehaviour, self.behaviour.current_behaviour)
        assert behaviour.behaviour_id == TransformBehaviour.auto_behaviour_id()

    def test_invalid_pairs(self, caplog: LogCaptureFixture) -> None:
        """Test the behaviour when the given pairs do not exist at the subgraphs."""
        self.fast_forward_to_behaviour(
            self.behaviour, FetchBehaviour.auto_behaviour_id(), self.synchronized_data
        )
        behaviour = cast(FetchBehaviour, self.behaviour.current_behaviour)
        # we do this because of https://github.com/valory-xyz/open-autonomy/pull/646
        behaviour._check_given_pairs = mock.MagicMock()  # type: ignore
        behaviour._progress.call_failed = True
        # set the retries to the max allowed for any subgraph (chose SpookySwap randomly)
        behaviour.context.spooky_subgraph.retries_info.retries_attempted = (
            behaviour.context.spooky_subgraph.retries_info.retries
        )

        behaviour.act_wrapper()
        # exceed the retries
        behaviour.context.spooky_subgraph.increment_retries()
        behaviour._pairs_exist = False
        behaviour._progress.call_failed = True

        # test empty retrieved history.
        with caplog.at_level(
            logging.ERROR,
            logger="aea.test_agent_name.packages.valory.skills.apy_estimation_abci",
        ):
            behaviour.act_wrapper()
        assert "Could not download any historical data!" in caplog.text

        self.mock_a2a_transaction()
        self._test_done_flag_set()

        behaviour._progress.call_failed = False
        behaviour.context.spooky_subgraph.retries_info.retries_attempted = 0

        self.end_round()
        behaviour = cast(BaseBehaviour, self.behaviour.current_behaviour)
        assert behaviour.behaviour_id == TransformBehaviour.auto_behaviour_id()

    @pytest.mark.parametrize("is_non_indexed_res", (True, False))
    def test_fetch_behaviour_non_indexed_block(
        self,
        is_non_indexed_res: bool,
        block_from_timestamp_q: str,
        timestamp_gte: str,
        block_from_number_q: str,
        uni_eth_price_usd_q: str,
        uni_pairs_q: str,
        pairs_ids: Dict[str, List[str]],
        pool_fields: Tuple[str, ...],
        caplog: LogCaptureFixture,
    ) -> None:
        """Run tests for fetch behaviour when a block has not been indexed yet."""
        self.fast_forward_to_behaviour(
            self.behaviour, FetchBehaviour.auto_behaviour_id(), self.synchronized_data
        )
        behaviour = cast(FetchBehaviour, self.behaviour.current_behaviour)
        behaviour.params.__dict__["pair_ids"] = pairs_ids
        # make sure that the first generated timestamp (`behaviour.params.start` property)
        # will be the `timestamp_gte` which is used in `block_from_timestamp_q`
        behaviour._end_timestamp = (
            int(timestamp_gte)
            + behaviour.params.interval * behaviour.params.n_observations
        )
        # we do this because of https://github.com/valory-xyz/open-autonomy/pull/646
        behaviour._check_given_pairs = mock.MagicMock()  # type: ignore
        behaviour._pairs_exist = True
        behaviour._progress.initialized = True
        behaviour._progress.current_dex_name = "uniswap_subgraph"
        expected_timestamp = 1
        behaviour._progress.timestamps_iterator = iter(((expected_timestamp, False),))

        request_kwargs: Dict[str, Union[str, bytes]] = dict(
            method="POST",
            url=behaviour.context.uniswap_subgraph.url,
            headers="Content-Type: application/json\r\n",
            version="",
        )
        response_kwargs = dict(
            version="",
            status_code=200,
            status_text="",
            headers="",
        )

        # block request.
        request_kwargs["url"] = behaviour.context.ethereum_subgraph.url
        request_kwargs["body"] = json.dumps({"query": block_from_timestamp_q}).encode(
            "utf-8"
        )
        res: Dict[str, Union[List[Dict[str, str]], Dict[str, List[Dict[str, str]]]]] = {
            "data": {
                "blocks": [{"timestamp": str(expected_timestamp), "number": "15178691"}]
            }
        }
        response_kwargs["body"] = json.dumps(res).encode("utf-8")
        behaviour.act_wrapper()
        self.mock_http_request(request_kwargs, response_kwargs)

        # ETH price request for non-indexed block.
        request_kwargs["url"] = behaviour.context.uniswap_subgraph.url
        request_kwargs["body"] = json.dumps({"query": uni_eth_price_usd_q}).encode(
            "utf-8"
        )

        if is_non_indexed_res:
            res = {
                "errors": [
                    {
                        "message": "Failed to decode `block.number` value: `subgraph "
                        "QmPJbGjktGa7c4UYWXvDRajPxpuJBSZxeQK5siNT3VpthP has only indexed up to block number 3730367 "
                        "and data for block number 15178691 is therefore not yet available`"
                    }
                ]
            }
        else:
            res = {"errors": [{"message": "another error"}]}

        with caplog.at_level(
            logging.WARNING,
            logger="aea.test_agent_name.packages.valory.skills.apy_estimation_abci",
        ):
            response_kwargs["body"] = json.dumps(res).encode("utf-8")
            behaviour.act_wrapper()
            self.mock_http_request(request_kwargs, response_kwargs)

        if not is_non_indexed_res:
            assert (
                "Attempted to handle an indexing error, but could not extract the latest indexed block!"
                in caplog.text
            )
            return

        # indexed block request.
        request_kwargs["url"] = behaviour.context.ethereum_subgraph.url
        request_kwargs["body"] = json.dumps({"query": block_from_number_q}).encode(
            "utf-8"
        )
        res = {"data": {"blocks": [{"timestamp": "1", "number": "3730360"}]}}
        response_kwargs["body"] = json.dumps(res).encode("utf-8")
        behaviour.act_wrapper()
        self.mock_http_request(request_kwargs, response_kwargs)

        # ETH price request for indexed block.
        request_kwargs["url"] = behaviour.context.uniswap_subgraph.url
        request_kwargs["body"] = json.dumps(
            {"query": uni_eth_price_usd_q.replace("15178691", "3730360")}
        ).encode("utf-8")
        res = {"data": {"bundles": [{"ethPrice": "0.8973548"}]}}
        response_kwargs["body"] = json.dumps(res).encode("utf-8")
        behaviour.act_wrapper()
        self.mock_http_request(request_kwargs, response_kwargs)

        # top pairs data.
        request_kwargs["body"] = json.dumps(
            {"query": uni_pairs_q.replace("15178691", "3730360")}
        ).encode("utf-8")
        res = {
            "data": {
                "pairs": [
                    {field: dummy_value for field in pool_fields}
                    for dummy_value in ("dum1", "dum2")
                ]
            }
        }
        response_kwargs["body"] = json.dumps(res).encode("utf-8")
        behaviour.act_wrapper()
        self.mock_http_request(request_kwargs, response_kwargs)

        self.end_round()
        behaviour = cast(BaseBehaviour, self.behaviour.current_behaviour)
        assert behaviour.behaviour_id == TransformBehaviour.auto_behaviour_id()

    @pytest.mark.parametrize("none_at_step", (0, 1, 2))
    def test_fetch_behaviour_non_indexed_block_none_res(
        self,
        none_at_step: int,
        block_from_timestamp_q: str,
        timestamp_gte: str,
        block_from_number_q: str,
        uni_eth_price_usd_q: str,
        pairs_ids: Dict[str, List[str]],
    ) -> None:
        """Test `async_act` when we receive `None` responses in `_check_non_indexed_block`."""
        self.fast_forward_to_behaviour(
            self.behaviour, FetchBehaviour.auto_behaviour_id(), self.synchronized_data
        )
        behaviour = cast(FetchBehaviour, self.behaviour.current_behaviour)
        behaviour.params.__dict__["pair_ids"] = pairs_ids
        # make sure that the first generated timestamp (`behaviour.params.start` property)
        # will be the `timestamp_gte` which is used in `block_from_timestamp_q`
        behaviour._end_timestamp = (
            int(timestamp_gte)
            + behaviour.params.interval * behaviour.params.n_observations
        )
        # we do this because of https://github.com/valory-xyz/open-autonomy/pull/646
        behaviour._check_given_pairs = mock.MagicMock()  # type: ignore
        behaviour._pairs_exist = True
        behaviour._progress.initialized = True
        behaviour._progress.current_dex_name = "uniswap_subgraph"
        expected_timestamp = 1
        behaviour._progress.timestamps_iterator = (
            iter(((expected_timestamp, False),))
            if expected_timestamp is not None
            else iter(())
        )

        request_kwargs: Dict[str, Union[str, bytes]] = dict(
            method="POST",
            url=behaviour.context.uniswap_subgraph.url,
            headers="Content-Type: application/json\r\n",
            version="",
        )
        response_kwargs = dict(
            version="",
            status_code=200,
            status_text="",
            headers="",
        )

        # block request.
        request_kwargs["url"] = behaviour.context.ethereum_subgraph.url
        request_kwargs["body"] = json.dumps({"query": block_from_timestamp_q}).encode(
            "utf-8"
        )
        res: Dict[str, Union[List[Dict[str, str]], Dict[str, List[Dict[str, str]]]]] = {
            "data": {
                "blocks": [{"timestamp": str(expected_timestamp), "number": "15178691"}]
            }
        }
        response_kwargs["body"] = json.dumps(res).encode("utf-8")
        behaviour.act_wrapper()
        self.mock_http_request(request_kwargs, response_kwargs)

        # ETH price request for non-indexed block.
        request_kwargs["url"] = behaviour.context.uniswap_subgraph.url
        request_kwargs["body"] = json.dumps({"query": uni_eth_price_usd_q}).encode(
            "utf-8"
        )

        res = (
            {
                "errors": [
                    {
                        "message": "Failed to decode `block.number` value: `subgraph "
                        "QmPJbGjktGa7c4UYWXvDRajPxpuJBSZxeQK5siNT3VpthP has only indexed up to block number 3730367 "
                        "and data for block number 15178691 is therefore not yet available`"
                    }
                ]
            }
            if none_at_step
            else {}
        )

        response_kwargs["body"] = json.dumps(res).encode("utf-8")
        behaviour.act_wrapper()
        self.mock_http_request(request_kwargs, response_kwargs)

        if none_at_step == 0:
            time.sleep(
                behaviour.context.ethereum_subgraph.retries_info.suggested_sleep_time
                + 0.01
            )
            behaviour.act_wrapper()
            return

        # indexed block request.
        request_kwargs["url"] = behaviour.context.ethereum_subgraph.url
        request_kwargs["body"] = json.dumps({"query": block_from_number_q}).encode(
            "utf-8"
        )
        res = (
            {"data": {"blocks": [{"timestamp": "1", "number": "3730360"}]}}
            if none_at_step != 1
            else {}
        )
        response_kwargs["body"] = json.dumps(res).encode("utf-8")
        behaviour.act_wrapper()
        self.mock_http_request(request_kwargs, response_kwargs)

        if none_at_step == 1:
            time.sleep(
                behaviour.context.ethereum_subgraph.retries_info.suggested_sleep_time
                + 0.01
            )
            behaviour.act_wrapper()
            return

        # ETH price request for indexed block.
        request_kwargs["url"] = behaviour.context.uniswap_subgraph.url
        request_kwargs["body"] = json.dumps(
            {"query": uni_eth_price_usd_q.replace("15178691", "3730360")}
        ).encode("utf-8")
        res = {}
        response_kwargs["body"] = json.dumps(res).encode("utf-8")
        behaviour.act_wrapper()
        self.mock_http_request(request_kwargs, response_kwargs)
        time.sleep(
            behaviour.context.ethereum_subgraph.retries_info.suggested_sleep_time + 0.01
        )
        behaviour.act_wrapper()
        self.end_round()

    def test_fetch_behaviour_retries_exceeded(self, caplog: LogCaptureFixture) -> None:
        """Run tests for exceeded retries."""
        self.skill.skill_context.state.round_sequence.abci_app._last_timestamp = (
            datetime.now()
        )

        self.fast_forward_to_behaviour(
            self.behaviour, FetchBehaviour.auto_behaviour_id(), self.synchronized_data
        )
        behaviour = cast(FetchBehaviour, self.behaviour.current_behaviour)
        assert behaviour is not None
        assert behaviour.behaviour_id == FetchBehaviour.auto_behaviour_id()
        # we do this because of https://github.com/valory-xyz/open-autonomy/pull/646
        behaviour._check_given_pairs = mock.MagicMock()  # type: ignore
        behaviour._pairs_exist = True

        retries = None
        for subgraph_name in (
            "spooky_subgraph",
            "fantom_subgraph",
            "uniswap_subgraph",
            "ethereum_subgraph",
        ):
            subgraph = getattr(behaviour.context, subgraph_name)
            subgraph_retries = subgraph.retries_info.retries

            if retries is None:
                retries = subgraph_retries
            else:
                assert retries == subgraph_retries

        assert retries is not None
        for i in range(retries + 1):
            for subgraph_name in (
                "spooky_subgraph",
                "fantom_subgraph",
                "uniswap_subgraph",
                "ethereum_subgraph",
            ):
                subgraph = getattr(behaviour.context, subgraph_name)
                subgraph.increment_retries()

                if i == retries:
                    assert subgraph.is_retries_exceeded()

        with caplog.at_level(
            logging.ERROR,
            logger="aea.test_agent_name.packages.valory.skills.apy_estimation_abci",
        ):
            behaviour.act_wrapper()

        assert (
            "Retries were exceeded while downloading the historical data!"
            in caplog.text
        )
        for subgraph_name in (
            "spooky_subgraph",
            "fantom_subgraph",
            "uniswap_subgraph",
            "ethereum_subgraph",
        ):
            subgraph = getattr(behaviour.context, subgraph_name)
            assert not subgraph.is_retries_exceeded()

        assert behaviour._hist_hash == ""

        self.mock_a2a_transaction()
        self._test_done_flag_set()
        self.end_round()

    def test_fetch_value_none(
        self,
        caplog: LogCaptureFixture,
        block_from_timestamp_q: str,
        timestamp_gte: str,
        uni_eth_price_usd_q: str,
        uni_pairs_q: str,
        pairs_ids: Dict[str, List[str]],
        pool_fields: Tuple[str, ...],
    ) -> None:
        """Test when fetched value is none."""
        self.fast_forward_to_behaviour(
            self.behaviour, FetchBehaviour.auto_behaviour_id(), self.synchronized_data
        )
        behaviour = cast(FetchBehaviour, self.behaviour.current_behaviour)
        behaviour.params.__dict__["pair_ids"] = pairs_ids
        # make sure that the first generated timestamp (`behaviour.params.start` property)
        # will be the `timestamp_gte` which is used in `block_from_timestamp_q`
        behaviour._end_timestamp = (
            int(timestamp_gte)
            + behaviour.params.interval * behaviour.params.n_observations
        )
        # we do this because of https://github.com/valory-xyz/open-autonomy/pull/646
        behaviour._check_given_pairs = mock.MagicMock()  # type: ignore
        behaviour._pairs_exist = True
        behaviour._progress.initialized = True
        behaviour._progress.current_dex_name = "uniswap_subgraph"
        expected_timestamp = 1
        behaviour._progress.timestamps_iterator = (
            iter(((expected_timestamp, False),))
            if expected_timestamp is not None
            else iter(())
        )

        request_kwargs: Dict[str, Union[str, bytes]] = dict(
            method="POST",
            url=behaviour.context.spooky_subgraph.url,
            headers="Content-Type: application/json\r\n",
            version="",
            body=b"",
        )
        response_kwargs = dict(
            version="",
            status_code=200,
            status_text="",
            headers="",
            body=b"",
        )

        # block request with None response.
        request_kwargs["url"] = behaviour.context.ethereum_subgraph.url
        request_kwargs["body"] = json.dumps({"query": block_from_timestamp_q}).encode(
            "utf-8"
        )
        response_kwargs["body"] = b""

        with caplog.at_level(
            logging.ERROR,
            logger="aea.test_agent_name.packages.valory.skills.apy_estimation_abci",
        ):
            behaviour.act_wrapper()
            self.mock_http_request(request_kwargs, response_kwargs)
        assert "[test_agent_name] Could not get block from eth" in caplog.text

        caplog.clear()
        time.sleep(
            behaviour.context.ethereum_subgraph.retries_info.suggested_sleep_time + 0.01
        )
        behaviour.act_wrapper()

        # block request.
        request_kwargs["body"] = json.dumps({"query": block_from_timestamp_q}).encode(
            "utf-8"
        )
        res = {
            "data": {
                "blocks": [{"timestamp": str(expected_timestamp), "number": "15178691"}]
            }
        }
        response_kwargs["body"] = json.dumps(res).encode("utf-8")
        behaviour.act_wrapper()
        self.mock_http_request(request_kwargs, response_kwargs)

        # ETH price request with None response.
        request_kwargs["url"] = behaviour.context.uniswap_subgraph.url
        request_kwargs["body"] = json.dumps({"query": uni_eth_price_usd_q}).encode(
            "utf-8"
        )
        response_kwargs["body"] = b""

        with caplog.at_level(
            logging.ERROR,
            logger="aea.test_agent_name.packages.valory.skills.apy_estimation_abci",
        ):
            behaviour.act_wrapper()
            self.mock_http_request(request_kwargs, response_kwargs)
        assert (
            "[test_agent_name] Could not get ETH price for block "
            "{'timestamp': '1', 'number': '15178691'} from uniswap" in caplog.text
        )

        caplog.clear()
        time.sleep(
            behaviour.context.uniswap_subgraph.retries_info.suggested_sleep_time + 0.01
        )
        behaviour.act_wrapper()

        # block request.
        request_kwargs["url"] = behaviour.context.ethereum_subgraph.url
        request_kwargs["body"] = json.dumps({"query": block_from_timestamp_q}).encode(
            "utf-8"
        )
        res = {"data": {"blocks": [{"timestamp": "1", "number": "15178691"}]}}
        response_kwargs["body"] = json.dumps(res).encode("utf-8")
        behaviour.act_wrapper()
        self.mock_http_request(request_kwargs, response_kwargs)

        # ETH price request.
        request_kwargs["url"] = behaviour.context.uniswap_subgraph.url
        request_kwargs["body"] = json.dumps({"query": uni_eth_price_usd_q}).encode(
            "utf-8"
        )
        res = {"data": {"bundles": [{"ethPrice": "0.8973548"}]}}
        response_kwargs["body"] = json.dumps(res).encode("utf-8")
        behaviour.act_wrapper()
        self.mock_http_request(request_kwargs, response_kwargs)

        # top pairs data with None response.
        request_kwargs["body"] = json.dumps({"query": uni_pairs_q}).encode("utf-8")
        response_kwargs["body"] = b""

        with caplog.at_level(
            logging.ERROR,
            logger="aea.test_agent_name.packages.valory.skills.apy_estimation_abci",
        ):
            behaviour.act_wrapper()
            self.mock_http_request(request_kwargs, response_kwargs)
        assert (
            "[test_agent_name] Could not get pool data for block {'timestamp': '1', 'number': '15178691'} "
            "from uniswap" in caplog.text
        )

        caplog.clear()
        time.sleep(
            behaviour.context.ethereum_subgraph.retries_info.suggested_sleep_time + 0.01
        )
        behaviour.act_wrapper()
        self.end_round()

    @mock.patch.object(
        BaseBehaviour,
        "send_to_ipfs",
        side_effect=wrap_dummy_ipfs_operation("hash"),
    )
    @pytest.mark.parametrize("target_per_pool", (0, 3))
    def test_fetch_behaviour_stop_iteration(
        self,
        _send_to_ipfs: mock._patch,
        caplog: LogCaptureFixture,
        no_action: Callable[[Any], None],
        target_per_pool: int,
    ) -> None:
        """Test `FetchBehaviour`'s `async_act` after all the timestamps have been generated."""
        self.skill.skill_context.state.round_sequence.abci_app._last_timestamp = (
            datetime.now()
        )

        # fast-forward to fetch behaviour.
        self.fast_forward_to_behaviour(
            self.behaviour, FetchBehaviour.auto_behaviour_id(), self.synchronized_data
        )
        behaviour = cast(FetchBehaviour, self.behaviour.current_behaviour)
        # set `n_observations` to `0` in order to raise a `StopIteration`.
        behaviour.params.__dict__["n_observations"] = 0
        # we do this because of https://github.com/valory-xyz/open-autonomy/pull/646
        behaviour._check_given_pairs = mock.MagicMock()  # type: ignore
        behaviour._pairs_exist = True

        # test empty retrieved history.
        with caplog.at_level(
            logging.ERROR,
            logger="aea.test_agent_name.packages.valory.skills.apy_estimation_abci",
        ):
            behaviour.act_wrapper()
        assert "Could not download any historical data!" in caplog.text
        self.mock_a2a_transaction()
        self._test_done_flag_set()
        self.end_round()

        # fast-forward to fetch behaviour.
        self.fast_forward_to_behaviour(
            self.behaviour, FetchBehaviour.auto_behaviour_id(), self.synchronized_data
        )

        # test with partly fetched history and valid save path.
        behaviour = cast(FetchBehaviour, self.behaviour.current_behaviour)
        # we do this because of https://github.com/valory-xyz/open-autonomy/pull/646
        behaviour._check_given_pairs = mock.MagicMock()  # type: ignore
        behaviour._pairs_exist = True

        behaviour._pairs_hist = [{"pool1": "test"}, {"pool2": "test"}]
        behaviour._target_per_pool = target_per_pool
        behaviour.act_wrapper()
        behaviour.act_wrapper()
        self.mock_a2a_transaction()
        self._test_done_flag_set()
        self.end_round()
        behaviour = cast(BaseBehaviour, self.behaviour.current_behaviour)
        assert behaviour.behaviour_id == TransformBehaviour.auto_behaviour_id()

        # fast-forward to fetch behaviour.
        self.fast_forward_to_behaviour(
            self.behaviour, FetchBehaviour.auto_behaviour_id(), self.synchronized_data
        )

    def test_clean_up(
        self,
    ) -> None:
        """Test clean-up."""
        self.fast_forward_to_behaviour(
            self.behaviour, FetchBehaviour.auto_behaviour_id(), self.synchronized_data
        )
        assert self.behaviour.current_behaviour is not None
        behaviour = cast(FetchBehaviour, self.behaviour.current_behaviour)

        for subgraph_name in (
            "spooky_subgraph",
            "fantom_subgraph",
            "uniswap_subgraph",
            "ethereum_subgraph",
        ):
            subgraph = getattr(behaviour.context, subgraph_name)
            subgraph.retries_info.retries_attempted = 1

        for subgraph_name in (
            "spooky_subgraph",
            "fantom_subgraph",
            "uniswap_subgraph",
            "ethereum_subgraph",
        ):
            subgraph = getattr(behaviour.context, subgraph_name)
            self.behaviour.current_behaviour.clean_up()
            assert subgraph.retries_info.retries_attempted == 0

        self.end_round()


class TestTransformBehaviour(APYEstimationFSMBehaviourBaseCase):
    """Test TransformBehaviour."""

    behaviour_class = TransformBehaviour
    next_behaviour_class = PreprocessBehaviour
    dummy_ipfs_object = {"test": "test"}

    def _fast_forward(self) -> None:
        """Setup `TestTransformBehaviour`."""
        # Send historical data to IPFS and get the hash.
        self.fast_forward_to_behaviour(
            self.behaviour,
            self.behaviour_class.auto_behaviour_id(),
            SynchronizedData(
                AbciAppDB(
                    setup_data=dict(most_voted_randomness=[0], history_hash=["hash"]),
                )
            ),
        )

        assert (
            cast(
                APYEstimationBaseBehaviour, self.behaviour.current_behaviour
            ).behaviour_id
            == self.behaviour_class.auto_behaviour_id()
        )

    @mock.patch.object(
        BaseBehaviour,
        "get_from_ipfs",
        side_effect=wrap_dummy_ipfs_operation(dummy_ipfs_object),
    )
    def test_prepare_task(self, _get_from_ipfs: mock._patch) -> None:
        """Test behaviour setup."""
        self._fast_forward()
        self.behaviour.context.task_manager.start()
        prepare_gen = cast(
            TransformBehaviour, self.behaviour.current_behaviour
        ).prepare_task()
        next(prepare_gen)
        with pytest.raises(StopIteration):
            next(prepare_gen)
        self.end_round()

    @mock.patch.object(
        BaseBehaviour,
        "get_from_ipfs",
        side_effect=wrap_dummy_ipfs_operation(dummy_ipfs_object),
    )
    def test_task_not_ready(
        self,
        _get_from_ipfs: mock._patch,
        monkeypatch: MonkeyPatch,
        caplog: LogCaptureFixture,
    ) -> None:
        """Run test for `transform_behaviour` when task result is not ready."""
        self._fast_forward()
        self.behaviour.context.task_manager.start()
        monkeypatch.setattr(AsyncResult, "ready", lambda *_: False)
        self.behaviour.current_behaviour.params.__dict__["sleep_time"] = SLEEP_TIME_TWEAK  # type: ignore

        with caplog.at_level(
            logging.DEBUG,
            logger="aea.test_agent_name.packages.valory.skills.apy_estimation_abci",
        ):
            self.behaviour.act_wrapper()
            assert (
                f"[test_agent_name] Entered in the {TransformBehaviour.auto_behaviour_id()!r} behaviour"
                in caplog.text
            )
            self.behaviour.act_wrapper()

        assert (
            "[test_agent_name] The transform task is not finished yet." in caplog.text
        )

        time.sleep(SLEEP_TIME_TWEAK + 0.01)
        self.behaviour.act_wrapper()

        assert (
            cast(
                APYEstimationBaseBehaviour, self.behaviour.current_behaviour
            ).behaviour_id
            == self.behaviour_class.auto_behaviour_id()
        )
        self.end_round()

    @mock.patch.object(
        BaseBehaviour,
        "send_to_ipfs",
        side_effect=wrap_dummy_ipfs_operation("hash"),
    )
    @pytest.mark.parametrize("ipfs_succeed", (True, False))
    def test_transform_behaviour(
        self,
        _send_to_ipfs: mock._patch,
        monkeypatch: MonkeyPatch,
        transformed_historical_data_no_datetime_conversion: pd.DataFrame,
        ipfs_succeed: bool,
    ) -> None:
        """Run test for `transform_behaviour`."""
        self._fast_forward()

        monkeypatch.setattr(
            tasks,
            "transform_hist_data",
            lambda _: transformed_historical_data_no_datetime_conversion,
        )
        monkeypatch.setattr(
            self._skill._skill_context._agent_context._task_manager,  # type: ignore
            "get_task_result",
            lambda *_: DummyAsyncResult(
                transformed_historical_data_no_datetime_conversion
            ),
        )
        monkeypatch.setattr(
            self._skill._skill_context._agent_context._task_manager,  # type: ignore
            "enqueue_task",
            lambda *_, **__: 3,
        )

        item_to_send = self.dummy_ipfs_object if ipfs_succeed else None

        with mock.patch.object(
            BaseBehaviour,
            "get_from_ipfs",
            side_effect=wrap_dummy_ipfs_operation(item_to_send),
        ):
            for _ in range(3):
                self.behaviour.act_wrapper()

        self.behaviour.act_wrapper()
        self.mock_a2a_transaction()
        self._test_done_flag_set()
        self.end_round()

        behaviour = cast(BaseBehaviour, self.behaviour.current_behaviour)
        assert behaviour.behaviour_id == self.next_behaviour_class.auto_behaviour_id()


class TestPreprocessBehaviour(APYEstimationFSMBehaviourBaseCase):
    """Test PreprocessBehaviour."""

    behaviour_class = PreprocessBehaviour
    next_behaviour_class = RandomnessBehaviour

    def _fast_forward(
        self,
        transformed_historical_data_no_datetime_conversion: pd.DataFrame,
    ) -> pd.DataFrame:
        """Fast-forward to behaviour."""
        # Increase the amount of dummy data for the train-test split,
        # as many times as the threshold in `group_and_filter_pair_data`.
        transformed_historical_data = pd.DataFrame(
            np.repeat(
                transformed_historical_data_no_datetime_conversion.values, 5, axis=0
            ),
            columns=transformed_historical_data_no_datetime_conversion.columns,
        )

        self.fast_forward_to_behaviour(
            self.behaviour,
            self.behaviour_class.auto_behaviour_id(),
            SynchronizedData(
                AbciAppDB(
                    setup_data=dict(
                        transformed_history_hash=["test"],
                        latest_transformation_period=[0],
                    )
                )
            ),
        )
        behaviour = cast(PreprocessBehaviour, self.behaviour.current_behaviour)
        assert behaviour.behaviour_id == self.behaviour_class.auto_behaviour_id()

        behaviour.params.__dict__["sleep_time"] = SLEEP_TIME_TWEAK  # type: ignore

        return transformed_historical_data

    @mock.patch.object(
        BaseBehaviour,
        "send_to_ipfs",
        side_effect=wrap_dummy_ipfs_operation("hash"),
    )
    @pytest.mark.parametrize(
        "data_found, task_ready", ((True, True), (True, False), (False, False))
    )
    def test_preprocess_behaviour(
        self,
        _send_to_ipfs: mock._patch,
        data_found: bool,
        task_ready: bool,
        transformed_historical_data_no_datetime_conversion: pd.DataFrame,
        monkeypatch: MonkeyPatch,
    ) -> None:
        """Run test for `preprocess_behaviour`."""
        transformed_historical_data = self._fast_forward(
            transformed_historical_data_no_datetime_conversion,
        )

        # Convert the `blockTimestamp` to a pandas datetime.
        transformed_historical_data["blockTimestamp"] = pd.to_datetime(
            transformed_historical_data["blockTimestamp"], unit="s"
        )
        monkeypatch.setattr(
            self._skill._skill_context._agent_context._task_manager,  # type: ignore
            "get_task_result",
            lambda *_: DummyAsyncResult(
                prepare_pair_data(transformed_historical_data), task_ready
            ),
        )
        monkeypatch.setattr(
            self._skill._skill_context._agent_context._task_manager,  # type: ignore
            "enqueue_task",
            lambda *_, **__: 3,
        )

        if data_found:
            with mock.patch.object(
                BaseBehaviour,
                "get_from_ipfs",
                side_effect=wrap_dummy_ipfs_operation({"test": "test"}),
            ):
                self.behaviour.act_wrapper()

        for _ in range(3):
            self.behaviour.act_wrapper()

        if data_found:
            assert (
                cast(PreprocessBehaviour, self.behaviour.current_behaviour)._pairs_hist
                is not None
            ), "Pairs history could not be loaded!"

        if task_ready:
            self.mock_a2a_transaction()
            self._test_done_flag_set()

        else:
            self.behaviour.act_wrapper()
            time.sleep(SLEEP_TIME_TWEAK + 0.01)
            self.behaviour.act_wrapper()
            behaviour = cast(PreprocessBehaviour, self.behaviour.current_behaviour)
            assert behaviour.behaviour_id == self.behaviour_class.auto_behaviour_id()

        self.end_round()
        behaviour = cast(PreprocessBehaviour, self.behaviour.current_behaviour)
        assert behaviour.behaviour_id == self.next_behaviour_class.auto_behaviour_id()


class TestPrepareBatchBehaviour(APYEstimationFSMBehaviourBaseCase):
    """Test `PrepareBatchBehaviour`."""

    behaviour_class = PrepareBatchBehaviour
    next_behaviour_class = UpdateForecasterBehaviour

    def _fast_forward(
        self,
    ) -> None:
        """Setup `PrepareBatchBehaviour`."""
        # fast-forward to the `PrepareBatchBehaviour` behaviour.
        self.fast_forward_to_behaviour(
            self.behaviour,
            self.behaviour_class.auto_behaviour_id(),
            SynchronizedData(
                AbciAppDB(
                    setup_data=AbciAppDB.data_to_lists(
                        dict(
                            latest_observation_hist_hash="hist",
                            batch_hash="batch",
                        )
                    ),
                )
            ),
        )

        assert (
            cast(
                APYEstimationBaseBehaviour, self.behaviour.current_behaviour
            ).behaviour_id
            == self.behaviour_class.auto_behaviour_id()
        )

    def test_prepare_task(
        self,
        monkeypatch: MonkeyPatch,
        transformed_historical_data_no_datetime_conversion: pd.DataFrame,
        batch: ResponseItemType,
        prepare_batch_task_result: Dict[str, pd.DataFrame],
    ) -> None:
        """Test behaviour setup."""
        self._fast_forward()

        monkeypatch.setattr(TaskManager, "enqueue_task", lambda *_, **__: 0)
        monkeypatch.setattr(
            TaskManager,
            "get_task_result",
            lambda *_: DummyAsyncResult(prepare_batch_task_result),
        )

        current_behaviour = cast(
            PrepareBatchBehaviour, self.behaviour.current_behaviour
        )
        gen = current_behaviour.prepare_task()

        hist = transformed_historical_data_no_datetime_conversion.iloc[
            [0, 2]
        ].reset_index(drop=True)
        with mock.patch.object(
            BaseBehaviour,
            "get_from_ipfs",
            side_effect=wrap_dummy_ipfs_operation(hist),
        ):
            next(gen)

        with mock.patch.object(
            BaseBehaviour,
            "get_from_ipfs",
            side_effect=wrap_dummy_ipfs_operation(batch),
        ):
            next(gen)

        with pytest.raises(StopIteration):
            next(gen)

        assert current_behaviour._task_prepared
        assert not any(batch is None for batch in current_behaviour._batches)
        pd.testing.assert_frame_equal(current_behaviour._batches[0], hist)
        assert current_behaviour._batches[1] == batch
        self.end_round()

    def test_task_not_ready(
        self,
        monkeypatch: MonkeyPatch,
        transformed_historical_data_no_datetime_conversion: pd.DataFrame,
        batch: ResponseItemType,
        prepare_batch_task_result: Dict[str, pd.DataFrame],
    ) -> None:
        """Run test for behaviour when task result is not ready."""
        self._fast_forward()

        monkeypatch.setattr(TaskManager, "enqueue_task", lambda *_, **__: 0)
        monkeypatch.setattr(
            TaskManager,
            "get_task_result",
            lambda *_: DummyAsyncResult(prepare_batch_task_result, ready=False),
        )

        hist = transformed_historical_data_no_datetime_conversion.iloc[
            [0, 2]
        ].reset_index(drop=True)
        with mock.patch.object(
            BaseBehaviour,
            "get_from_ipfs",
            side_effect=wrap_dummy_ipfs_operation(hist),
        ):
            self.behaviour.act_wrapper()

        with mock.patch.object(
            BaseBehaviour,
            "get_from_ipfs",
            side_effect=wrap_dummy_ipfs_operation(batch),
        ):
            self.behaviour.act_wrapper()

        cast(PrepareBatchBehaviour, self.behaviour.current_behaviour).params.__dict__[
            "sleep_time"
        ] = SLEEP_TIME_TWEAK
        self.behaviour.act_wrapper()
        time.sleep(SLEEP_TIME_TWEAK + 0.01)
        self.behaviour.act_wrapper()

        assert (
            cast(
                APYEstimationBaseBehaviour, self.behaviour.current_behaviour
            ).behaviour_id
            == self.behaviour_class.auto_behaviour_id()
        )
        self.end_round()

    @mock.patch.object(
        BaseBehaviour,
        "send_to_ipfs",
        side_effect=wrap_dummy_ipfs_operation("hash"),
    )
    @pytest.mark.parametrize("ipfs_succeed", (True, False))
    def test_prepare_batch_behaviour(
        self,
        _send_to_ipfs: mock._patch,
        monkeypatch: MonkeyPatch,
        transformed_historical_data_no_datetime_conversion: pd.DataFrame,
        batch: ResponseItemType,
        prepare_batch_task_result: Dict[str, pd.DataFrame],
        ipfs_succeed: bool,
    ) -> None:
        """Run test for `prepare_behaviour`."""
        self._fast_forward()

        monkeypatch.setattr(TaskManager, "enqueue_task", lambda *_, **__: 0)
        monkeypatch.setattr(
            TaskManager,
            "get_task_result",
            lambda *_: DummyAsyncResult(prepare_batch_task_result),
        )

        behaviour = cast(BaseBehaviour, self.behaviour.current_behaviour)
        assert behaviour.behaviour_id == self.behaviour_class.auto_behaviour_id()

        if ipfs_succeed:
            hist = transformed_historical_data_no_datetime_conversion.iloc[
                [0, 2]
            ].reset_index(drop=True)
            with mock.patch.object(
                BaseBehaviour,
                "get_from_ipfs",
                side_effect=wrap_dummy_ipfs_operation(hist),
            ):
                self.behaviour.act_wrapper()

            with mock.patch.object(
                BaseBehaviour,
                "get_from_ipfs",
                side_effect=wrap_dummy_ipfs_operation(batch),
            ):
                self.behaviour.act_wrapper()

        else:
            with mock.patch.object(
                BaseBehaviour,
                "get_from_ipfs",
                side_effect=wrap_dummy_ipfs_operation(None),
            ):
                for _ in range(2):
                    self.behaviour.act_wrapper()

        for _ in range(2):
            self.behaviour.act_wrapper()
        self.mock_a2a_transaction()
        self._test_done_flag_set()
        self.end_round()
        behaviour = cast(BaseBehaviour, self.behaviour.current_behaviour)
        assert behaviour.behaviour_id == self.next_behaviour_class.auto_behaviour_id()


class TestRandomnessBehaviour(APYEstimationFSMBehaviourBaseCase):
    """Test RandomnessBehaviour."""

    randomness_behaviour_class = RandomnessBehaviour  # type: ignore
    next_behaviour_class = OptimizeBehaviour

    drand_response = {
        "round": 1416669,
        "randomness": "f6be4bf1fa229f22340c1a5b258f809ac4af558200775a67dacb05f0cb258a11",
        "signature": (
            "b44d00516f46da3a503f9559a634869b6dc2e5d839e46ec61a090e3032172954929a5"
            "d9bd7197d7739fe55db770543c71182562bd0ad20922eb4fe6b8a1062ed21df3b68de"
            "44694eb4f20b35262fa9d63aa80ad3f6172dd4d33a663f21179604"
        ),
        "previous_signature": (
            "903c60a4b937a804001032499a855025573040cb86017c38e2b1c3725286756ce8f33"
            "61188789c17336beaf3f9dbf84b0ad3c86add187987a9a0685bc5a303e37b008fba8c"
            "44f02a416480dd117a3ff8b8075b1b7362c58af195573623187463"
        ),
    }

    def test_randomness_behaviour(
        self,
    ) -> None:
        """Test RandomnessBehaviour."""

        self.fast_forward_to_behaviour(
            self.behaviour,
            self.randomness_behaviour_class.auto_behaviour_id(),
            self.synchronized_data,
        )
        assert (
            cast(
                BaseBehaviour,
                cast(BaseBehaviour, self.behaviour.current_behaviour),
            ).behaviour_id
            == self.randomness_behaviour_class.auto_behaviour_id()
        )
        self.behaviour.act_wrapper()
        self.mock_http_request(
            request_kwargs=dict(
                method="GET",
                headers="",
                version="",
                body=b"",
                url="https://drand.cloudflare.com/public/latest",
            ),
            response_kwargs=dict(
                version="",
                status_code=200,
                status_text="",
                headers="",
                body=json.dumps(self.drand_response).encode("utf-8"),
            ),
        )

        self.behaviour.act_wrapper()
        self.mock_a2a_transaction()
        self._test_done_flag_set()
        self.end_round()

        behaviour = cast(BaseBehaviour, self.behaviour.current_behaviour)
        assert behaviour.behaviour_id == self.next_behaviour_class.auto_behaviour_id()

    def test_invalid_drand_value(
        self,
    ) -> None:
        """Test invalid drand values."""
        self.fast_forward_to_behaviour(
            self.behaviour,
            self.randomness_behaviour_class.auto_behaviour_id(),
            self.synchronized_data,
        )
        assert (
            cast(
                BaseBehaviour,
                cast(BaseBehaviour, self.behaviour.current_behaviour),
            ).behaviour_id
            == self.randomness_behaviour_class.auto_behaviour_id()
        )
        self.behaviour.act_wrapper()

        drand_invalid = self.drand_response.copy()
        drand_invalid["randomness"] = binascii.hexlify(b"randomness_hex").decode()
        self.mock_http_request(
            request_kwargs=dict(
                method="GET",
                headers="",
                version="",
                body=b"",
                url="https://drand.cloudflare.com/public/latest",
            ),
            response_kwargs=dict(
                version="",
                status_code=200,
                status_text="",
                headers="",
                body=json.dumps(drand_invalid).encode(),
            ),
        )
        self.end_round()

    def test_invalid_response(
        self,
    ) -> None:
        """Test invalid json response."""
        self.fast_forward_to_behaviour(
            self.behaviour,
            self.randomness_behaviour_class.auto_behaviour_id(),
            self.synchronized_data,
        )
        assert (
            cast(
                BaseBehaviour,
                cast(BaseBehaviour, self.behaviour.current_behaviour),
            ).behaviour_id
            == self.randomness_behaviour_class.auto_behaviour_id()
        )

        self.behaviour.act_wrapper()

        self.mock_http_request(
            request_kwargs=dict(
                method="GET",
                headers="",
                version="",
                body=b"",
                url="https://drand.cloudflare.com/public/latest",
            ),
            response_kwargs=dict(
                version="", status_code=200, status_text="", headers="", body=b""
            ),
        )
        self.behaviour.act_wrapper()
        time.sleep(
            cast(
                BaseBehaviour, self.behaviour.current_behaviour
            ).context.randomness_api.retries_info.suggested_sleep_time
            + 0.01
        )
        self.behaviour.act_wrapper()
        self.end_round()

    def test_max_retries_reached(
        self,
    ) -> None:
        """Test with max retries reached."""
        self.fast_forward_to_behaviour(
            self.behaviour,
            self.randomness_behaviour_class.auto_behaviour_id(),
            self.synchronized_data,
        )
        assert (
            cast(
                BaseBehaviour,
                cast(BaseBehaviour, self.behaviour.current_behaviour),
            ).behaviour_id
            == self.randomness_behaviour_class.auto_behaviour_id()
        )
        with mock.patch.dict(
            self.behaviour.context.randomness_api.__dict__,
            {"is_retries_exceeded": mock.MagicMock(return_value=True)},
        ):
            self.behaviour.act_wrapper()
            behaviour = cast(BaseBehaviour, self.behaviour.current_behaviour)
            assert (
                behaviour.behaviour_id
                == self.randomness_behaviour_class.auto_behaviour_id()
            )
            self._test_done_flag_set()

        self.end_round()

    def test_clean_up(
        self,
    ) -> None:
        """Test when `observed` value is none."""
        self.fast_forward_to_behaviour(
            self.behaviour,
            self.randomness_behaviour_class.auto_behaviour_id(),
            self.synchronized_data,
        )
        assert (
            cast(
                BaseBehaviour,
                cast(BaseBehaviour, self.behaviour.current_behaviour),
            ).behaviour_id
            == self.randomness_behaviour_class.auto_behaviour_id()
        )
        self.behaviour.context.randomness_api.retries_info.retries_attempted = 1
        assert self.behaviour.current_behaviour is not None
        self.behaviour.current_behaviour.clean_up()
        assert self.behaviour.context.randomness_api.retries_info.retries_attempted == 0
        self.end_round()


class TestOptimizeBehaviour(APYEstimationFSMBehaviourBaseCase):
    """Test OptimizeBehaviour."""

    behaviour_class = OptimizeBehaviour
    next_behaviour_class = TrainBehaviour
    dummy_train_splits = {
        f"train_{i}": pd.DataFrame([i for i in range(5)]) for i in range(3)
    }

    def _fast_forward(
        self,
    ) -> None:
        """Setup `OptimizeBehaviour`."""
        # fast-forward to the `OptimizeBehaviour` behaviour.
        self.fast_forward_to_behaviour(
            self.behaviour,
            self.behaviour_class.auto_behaviour_id(),
            SynchronizedData(
                AbciAppDB(
                    setup_data=dict(
                        most_voted_randomness=[0],
                        most_voted_split=["train_test_hash"],
                    ),
                )
            ),
        )

        assert (
            cast(OptimizeBehaviour, self.behaviour.current_behaviour).behaviour_id
            == self.behaviour_class.auto_behaviour_id()
        )

    def test_prepare_task(
        self,
        monkeypatch: MonkeyPatch,
        optimize_task_result_empty: optuna.Study,
    ) -> None:
        """Test behaviour's task preparation."""
        self._fast_forward()

        monkeypatch.setattr(TaskManager, "enqueue_task", lambda *_, **__: 0)
        monkeypatch.setattr(
            TaskManager,
            "get_task_result",
            lambda *_: DummyAsyncResult(optimize_task_result_empty),
        )
        current_behaviour = cast(OptimizeBehaviour, self.behaviour.current_behaviour)
        gen = current_behaviour.prepare_task()

        with mock.patch.object(
            BaseBehaviour,
            "get_from_ipfs",
            side_effect=wrap_dummy_ipfs_operation(self.dummy_train_splits),
        ):
            next(gen)

        with pytest.raises(StopIteration):
            next(gen)

        assert current_behaviour._y == self.dummy_train_splits
        self.end_round()

    def test_task_not_ready(
        self,
        monkeypatch: MonkeyPatch,
        optimize_task_result_empty: optuna.Study,
    ) -> None:
        """Run test for behaviour when task result is not ready."""
        self._fast_forward()

        monkeypatch.setattr(TaskManager, "enqueue_task", lambda *_, **__: 0)
        monkeypatch.setattr(
            TaskManager,
            "get_task_result",
            lambda *_: DummyAsyncResult(optimize_task_result_empty, ready=False),
        )

        with mock.patch.object(
            BaseBehaviour,
            "get_from_ipfs",
            side_effect=wrap_dummy_ipfs_operation(self.dummy_train_splits),
        ):
            self.behaviour.act_wrapper()

        cast(OptimizeBehaviour, self.behaviour.current_behaviour).params.__dict__[
            "sleep_time"
        ] = SLEEP_TIME_TWEAK
        self.behaviour.act_wrapper()
        time.sleep(SLEEP_TIME_TWEAK + 0.01)
        self.behaviour.act_wrapper()

        assert (
            cast(
                APYEstimationBaseBehaviour, self.behaviour.current_behaviour
            ).behaviour_id
            == self.behaviour_class.auto_behaviour_id()
        )
        self.end_round()

    @mock.patch.object(
        BaseBehaviour,
        "send_to_ipfs",
        side_effect=wrap_dummy_ipfs_operation("hash"),
    )
    @pytest.mark.parametrize("ipfs_succeed", (True, False))
    def test_optimize_behaviour(
        self,
        _send_to_ipfs: mock._patch,
        monkeypatch: MonkeyPatch,
        caplog: LogCaptureFixture,
        optimize_task_result: PoolToHyperParamsWithStatusType,
        ipfs_succeed: bool,
    ) -> None:
        """Run test for `optimize_behaviour`."""
        self._fast_forward()

        monkeypatch.setattr(TaskManager, "enqueue_task", lambda *_, **__: 3)
        monkeypatch.setattr(
            TaskManager,
            "get_task_result",
            lambda *_: DummyAsyncResult(optimize_task_result),
        )

        split_mock = self.dummy_train_splits if ipfs_succeed else None

        with mock.patch.object(
            BaseBehaviour,
            "get_from_ipfs",
            side_effect=wrap_dummy_ipfs_operation(split_mock),
        ):
            self.behaviour.act_wrapper()

        with caplog.at_level(
            logging.WARNING,
            logger="aea.test_agent_name.packages.valory.skills.apy_estimation_abci",
        ):
            # note: this will capture the warning only if the first item in the `optimize_task_result` dict
            # is the one representing the non-finished trial, i.e., `False`.
            self.behaviour.act_wrapper()

        if ipfs_succeed:
            for _ in range(len(optimize_task_result)):
                self.behaviour.act_wrapper()
            assert (
                "The optimization could not be done for pool `test1`! "
                "Please make sure that there is a sufficient number of data for the optimization procedure. "
                "Parameters have been set randomly!"
            ) in caplog.text

        self.behaviour.act_wrapper()
        self.mock_a2a_transaction()
        self._test_done_flag_set()
        self.end_round()

        behaviour = cast(BaseBehaviour, self.behaviour.current_behaviour)
        assert behaviour.behaviour_id == self.next_behaviour_class.auto_behaviour_id()


class TestTrainBehaviour(APYEstimationFSMBehaviourBaseCase):
    """Test TrainBehaviour."""

    behaviour_class = TrainBehaviour
    next_behaviour_class = _TestBehaviour
    dummy_params = {
        "pool1.json": {"p": 1, "q": 1, "d": 1, "m": 1},
        "pool2.json": {"p": 2, "q": 2, "d": 1, "m": 1},
    }
    dummy_split = {
        f"pool{i}.csv": pd.DataFrame([i for i in range(5)]) for i in range(3)
    }

    def _fast_forward(
        self,
        full_training: bool = False,
    ) -> None:
        """Setup `TestTrainBehaviour`."""
        # fast-forward to the `TrainBehaviour` behaviour.
        self.fast_forward_to_behaviour(
            self.behaviour,
            self.behaviour_class.auto_behaviour_id(),
            SynchronizedData(
                AbciAppDB(
                    setup_data=AbciAppDB.data_to_lists(
                        dict(
                            full_training=full_training,
                            params_hash="params",
                            most_voted_split="train_test",
                        )
                    ),
                )
            ),
        )

        assert (
            cast(
                APYEstimationBaseBehaviour, self.behaviour.current_behaviour
            ).behaviour_id
            == self.behaviour_class.auto_behaviour_id()
        )

    @pytest.mark.parametrize(
        "ipfs_succeed, full_training", product((True, False), repeat=2)
    )
    def test_prepare_task(
        self,
        monkeypatch: MonkeyPatch,
        no_action: Callable[[Any], None],
        ipfs_succeed: bool,
        full_training: bool,
    ) -> None:
        """Test behaviour setup."""
        self._fast_forward(full_training)

        monkeypatch.setattr(TaskManager, "enqueue_task", lambda *_, **__: 0)
        monkeypatch.setattr(TaskManager, "get_task_result", no_action)

        current_behaviour = cast(TrainBehaviour, self.behaviour.current_behaviour)
        gen = current_behaviour.prepare_task()

        if ipfs_succeed:
            with mock.patch.object(
                BaseBehaviour,
                "get_from_ipfs",
                side_effect=wrap_dummy_ipfs_operation(self.dummy_params),
            ):
                next(gen)
            with mock.patch.object(
                BaseBehaviour,
                "get_from_ipfs",
                side_effect=wrap_dummy_ipfs_operation(self.dummy_split),
            ):
                next(gen)
                if full_training:
                    next(gen)
            with pytest.raises(StopIteration):
                next(gen)

            assert not any(
                arg is None
                for arg in (current_behaviour._y, current_behaviour._best_params)
            )
        else:
            with mock.patch.object(
                BaseBehaviour,
                "get_from_ipfs",
                side_effect=wrap_dummy_ipfs_operation(None),
            ):
                for _ in range(2):
                    next(gen)

            with pytest.raises(StopIteration):
                next(gen)

            assert all(
                arg is None
                for arg in (current_behaviour._y, current_behaviour._best_params)
            )

        self.end_round()

    def test_task_not_ready(
        self,
        monkeypatch: MonkeyPatch,
    ) -> None:
        """Run test for behaviour when task result is not ready."""
        self._fast_forward()

        self.behaviour.context.task_manager.start()

        monkeypatch.setattr(AsyncResult, "ready", lambda *_: False)
        cast(TrainBehaviour, self.behaviour.current_behaviour).params.__dict__[
            "sleep_time"
        ] = SLEEP_TIME_TWEAK

        with mock.patch.object(
            BaseBehaviour,
            "get_from_ipfs",
            side_effect=wrap_dummy_ipfs_operation(self.dummy_params),
        ):
            self.behaviour.act_wrapper()

        with mock.patch.object(
            BaseBehaviour,
            "get_from_ipfs",
            side_effect=wrap_dummy_ipfs_operation(self.dummy_split),
        ):
            for _ in range(2):
                self.behaviour.act_wrapper()

        self.behaviour.act_wrapper()
        time.sleep(SLEEP_TIME_TWEAK + 0.01)
        self.behaviour.act_wrapper()

        assert (
            cast(
                APYEstimationBaseBehaviour, self.behaviour.current_behaviour
            ).behaviour_id
            == self.behaviour_class.auto_behaviour_id()
        )
        self.end_round()

    @mock.patch.object(
        BaseBehaviour,
        "send_to_ipfs",
        side_effect=wrap_dummy_ipfs_operation("hash"),
    )
    @pytest.mark.parametrize("ipfs_succeed", (True, False))
    def test_train_behaviour(
        self,
        _send_to_ipfs: mock._patch,
        monkeypatch: MonkeyPatch,
        train_task_result: PoolIdToForecasterType,
        ipfs_succeed: bool,
    ) -> None:
        """Run test for `train_behaviour`."""
        self._fast_forward()

        monkeypatch.setattr(TaskManager, "enqueue_task", lambda *_, **__: 3)
        monkeypatch.setattr(
            TaskManager,
            "get_task_result",
            lambda *_: DummyAsyncResult(train_task_result),
        )

        with mock.patch.object(
            BaseBehaviour,
            "get_from_ipfs",
            side_effect=wrap_dummy_ipfs_operation(self.dummy_params),
        ):
            self.behaviour.act_wrapper()

        item_to_mock = self.dummy_split if ipfs_succeed else None

        with mock.patch.object(
            BaseBehaviour,
            "get_from_ipfs",
            side_effect=wrap_dummy_ipfs_operation(item_to_mock),
        ):
            for _ in range(2):
                self.behaviour.act_wrapper()

        # act.
        self.behaviour.act_wrapper()
        self.mock_a2a_transaction()
        self._test_done_flag_set()
        self.end_round()

        behaviour = cast(BaseBehaviour, self.behaviour.current_behaviour)
        assert behaviour.behaviour_id == self.next_behaviour_class.auto_behaviour_id()


class TestTestBehaviour(APYEstimationFSMBehaviourBaseCase):
    """Test TestBehaviour."""

    behaviour_class = _TestBehaviour
    next_behaviour_class = TrainBehaviour
    dummy_models = {f"pool{i}.joblib": DummyPipeline() for i in range(3)}
    dummy_splits = {
        f"pool{i}.csv": pd.DataFrame([i for i in range(5)]) for i in range(3)
    }

    def _fast_forward(
        self,
    ) -> None:
        """Setup `TestTrainBehaviour`."""

        # fast-forward to the `TestBehaviour` behaviour.
        self.fast_forward_to_behaviour(
            self.behaviour,
            self.behaviour_class.auto_behaviour_id(),
            SynchronizedData(
                AbciAppDB(
                    setup_data=dict(
                        models_hash=["model"],
                        most_voted_split=["train_test"],
                    ),
                )
            ),
        )

        assert (
            cast(
                APYEstimationBaseBehaviour, self.behaviour.current_behaviour
            ).behaviour_id
            == self.behaviour_class.auto_behaviour_id()
        )

    @pytest.mark.parametrize("ipfs_succeed", (True, False))
    def test_prepare_task(
        self,
        monkeypatch: MonkeyPatch,
        ipfs_succeed: bool,
        no_action: Callable[[Any], None],
    ) -> None:
        """Test the task preparation."""
        self._fast_forward()

        monkeypatch.setattr(TaskManager, "enqueue_task", lambda *_, **__: 0)
        monkeypatch.setattr(TaskManager, "get_task_result", no_action)

        current_behaviour = cast(_TestBehaviour, self.behaviour.current_behaviour)
        gen = current_behaviour.prepare_task()

        is_none = (
            arg is None
            for arg in (
                getattr(current_behaviour, arg_name)
                for arg_name in ("_y_train", "_y_test", "_forecasters")
            )
        )

        if ipfs_succeed:
            with mock.patch.object(
                BaseBehaviour,
                "get_from_ipfs",
                side_effect=wrap_dummy_ipfs_operation(self.dummy_splits),
            ):
                for _ in range(2):
                    next(gen)

            with mock.patch.object(
                BaseBehaviour,
                "get_from_ipfs",
                side_effect=wrap_dummy_ipfs_operation(self.dummy_models),
            ):
                next(gen)

            with pytest.raises(StopIteration):
                next(gen)

            assert not any(is_none)

        else:
            with mock.patch.object(
                BaseBehaviour,
                "get_from_ipfs",
                side_effect=wrap_dummy_ipfs_operation(None),
            ):
                for _ in range(3):
                    next(gen)

                with pytest.raises(StopIteration):
                    next(gen)

            assert all(is_none)

        self.end_round()

    def test_task_not_ready(
        self,
        monkeypatch: MonkeyPatch,
    ) -> None:
        """Run test for behaviour when task result is not ready."""
        self._fast_forward()

        self.behaviour.context.task_manager.start()
        monkeypatch.setattr(AsyncResult, "ready", lambda *_: False)
        cast(_TestBehaviour, self.behaviour.current_behaviour).params.__dict__[
            "sleep_time"
        ] = SLEEP_TIME_TWEAK

        with mock.patch.object(
            BaseBehaviour,
            "get_from_ipfs",
            side_effect=wrap_dummy_ipfs_operation(self.dummy_splits),
        ):
            for _ in range(2):
                self.behaviour.act_wrapper()

        with mock.patch.object(
            BaseBehaviour,
            "get_from_ipfs",
            side_effect=wrap_dummy_ipfs_operation(self.dummy_models),
        ):
            self.behaviour.act_wrapper()

        self.behaviour.act_wrapper()
        time.sleep(SLEEP_TIME_TWEAK + 0.01)
        self.behaviour.act_wrapper()

        assert (
            cast(
                APYEstimationBaseBehaviour, self.behaviour.current_behaviour
            ).behaviour_id
            == self.behaviour_class.auto_behaviour_id()
        )
        self.end_round()

    @mock.patch.object(
        BaseBehaviour,
        "send_to_ipfs",
        side_effect=wrap_dummy_ipfs_operation("hash"),
    )
    @pytest.mark.parametrize("ipfs_succeed", (True, False))
    def test_test_behaviour(
        self,
        _send_to_ipfs: mock._patch,
        monkeypatch: MonkeyPatch,
        ipfs_succeed: bool,
        _test_task_result: PoolIdToTestReportType,
    ) -> None:
        """Run test for `test_behaviour`."""
        self._fast_forward()

        monkeypatch.setattr(TaskManager, "enqueue_task", lambda *_, **__: 3)
        monkeypatch.setattr(
            TaskManager,
            "get_task_result",
            lambda *_: DummyAsyncResult(_test_task_result),
        )

        if ipfs_succeed:
            with mock.patch.object(
                BaseBehaviour,
                "get_from_ipfs",
                side_effect=wrap_dummy_ipfs_operation(self.dummy_splits),
            ):
                for _ in range(2):
                    self.behaviour.act_wrapper()

            with mock.patch.object(
                BaseBehaviour,
                "get_from_ipfs",
                side_effect=wrap_dummy_ipfs_operation(self.dummy_models),
            ):
                self.behaviour.act_wrapper()

            self.behaviour.act_wrapper()

        else:
            with mock.patch.object(
                BaseBehaviour,
                "get_from_ipfs",
                side_effect=wrap_dummy_ipfs_operation(None),
            ):
                for _ in range(3):
                    self.behaviour.act_wrapper()

        # test act.
        self.behaviour.act_wrapper()
        self.mock_a2a_transaction()
        self._test_done_flag_set()
        self.end_round()

        behaviour = cast(BaseBehaviour, self.behaviour.current_behaviour)
        assert behaviour.behaviour_id == self.next_behaviour_class.auto_behaviour_id()


class TestUpdateForecasterBehaviour(APYEstimationFSMBehaviourBaseCase):
    """Test `UpdateForecasterBehaviour`."""

    behaviour_class = UpdateForecasterBehaviour
    next_behaviour_class = EstimateBehaviour
    dummy_models = {f"pool{i}.joblib": DummyPipeline() for i in range(3)}

    def _fast_forward(
        self,
    ) -> None:
        """Setup `TestUpdateForecasterBehaviour`."""
        # fast-forward to the `TestBehaviour` behaviour.
        self.fast_forward_to_behaviour(
            self.behaviour,
            self.behaviour_class.auto_behaviour_id(),
            SynchronizedData(
                AbciAppDB(
                    setup_data=dict(
                        models_hash=["model"],
                        latest_observation_hist_hash=["observation"],
                    ),
                )
            ),
        )

        assert (
            cast(
                APYEstimationBaseBehaviour, self.behaviour.current_behaviour
            ).behaviour_id
            == self.behaviour_class.auto_behaviour_id()
        )

    @pytest.mark.parametrize("ipfs_succeed", (True, False))
    def test_prepare_task(
        self,
        monkeypatch: MonkeyPatch,
        no_action: Callable[[Any], None],
        prepare_batch_task_result: pd.DataFrame,
        ipfs_succeed: bool,
    ) -> None:
        """Run test for `UpdateForecasterBehaviour`'s setup method."""
        self._fast_forward()
        monkeypatch.setattr(TaskManager, "enqueue_task", lambda *_, **__: 0)
        monkeypatch.setattr(TaskManager, "get_task_result", no_action)

        current_behaviour = cast(
            UpdateForecasterBehaviour, self.behaviour.current_behaviour
        )
        assert (
            current_behaviour.behaviour_id
            == UpdateForecasterBehaviour.auto_behaviour_id()
        )
        gen = current_behaviour.prepare_task()

        if ipfs_succeed:
            with mock.patch.object(
                BaseBehaviour,
                "get_from_ipfs",
                side_effect=wrap_dummy_ipfs_operation(prepare_batch_task_result),
            ):
                next(gen)

            with mock.patch.object(
                BaseBehaviour,
                "get_from_ipfs",
                side_effect=wrap_dummy_ipfs_operation(self.dummy_models),
            ):
                next(gen)

            with pytest.raises(StopIteration):
                next(gen)

            pd.testing.assert_frame_equal(
                current_behaviour._y, prepare_batch_task_result
            )
            assert current_behaviour._forecasters == self.dummy_models

        else:
            with mock.patch.object(
                BaseBehaviour,
                "get_from_ipfs",
                side_effect=wrap_dummy_ipfs_operation(None),
            ):
                for _ in range(2):
                    next(gen)

            with pytest.raises(StopIteration):
                next(gen)

            is_none = (
                arg is None
                for arg in (
                    getattr(current_behaviour, arg_name)
                    for arg_name in ("_y", "_forecasters")
                )
            )

            assert all(is_none)

        self.end_round()

    def test_task_not_ready(
        self,
        monkeypatch: MonkeyPatch,
        prepare_batch_task_result: pd.DataFrame,
    ) -> None:
        """Run test for behaviour when task result is not ready."""
        self._fast_forward()

        self.behaviour.context.task_manager.start()

        monkeypatch.setattr(AsyncResult, "ready", lambda *_: False)
        cast(TrainBehaviour, self.behaviour.current_behaviour).params.__dict__[
            "sleep_time"
        ] = SLEEP_TIME_TWEAK

        with mock.patch.object(
            BaseBehaviour,
            "get_from_ipfs",
            side_effect=wrap_dummy_ipfs_operation(prepare_batch_task_result),
        ):
            self.behaviour.act_wrapper()

        with mock.patch.object(
            BaseBehaviour,
            "get_from_ipfs",
            side_effect=wrap_dummy_ipfs_operation(self.dummy_models),
        ):
            self.behaviour.act_wrapper()

        self.behaviour.act_wrapper()
        time.sleep(SLEEP_TIME_TWEAK + 0.01)
        self.behaviour.act_wrapper()

        assert (
            cast(
                APYEstimationBaseBehaviour, self.behaviour.current_behaviour
            ).behaviour_id
            == self.behaviour_class.auto_behaviour_id()
        )
        self.end_round()

    @mock.patch.object(
        BaseBehaviour,
        "send_to_ipfs",
        side_effect=wrap_dummy_ipfs_operation("hash"),
    )
    @pytest.mark.parametrize("ipfs_succeed", (True, False))
    def test_update_forecaster_behaviour(
        self,
        _send_to_ipfs: mock._patch,
        monkeypatch: MonkeyPatch,
        prepare_batch_task_result: pd.DataFrame,
        train_task_result: PoolIdToForecasterType,
        ipfs_succeed: bool,
    ) -> None:
        """Run test for `UpdateForecasterBehaviour`."""
        self._fast_forward()
        monkeypatch.setattr(TaskManager, "enqueue_task", lambda *_, **__: 3)
        monkeypatch.setattr(
            TaskManager,
            "get_task_result",
            lambda *_: DummyAsyncResult(train_task_result),
        )

        if ipfs_succeed:
            with mock.patch.object(
                BaseBehaviour,
                "get_from_ipfs",
                side_effect=wrap_dummy_ipfs_operation(prepare_batch_task_result),
            ):
                self.behaviour.act_wrapper()

            with mock.patch.object(
                BaseBehaviour,
                "get_from_ipfs",
                side_effect=wrap_dummy_ipfs_operation(self.dummy_models),
            ):
                self.behaviour.act_wrapper()

            self.behaviour.act_wrapper()

        else:
            with mock.patch.object(
                BaseBehaviour,
                "get_from_ipfs",
                side_effect=wrap_dummy_ipfs_operation(None),
            ):
                for _ in range(2):
                    self.behaviour.act_wrapper()

        self.behaviour.act_wrapper()
        self.mock_a2a_transaction()
        self._test_done_flag_set()
        self.end_round()
        behaviour = cast(UpdateForecasterBehaviour, self.behaviour.current_behaviour)
        assert behaviour.behaviour_id == self.next_behaviour_class.auto_behaviour_id()


class TestEstimateBehaviour(APYEstimationFSMBehaviourBaseCase):
    """Test EstimateBehaviour."""

    behaviour_class = EstimateBehaviour
    next_behaviour_class = EmitEstimatesBehaviour
    dummy_models = {f"pool{i}.joblib": DummyPipeline() for i in range(3)}

    def _fast_forward(
        self,
    ) -> None:
        """Setup `TestTransformBehaviour`."""
        self.fast_forward_to_behaviour(
            self.behaviour,
            self.behaviour_class.auto_behaviour_id(),
            SynchronizedData(
                AbciAppDB(
                    setup_data=dict(
                        models_hash=["models"],
                        transformed_history_hash=["transform"],
                        latest_transformation_period=[0],
                    ),
                )
            ),
        )

        assert (
            cast(
                APYEstimationBaseBehaviour, self.behaviour.current_behaviour
            ).behaviour_id
            == self.behaviour_class.auto_behaviour_id()
        )

    @pytest.mark.parametrize("ipfs_succeed", (True, False))
    def test_prepare_task(
        self,
        monkeypatch: MonkeyPatch,
        no_action: Callable[[Any], None],
        transformed_historical_data_no_datetime_conversion: pd.DataFrame,
        ipfs_succeed: bool,
    ) -> None:
        """Run test for `EstimateBehaviour`'s task preparation method."""
        self._fast_forward()
        monkeypatch.setattr(TaskManager, "enqueue_task", lambda *_, **__: 0)
        monkeypatch.setattr(TaskManager, "get_task_result", no_action)

        current_behaviour = cast(
            UpdateForecasterBehaviour, self.behaviour.current_behaviour
        )
        gen = current_behaviour.prepare_task()

        if ipfs_succeed:
            with mock.patch.object(
                BaseBehaviour,
                "get_from_ipfs",
                side_effect=wrap_dummy_ipfs_operation(
                    transformed_historical_data_no_datetime_conversion
                ),
            ):
                next(gen)

            with mock.patch.object(
                BaseBehaviour,
                "get_from_ipfs",
                side_effect=wrap_dummy_ipfs_operation(self.dummy_models),
            ):
                next(gen)

            with pytest.raises(StopIteration):
                next(gen)

            assert current_behaviour._forecasters is not None

        else:
            with mock.patch.object(
                BaseBehaviour,
                "get_from_ipfs",
                side_effect=wrap_dummy_ipfs_operation(None),
            ):
                for _ in range(2):
                    next(gen)

            with pytest.raises(StopIteration):
                next(gen)

            assert current_behaviour._forecasters is None

        self.end_round()

    def test_task_not_ready(
        self,
        monkeypatch: MonkeyPatch,
        transformed_historical_data_no_datetime_conversion: pd.DataFrame,
    ) -> None:
        """Run test for behaviour when task result is not ready."""
        self._fast_forward()

        self.behaviour.context.task_manager.start()

        monkeypatch.setattr(AsyncResult, "ready", lambda *_: False)
        cast(TrainBehaviour, self.behaviour.current_behaviour).params.__dict__[
            "sleep_time"
        ] = SLEEP_TIME_TWEAK

        with mock.patch.object(
            BaseBehaviour,
            "get_from_ipfs",
            side_effect=wrap_dummy_ipfs_operation(
                transformed_historical_data_no_datetime_conversion
            ),
        ):
            self.behaviour.act_wrapper()

        with mock.patch.object(
            BaseBehaviour,
            "get_from_ipfs",
            side_effect=wrap_dummy_ipfs_operation(self.dummy_models),
        ):
            self.behaviour.act_wrapper()

        self.behaviour.act_wrapper()
        time.sleep(SLEEP_TIME_TWEAK + 0.01)
        self.behaviour.act_wrapper()

        assert (
            cast(
                APYEstimationBaseBehaviour, self.behaviour.current_behaviour
            ).behaviour_id
            == self.behaviour_class.auto_behaviour_id()
        )
        self.end_round()

    @mock.patch.object(
        BaseBehaviour,
        "send_to_ipfs",
        side_effect=wrap_dummy_ipfs_operation("hash"),
    )
    @pytest.mark.parametrize("ipfs_succeed", (True, False))
    def test_estimate_behaviour(
        self,
        _send_to_ipfs: mock._patch,
        monkeypatch: MonkeyPatch,
        transformed_historical_data_no_datetime_conversion: pd.DataFrame,
        ipfs_succeed: bool,
    ) -> None:
        """Run test for `EstimateBehaviour`."""
        self._fast_forward()
        monkeypatch.setattr(TaskManager, "enqueue_task", lambda *_, **__: 0)
        monkeypatch.setattr(
            TaskManager, "get_task_result", lambda *_: DummyAsyncResult(pd.DataFrame())
        )

        if ipfs_succeed:
            with mock.patch.object(
                BaseBehaviour,
                "get_from_ipfs",
                side_effect=wrap_dummy_ipfs_operation(
                    transformed_historical_data_no_datetime_conversion
                ),
            ):
                self.behaviour.act_wrapper()

            with mock.patch.object(
                BaseBehaviour,
                "get_from_ipfs",
                side_effect=wrap_dummy_ipfs_operation(self.dummy_models),
            ):
                self.behaviour.act_wrapper()

            self.behaviour.act_wrapper()

        else:
            with mock.patch.object(
                BaseBehaviour,
                "get_from_ipfs",
                side_effect=wrap_dummy_ipfs_operation(None),
            ):
                for _ in range(2):
                    self.behaviour.act_wrapper()

        self.behaviour.act_wrapper()
        self.mock_a2a_transaction()
        self._test_done_flag_set()
        self.end_round()
        behaviour = cast(BaseBehaviour, self.behaviour.current_behaviour)
        assert behaviour.behaviour_id == self.next_behaviour_class.auto_behaviour_id()


class TestEmitEstimatesBehaviour(APYEstimationFSMBehaviourBaseCase):
    """Test `EmitEstimatesBehaviour`."""

    behaviour_class = EmitEstimatesBehaviour

    @pytest.mark.parametrize(
        "ipfs_succeed, log_level, log_message",
        (
            (True, logging.INFO, "Finalized estimates: "),
            (
                False,
                logging.ERROR,
                "There was an error while trying to fetch and load the estimations from IPFS!",
            ),
        ),
    )
    def test_get_finalized_estimates(
        self,
        caplog: LogCaptureFixture,
        ipfs_succeed: bool,
        log_level: int,
        log_message: str,
    ) -> None:
        """Test `_get_finalized_estimates` method."""
        # Send dummy estimations to IPFS and get the hash.
        self.fast_forward_to_behaviour(
            behaviour=self.behaviour,
            behaviour_id=self.behaviour_class.auto_behaviour_id(),
            synchronized_data=SynchronizedData(
                AbciAppDB(setup_data=dict(estimates_hash=["hash"]))
            ),
        )
        behaviour = cast(EmitEstimatesBehaviour, self.behaviour.current_behaviour)
        assert behaviour.behaviour_id == self.behaviour_class.auto_behaviour_id()

        gen = behaviour._get_finalized_estimates()
        estimates_mock = (
            pd.DataFrame({"pool1": [1.435, 4.234], "pool2": [3.45, 23.64]})
            if ipfs_succeed
            else None
        )

        with mock.patch.object(
            BaseBehaviour,
            "get_from_ipfs",
            side_effect=wrap_dummy_ipfs_operation(estimates_mock),
        ):
            next(gen)

        with caplog.at_level(
            log_level,
            logger="aea.test_agent_name.packages.valory.skills.apy_estimation_abci",
        ):
            with pytest.raises(StopIteration):
                next(gen)

        assert log_message in caplog.text
        self.end_round()

    @pytest.mark.parametrize(
        "input_, expected",
        (
            (
                {
                    "period_count": 0,
                    "agent_address": "test",
                    "n_participants": 0,
                    "estimations": pd.DataFrame(
                        {"pool1": [1.435, 4.234], "pool2": [3.45, 23.64]}
                    ).to_json(),
                    "total_estimations": 0,
                },
                (
                    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00000000000000"
                    b"0000000000000000test\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                    b'\x00\x00\x00\x00{"pool1":{"0":1.435,"1":4.234},"pool2":{"0":3.45,"1":23.64}}',
                ),
            ),
            (
                {
                    "period_count": 34560,
                    "agent_address": "test",
                    "n_participants": 62340,
                    "estimations": pd.DataFrame(
                        {"pool1": [1.435], "pool2": [3.45]}
                    ).to_json(),
                    "total_estimations": 67850,
                },
                (
                    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x87\x00000000000000"
                    b"0000000000000000test\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                    b"\x00\x00\xf3\x84\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\t\n"
                    b'{"pool1":{"0":1.435},"pool2":{"0":3.45}}',
                ),
            ),
        ),
    )
    def test_pack_for_server(self, input_: Dict, expected: bytes) -> None:
        """Test for `_pack_for_server` method."""
        self.fast_forward_to_behaviour(
            behaviour=self.behaviour,
            behaviour_id=self.behaviour_class.auto_behaviour_id(),
            synchronized_data=SynchronizedData(AbciAppDB(setup_data=dict())),
        )

        behaviour = cast(EmitEstimatesBehaviour, self.behaviour.current_behaviour)
        assert behaviour.behaviour_id == self.behaviour_class.auto_behaviour_id()
        actual = behaviour._pack_for_server(**input_)
        assert actual == b"".join(expected)  # type: ignore
        self.end_round()

    def test_send_to_server(self, caplog: LogCaptureFixture) -> None:
        """Test for `_send_to_server` method."""
        self.fast_forward_to_behaviour(
            behaviour=self.behaviour,
            behaviour_id=self.behaviour_class.auto_behaviour_id(),
            synchronized_data=SynchronizedData(
                AbciAppDB(setup_data=dict(participant_to_estimate=[{}]))
            ),
        )

        behaviour = cast(EmitEstimatesBehaviour, self.behaviour.current_behaviour)
        assert behaviour.behaviour_id == self.behaviour_class.auto_behaviour_id()

        expected = {"response": "test"}
        # we do this because of https://github.com/valory-xyz/open-autonomy/pull/646
        behaviour._pack_for_server = lambda **_: mock.MagicMock(hex=lambda: "test_hex")  # type: ignore
        behaviour.get_signature = lambda _, **__: iter(("test",))  # type: ignore
        behaviour.get_http_response = lambda **_: iter(("test",))  # type: ignore
        behaviour.context.server_api.__dict__["process_response"] = lambda _: expected  # type: ignore
        # init the generator
        send_gen = behaviour._send_to_server(pd.DataFrame())

        with caplog.at_level(
            logging.INFO,
            logger="aea.test_agent_name.packages.valory.skills.apy_estimation_abci",
        ), pytest.raises(StopIteration):
            for _ in range(3):
                send_gen.send(None)

        assert f"Broadcast response: {expected}" in caplog.text
        self.end_round()

    @pytest.mark.parametrize(
        "is_most_voted_estimate_set, estimations, is_broadcasting_to_server, log_level, log_message",
        (
            (
                False,
                pd.DataFrame(),
                True,
                logging.ERROR,
                "Finalized estimates not available!",
            ),
            (True, None, True, logging.ERROR, ""),
            (True, pd.DataFrame(), False, logging.ERROR, ""),
            (True, pd.DataFrame(), True, logging.ERROR, ""),
        ),
    )
    def test_emit_behaviour(
        self,
        caplog: LogCaptureFixture,
        is_most_voted_estimate_set: bool,
        estimations: pd.DataFrame,
        is_broadcasting_to_server: bool,
        log_level: int,
        log_message: str,
    ) -> None:
        """Test reset behaviour."""
        self.fast_forward_to_behaviour(
            behaviour=self.behaviour,
            behaviour_id=self.behaviour_class.auto_behaviour_id(),
            synchronized_data=SynchronizedData(
                AbciAppDB(
                    setup_data=dict(
                        estimates_hash=["not_None"]
                        if is_most_voted_estimate_set
                        else [None],
                        period_count=[0],
                    )
                )
            ),
        )
        behaviour = cast(EmitEstimatesBehaviour, self.behaviour.current_behaviour)
        assert behaviour.behaviour_id == self.behaviour_class.auto_behaviour_id()

        # we do this because of https://github.com/valory-xyz/open-autonomy/pull/646
        behaviour._send_to_server = mock.MagicMock()  # type: ignore

        behaviour.params.__dict__[
            "is_broadcasting_to_server"
        ] = is_broadcasting_to_server
        behaviour.params.__dict__["reset_pause_duration"] = SLEEP_TIME_TWEAK

        if is_most_voted_estimate_set:
            with mock.patch.object(
                BaseBehaviour,
                "get_from_ipfs",
                side_effect=wrap_dummy_ipfs_operation(estimations),
            ):
                behaviour.act_wrapper()

        with caplog.at_level(
            log_level,
            logger="aea.test_agent_name.packages.valory.skills.apy_estimation_abci",
        ):
            behaviour.act_wrapper()

        if log_message:
            assert log_message in caplog.text

        if (
            is_broadcasting_to_server
            and is_most_voted_estimate_set
            and estimations is not None
        ):
            behaviour._send_to_server.assert_called_once_with(estimations)

        time.sleep(SLEEP_TIME_TWEAK + 0.01)
        behaviour.act_wrapper()

        self.mock_a2a_transaction()
        self._test_done_flag_set()
        self.end_round()

        behaviour = cast(BaseBehaviour, self.behaviour.current_behaviour)
        assert (
            behaviour.behaviour_id
            == "degenerate_behaviour_finished_a_p_y_estimation_round"
        )
