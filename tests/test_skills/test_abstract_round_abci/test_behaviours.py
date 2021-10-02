# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2021 Valory AG
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

"""Test the behaviours.py module of the skill."""
from typing import Generator
from unittest.mock import MagicMock

from packages.valory.skills.abstract_round_abci.behaviour_utils import BaseState
from packages.valory.skills.abstract_round_abci.behaviours import AbstractRoundBehaviour


class StateA(BaseState):
    """Dummy state behaviour."""

    state_id = "state_a"

    def async_act(self) -> Generator:
        """Dummy act method."""


class ConcreteRoundBehaviour(AbstractRoundBehaviour):
    """Concrete round behaviour."""

    initial_state_cls = StateA
    transition_function = {StateA: {"loop": StateA}}


class TestAbstractRoundBehaviour:
    """Test 'AbstractRoundBehaviour' class."""

    def setup(self):
        """Set up the tests."""
        self.behaviour = ConcreteRoundBehaviour(name="", skill_context=MagicMock())

    def test_setup(self):
        """Test 'setup' method."""
        self.behaviour.setup()