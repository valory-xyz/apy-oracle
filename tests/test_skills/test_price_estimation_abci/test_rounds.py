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

"""Test the base.py module of the skill."""
import logging  # noqa: F401
import re
from types import MappingProxyType
from typing import Dict, FrozenSet, cast
from unittest.mock import MagicMock

import pytest
from aea.exceptions import AEAEnforceError

from packages.valory.skills.abstract_round_abci.base import (
    ABCIAppInternalError,
    ConsensusParams,
    TransactionNotValidError,
)
from packages.valory.skills.price_estimation_abci.payloads import (
    DeploySafePayload,
    EstimatePayload,
    FinalizationTxPayload,
    ObservationPayload,
    RandomnessPayload,
    RegistrationPayload,
    SelectKeeperPayload,
    SignaturePayload,
    TransactionHashPayload,
    ValidatePayload,
)
from packages.valory.skills.price_estimation_abci.rounds import (
    CollectObservationRound,
    CollectSignatureRound,
    ConsensusReachedRound,
    DeploySafeRound,
    EstimateConsensusRound,
    Event,
    FinalizationRound,
    PeriodState,
    RandomnessRound,
    RegistrationRound,
    SelectKeeperARound,
    SelectKeeperBRound,
    SelectKeeperRound,
    TxHashRound,
    ValidateRound,
    ValidateSafeRound,
    ValidateTransactionRound,
    encode_float,
    rotate_list,
)


MAX_PARTICIPANTS: int = 4
RANDOMNESS: str = "d1c29dce46f979f9748210d24bce4eae8be91272f5ca1a6aea2832d3dd676f51"


def get_participants() -> FrozenSet[str]:
    """Participants"""
    return frozenset([f"agent_{i}" for i in range(MAX_PARTICIPANTS)])


def get_participant_to_randomness(
    participants: FrozenSet[str], round_id: int
) -> Dict[str, RandomnessPayload]:
    """participant_to_randomness"""
    return {
        participant: RandomnessPayload(
            sender=participant,
            round_id=round_id,
            randomness=RANDOMNESS,
        )
        for participant in participants
    }


def get_most_voted_randomness() -> str:
    """most_voted_randomness"""
    return RANDOMNESS


def get_participant_to_selection(
    participants: FrozenSet[str],
) -> Dict[str, SelectKeeperPayload]:
    """participant_to_selection"""
    return {
        participant: SelectKeeperPayload(sender=participant, keeper="keeper")
        for participant in participants
    }


def get_most_voted_keeper_address() -> str:
    """most_voted_keeper_address"""
    return "keeper"


def get_safe_contract_address() -> str:
    """safe_contract_address"""
    return "0x6f6ab56aca12"


def get_participant_to_votes(
    participants: FrozenSet[str], vote: bool = True
) -> Dict[str, ValidatePayload]:
    """participant_to_votes"""
    return {
        participant: ValidatePayload(sender=participant, vote=vote)
        for participant in participants
    }


def get_participant_to_observations(
    participants: FrozenSet[str],
) -> Dict[str, ObservationPayload]:
    """participant_to_observations"""
    return {
        participant: ObservationPayload(sender=participant, observation=1.0)
        for participant in participants
    }


def get_participant_to_estimate(
    participants: FrozenSet[str],
) -> Dict[str, EstimatePayload]:
    """participant_to_estimate"""
    return {
        participant: EstimatePayload(sender=participant, estimate=1.0)
        for participant in participants
    }


def get_estimate() -> float:
    """Estimate"""
    return 1.0


def get_most_voted_estimate() -> float:
    """most_voted_estimate"""
    return 1.0


def get_participant_to_tx_hash(
    participants: FrozenSet[str],
) -> Dict[str, TransactionHashPayload]:
    """participant_to_tx_hash"""
    return {
        participant: TransactionHashPayload(sender=participant, tx_hash="tx_hash")
        for participant in participants
    }


def get_most_voted_tx_hash() -> str:
    """most_voted_tx_hash"""
    return "tx_hash"


def get_participant_to_signature(participants: FrozenSet[str]) -> Dict[str, str]:
    """participant_to_signature"""
    return {participant: "signature" for participant in participants}


def get_final_tx_hash() -> str:
    """final_tx_hash"""
    return "tx_hash"


class BaseRoundTestClass:
    """Base test class for Rounds."""

    period_state: PeriodState
    consensus_params: ConsensusParams
    participants: FrozenSet[str]

    @classmethod
    def setup(
        cls,
    ) -> None:
        """Setup the test class."""

        cls.participants = get_participants()
        cls.period_state = PeriodState(participants=cls.participants)
        cls.consensus_params = ConsensusParams(max_participants=MAX_PARTICIPANTS)


class TestRegistrationRound(BaseRoundTestClass):
    """Test RegistrationRound."""

    def test_run(
        self,
    ) -> None:
        """Run test."""

        test_round = RegistrationRound(
            state=self.period_state, consensus_params=self.consensus_params
        )

        registration_payloads = [
            RegistrationPayload(sender=participant) for participant in self.participants
        ]

        first_participant = registration_payloads.pop(0)
        test_round.process_payload(first_participant)
        assert test_round.participants == {
            first_participant.sender,
        }
        assert test_round.end_block() is None

        for participant_payload in registration_payloads:
            test_round.process_payload(participant_payload)
        assert test_round.registration_threshold_reached

        actual_next_state = PeriodState(participants=test_round.participants)

        res = test_round.end_block()
        assert res is not None
        state, event = res
        assert (
            cast(PeriodState, state).participants
            == cast(PeriodState, actual_next_state).participants
        )
        assert event == Event.DONE


class TestRandomnessRound(BaseRoundTestClass):
    """Test RandomnessRound."""

    def test_run(
        self,
    ) -> None:
        """Run tests."""

        test_round = RandomnessRound(self.period_state, self.consensus_params)

        randomness_payloads = get_participant_to_randomness(self.participants, 1)
        first_payload = randomness_payloads.pop(
            sorted(list(randomness_payloads.keys()))[0]
        )
        test_round.process_payload(first_payload)

        assert (
            test_round.participant_to_randomness[first_payload.sender] == first_payload
        )
        assert not test_round.threshold_reached
        assert test_round.end_block() is None

        with pytest.raises(ABCIAppInternalError, match="not enough randomness"):
            _ = test_round.most_voted_randomness

        with pytest.raises(
            ABCIAppInternalError,
            match=re.escape(
                "internal error: sender sender is not in the set of participants: ['agent_0', 'agent_1', 'agent_2', 'agent_3']"
            ),
        ):
            test_round.process_payload(
                RandomnessPayload(sender="sender", round_id=0, randomness="")
            )

        with pytest.raises(
            ABCIAppInternalError,
            match=f"internal error: sender agent_0 has already sent the randomness: {RANDOMNESS}",
        ):
            test_round.process_payload(first_payload)

        with pytest.raises(
            TransactionNotValidError,
            match=f"sender agent_0 has already sent the randomness: {RANDOMNESS}",
        ):
            test_round.check_payload(first_payload)

        with pytest.raises(TransactionNotValidError):
            test_round.check_payload(
                RandomnessPayload(sender="sender", round_id=0, randomness="")
            )

        for randomness_payload in randomness_payloads.values():
            test_round.process_payload(randomness_payload)
        assert test_round.most_voted_randomness == RANDOMNESS
        assert test_round.threshold_reached

        actual_next_state = self.period_state.update(
            participant_to_randomness=MappingProxyType(
                dict(get_participant_to_randomness(self.participants, 1))
            )
        )

        res = test_round.end_block()
        assert res is not None
        state, event = res
        assert (
            cast(PeriodState, state).participant_to_randomness.keys()
            == cast(PeriodState, actual_next_state).participant_to_randomness.keys()
        )
        assert event == Event.DONE


class TestSelectKeeperRound(BaseRoundTestClass):
    """Test SelectKeeperRound"""

    @classmethod
    def setup(cls) -> None:
        """Set up the test."""
        super().setup()
        SelectKeeperRound.round_id = "round_id"

    def teardown(self) -> None:
        """Tear down the test."""
        delattr(SelectKeeperRound, "round_id")

    def test_run(
        self,
    ) -> None:
        """Run tests."""

        test_round = SelectKeeperRound(
            state=self.period_state, consensus_params=self.consensus_params
        )

        select_keeper_payloads = get_participant_to_selection(self.participants)
        first_payload = select_keeper_payloads.pop(
            sorted(list(select_keeper_payloads.keys()))[0]
        )

        test_round.process_payload(first_payload)
        assert (
            test_round.participant_to_selection[first_payload.sender] == first_payload
        )
        assert not test_round.selection_threshold_reached
        assert test_round.end_block() is None

        with pytest.raises(ABCIAppInternalError, match="keeper has not enough votes"):
            _ = test_round.most_voted_keeper_address

        with pytest.raises(
            ABCIAppInternalError,
            match=re.escape(
                "internal error: sender sender is not in the set of participants: ['agent_0', 'agent_1', 'agent_2', 'agent_3']"
            ),
        ):
            test_round.process_payload(SelectKeeperPayload(sender="sender", keeper=""))

        with pytest.raises(
            ABCIAppInternalError,
            match="internal error: sender agent_0 has already sent the selection: keeper",
        ):
            test_round.process_payload(first_payload)

        with pytest.raises(
            TransactionNotValidError,
            match="sender agent_0 has already sent the selection: keeper",
        ):
            test_round.check_payload(first_payload)

        with pytest.raises(TransactionNotValidError):
            test_round.check_payload(
                SelectKeeperPayload(sender="sender", keeper="keeper")
            )

        for payload in select_keeper_payloads.values():
            test_round.process_payload(payload)
        assert test_round.selection_threshold_reached
        assert test_round.most_voted_keeper_address == "keeper"

        actual_next_state = self.period_state.update(
            participant_to_selection=MappingProxyType(
                dict(get_participant_to_selection(self.participants))
            )
        )

        res = test_round.end_block()
        assert res is not None
        state, event = res
        assert (
            cast(PeriodState, state).participant_to_selection.keys()
            == cast(PeriodState, actual_next_state).participant_to_selection.keys()
        )
        assert event == Event.DONE


class TestDeploySafeRound(BaseRoundTestClass):
    """Test DeploySafeRound."""

    def test_run(
        self,
    ) -> None:
        """Run tests."""

        self.period_state = cast(
            PeriodState,
            self.period_state.update(
                most_voted_keeper_address=sorted(list(self.participants))[0]
            ),
        )

        test_round = DeploySafeRound(
            state=self.period_state, consensus_params=self.consensus_params
        )

        with pytest.raises(
            TransactionNotValidError,
            match=re.escape(
                "sender sender is not in the set of participants: ['agent_0', 'agent_1', 'agent_2', 'agent_3']"
            ),
        ):
            test_round.check_payload(
                DeploySafePayload(
                    sender="sender", safe_contract_address=get_safe_contract_address()
                )
            )

        with pytest.raises(
            TransactionNotValidError,
            match="sender agent_1 is not the elected sender: agent_0",
        ):
            test_round.check_payload(
                DeploySafePayload(
                    sender=sorted(list(self.participants))[1],
                    safe_contract_address=get_safe_contract_address(),
                )
            )

        assert not test_round.is_contract_set
        assert test_round.end_block() is None

        with pytest.raises(
            ABCIAppInternalError,
            match=re.escape(
                "internal error: sender sender is not in the set of participants: ['agent_0', 'agent_1', 'agent_2', 'agent_3']"
            ),
        ):
            test_round.process_payload(
                DeploySafePayload(
                    sender="sender", safe_contract_address=get_safe_contract_address()
                )
            )

        with pytest.raises(
            ABCIAppInternalError,
            match="internal error: sender agent_1 is not the elected sender: agent_0",
        ):
            test_round.process_payload(
                DeploySafePayload(
                    sender=sorted(list(self.participants))[1],
                    safe_contract_address=get_safe_contract_address(),
                )
            )

        test_round.process_payload(
            DeploySafePayload(
                sender=sorted(list(self.participants))[0],
                safe_contract_address=get_safe_contract_address(),
            )
        )

        with pytest.raises(
            ABCIAppInternalError,
            match="internal error: sender agent_0 has already sent the contract address: 0x6f6ab56aca12",
        ):
            test_round.process_payload(
                DeploySafePayload(
                    sender=sorted(list(self.participants))[0],
                    safe_contract_address=get_safe_contract_address(),
                )
            )

        with pytest.raises(
            TransactionNotValidError,
            match="sender agent_0 has already sent the contract address: 0x6f6ab56aca12",
        ):
            test_round.check_payload(
                DeploySafePayload(
                    sender=sorted(list(self.participants))[0],
                    safe_contract_address=get_safe_contract_address(),
                )
            )

        assert test_round.is_contract_set
        actual_state = self.period_state.update(
            safe_contract_address=get_safe_contract_address()
        )
        res = test_round.end_block()
        assert res is not None
        state, event = res
        assert (
            cast(PeriodState, state).safe_contract_address
            == cast(PeriodState, actual_state).safe_contract_address
        )
        assert event == Event.DONE


class TestValidateRound(BaseRoundTestClass):
    """Test ValidateRound."""

    @classmethod
    def setup(cls) -> None:
        """Set up the test."""
        super().setup()
        ValidateRound.exit_event = Event.EXIT_A
        ValidateRound.round_id = "round_id"

    def teardown(self) -> None:
        """Tear down the test."""
        delattr(ValidateRound, "exit_event")
        delattr(ValidateRound, "round_id")

    def test_positive_votes(
        self,
    ) -> None:
        """Test ValidateRound."""

        test_round = ValidateRound(
            state=self.period_state, consensus_params=self.consensus_params
        )

        with pytest.raises(
            TransactionNotValidError,
            match=re.escape(
                "sender sender is not in the set of participants: ['agent_0', 'agent_1', 'agent_2', 'agent_3']"
            ),
        ):
            test_round.process_payload(ValidatePayload(sender="sender", vote=True))

        with pytest.raises(
            TransactionNotValidError,
            match=re.escape(
                "sender sender is not in the set of participants: ['agent_0', 'agent_1', 'agent_2', 'agent_3']"
            ),
        ):
            test_round.check_payload(ValidatePayload(sender="sender", vote=True))

        participant_to_votes_payloads = get_participant_to_votes(self.participants)
        first_payload = participant_to_votes_payloads.pop(
            sorted(list(participant_to_votes_payloads.keys()))[0]
        )
        test_round.process_payload(first_payload)

        with pytest.raises(
            ABCIAppInternalError,
            match="internal error: sender agent_0 has already sent its vote: True",
        ):
            test_round.process_payload(first_payload)

        assert test_round.participant_to_votes[first_payload.sender] == first_payload

        with pytest.raises(
            TransactionNotValidError,
            match="sender agent_0 has already sent its vote: True",
        ):
            test_round.check_payload(first_payload)

        assert test_round.end_block() is None
        assert not test_round.positive_vote_threshold_reached
        for payload in participant_to_votes_payloads.values():
            test_round.process_payload(payload)

        assert test_round.positive_vote_threshold_reached

        actual_next_state = self.period_state.update(
            participant_to_votes=MappingProxyType(
                dict(get_participant_to_votes(self.participants))
            )
        )
        res = test_round.end_block()
        assert res is not None
        state, event = res
        assert (
            cast(PeriodState, state).participant_to_votes.keys()
            == cast(PeriodState, actual_next_state).participant_to_votes.keys()
        )
        assert event == Event.DONE

    def test_negative_votes(
        self,
    ) -> None:
        """Test ValidateRound."""

        test_round = ValidateRound(
            state=self.period_state, consensus_params=self.consensus_params
        )

        participant_to_votes_payloads = get_participant_to_votes(
            self.participants, vote=False
        )
        first_payload = participant_to_votes_payloads.pop(
            sorted(list(participant_to_votes_payloads.keys()))[0]
        )
        test_round.process_payload(first_payload)

        assert test_round.participant_to_votes[first_payload.sender] == first_payload
        assert test_round.end_block() is None
        assert not test_round.negative_vote_threshold_reached
        for payload in participant_to_votes_payloads.values():
            test_round.process_payload(payload)

        assert test_round.negative_vote_threshold_reached

        res = test_round.end_block()
        assert res is not None
        state, event = res
        assert event == Event.EXIT_A
        with pytest.raises(
            AEAEnforceError, match="'participant_to_votes' field is None"
        ):
            _ = cast(PeriodState, state).participant_to_votes


class TestCollectObservationRound(BaseRoundTestClass):
    """Test CollectObservationRound."""

    def test_run(
        self,
    ) -> None:
        """Runs tests."""

        test_round = CollectObservationRound(
            state=self.period_state, consensus_params=self.consensus_params
        )

        with pytest.raises(
            ABCIAppInternalError,
            match=re.escape(
                "internal error: sender sender is not in the set of participants: ['agent_0', 'agent_1', 'agent_2', 'agent_3']"
            ),
        ):
            test_round.process_payload(
                ObservationPayload(
                    sender="sender",
                    observation=1.0,
                )
            )

        with pytest.raises(
            TransactionNotValidError,
            match=re.escape(
                "sender sender is not in the set of participants: ['agent_0', 'agent_1', 'agent_2', 'agent_3']"
            ),
        ):
            test_round.check_payload(
                ObservationPayload(
                    sender="sender",
                    observation=1.0,
                )
            )

        participant_to_observations_payloads = get_participant_to_observations(
            self.participants
        )
        first_payload = participant_to_observations_payloads.pop(
            sorted(list(participant_to_observations_payloads.keys()))[0]
        )

        test_round.process_payload(first_payload)
        assert (
            test_round.participant_to_observations[first_payload.sender]
            == first_payload
        )
        assert not test_round.observation_threshold_reached
        assert test_round.end_block() is None

        with pytest.raises(
            ABCIAppInternalError,
            match="internal error: sender agent_0 has already sent its observation: 1.0",
        ):
            test_round.process_payload(first_payload)

        with pytest.raises(
            TransactionNotValidError,
            match="sender agent_0 has already sent its observation: 1.0",
        ):
            test_round.check_payload(
                ObservationPayload(
                    sender=sorted(list(self.participants))[0],
                    observation=1.0,
                )
            )

        for payload in participant_to_observations_payloads.values():
            test_round.process_payload(payload)

        assert test_round.observation_threshold_reached
        actual_next_state = self.period_state.update(
            participant_to_observations=dict(
                get_participant_to_observations(self.participants)
            )
        )
        res = test_round.end_block()
        assert res is not None
        state, event = res
        assert (
            cast(PeriodState, state).participant_to_observations.keys()
            == cast(PeriodState, actual_next_state).participant_to_observations.keys()
        )
        assert event == Event.DONE


class TestEstimateConsensusRound(BaseRoundTestClass):
    """Test EstimateConsensusRound."""

    def test_run(
        self,
    ) -> None:
        """Runs test."""

        test_round = EstimateConsensusRound(
            state=self.period_state, consensus_params=self.consensus_params
        )

        with pytest.raises(
            ABCIAppInternalError,
            match=re.escape(
                "internal error: sender sender is not in the set of participants: ['agent_0', 'agent_1', 'agent_2', 'agent_3']"
            ),
        ):
            test_round.process_payload(EstimatePayload(sender="sender", estimate=1.0))

        with pytest.raises(
            TransactionNotValidError,
            match=re.escape(
                "sender sender is not in the set of participants: ['agent_0', 'agent_1', 'agent_2', 'agent_3']"
            ),
        ):
            test_round.check_payload(EstimatePayload(sender="sender", estimate=1.0))

        participant_to_estimate_payloads = get_participant_to_estimate(
            self.participants
        )

        first_payload = participant_to_estimate_payloads.pop(
            sorted(list(participant_to_estimate_payloads.keys()))[0]
        )
        test_round.process_payload(first_payload)

        assert test_round.participant_to_estimate[first_payload.sender] == first_payload
        assert test_round.end_block() is None
        assert not test_round.estimate_threshold_reached

        with pytest.raises(ABCIAppInternalError, match="estimate has not enough votes"):
            _ = test_round.most_voted_estimate

        with pytest.raises(
            ABCIAppInternalError,
            match="internal error: sender agent_0 has already sent the estimate: 1.0",
        ):
            test_round.process_payload(first_payload)

        with pytest.raises(
            TransactionNotValidError,
            match="sender agent_0 has already sent the estimate: 1.0",
        ):
            test_round.check_payload(
                EstimatePayload(sender=sorted(list(self.participants))[0], estimate=1.0)
            )

        for payload in participant_to_estimate_payloads.values():
            test_round.process_payload(payload)

        assert test_round.estimate_threshold_reached
        assert test_round.most_voted_estimate == 1.0

        actual_next_state = self.period_state.update(
            participant_to_estimate=dict(
                get_participant_to_estimate(self.participants)
            ),
            most_voted_estimate=test_round.most_voted_estimate,
        )
        res = test_round.end_block()
        assert res is not None
        state, event = res
        assert (
            cast(PeriodState, state).participant_to_estimate.keys()
            == cast(PeriodState, actual_next_state).participant_to_estimate.keys()
        )
        assert event == Event.DONE


class TestTxHashRound(BaseRoundTestClass):
    """Test TxHashRound."""

    def test_run(
        self,
    ) -> None:
        """Runs test."""

        test_round = TxHashRound(
            state=self.period_state, consensus_params=self.consensus_params
        )

        participant_to_tx_hash_payloads = get_participant_to_tx_hash(self.participants)
        first_payload = participant_to_tx_hash_payloads.pop(
            sorted(list(participant_to_tx_hash_payloads.keys()))[0]
        )

        test_round.process_payload(first_payload)

        with pytest.raises(
            ABCIAppInternalError,
            match=re.escape(
                "internal error: sender sender is not in the set of participants: ['agent_0', 'agent_1', 'agent_2', 'agent_3']"
            ),
        ):
            test_round.process_payload(
                TransactionHashPayload(sender="sender", tx_hash="tx_hash")
            )

        with pytest.raises(
            TransactionNotValidError,
            match=re.escape(
                "sender sender is not in the set of participants: ['agent_0', 'agent_1', 'agent_2', 'agent_3']"
            ),
        ):
            test_round.check_payload(
                TransactionHashPayload(sender="sender", tx_hash="tx_hash")
            )

        assert test_round.participant_to_tx_hash[first_payload.sender] == first_payload
        assert test_round.end_block() is None
        assert not test_round.tx_threshold_reached

        with pytest.raises(ABCIAppInternalError, match="tx hash has not enough votes"):
            _ = test_round.most_voted_tx_hash

        with pytest.raises(
            ABCIAppInternalError,
            match="internal error: sender agent_0 has already sent the tx hash: tx_hash",
        ):
            test_round.process_payload(first_payload)

        with pytest.raises(
            TransactionNotValidError,
            match="sender agent_0 has already sent the tx hash: tx_hash",
        ):
            test_round.check_payload(first_payload)

        for payload in participant_to_tx_hash_payloads.values():
            test_round.process_payload(payload)

        assert test_round.tx_threshold_reached
        assert test_round.most_voted_tx_hash == "tx_hash"
        res = test_round.end_block()
        assert res is not None
        _, event = res
        assert event == Event.DONE


class TestCollectSignatureRound(BaseRoundTestClass):
    """Test CollectSignatureRound."""

    def test_run(
        self,
    ) -> None:
        """Runs tests."""

        test_round = CollectSignatureRound(
            state=self.period_state, consensus_params=self.consensus_params
        )

        with pytest.raises(
            ABCIAppInternalError,
            match=re.escape(
                "internal error: sender sender is not in the set of participants: ['agent_0', 'agent_1', 'agent_2', 'agent_3']"
            ),
        ):
            test_round.process_payload(
                SignaturePayload(sender="sender", signature="signature")
            )

        with pytest.raises(
            TransactionNotValidError,
            match=re.escape(
                "sender sender is not in the set of participants: ['agent_0', 'agent_1', 'agent_2', 'agent_3']"
            ),
        ):
            test_round.check_payload(
                SignaturePayload(sender="sender", signature="signature")
            )

        participant_to_signature = {
            participant: SignaturePayload(sender=participant, signature=signature)
            for participant, signature in get_participant_to_signature(
                self.participants
            ).items()
        }
        first_payload = participant_to_signature.pop(
            sorted(list(participant_to_signature.keys()))[0]
        )

        test_round.process_payload(first_payload)
        assert not test_round.signature_threshold_reached
        assert (
            test_round.signatures_by_participant[first_payload.sender]
            == first_payload.signature
        )
        assert test_round.end_block() is None

        with pytest.raises(
            ABCIAppInternalError,
            match="internal error: sender agent_0 has already sent its signature: signature",
        ):
            test_round.process_payload(first_payload)

        with pytest.raises(
            TransactionNotValidError,
            match="sender agent_0 has already sent its signature: signature",
        ):
            test_round.check_payload(first_payload)

        for payload in participant_to_signature.values():
            test_round.process_payload(payload)

        res = test_round.end_block()
        assert res is not None
        _, event = res
        assert event == Event.DONE


class TestFinalizationRound(BaseRoundTestClass):
    """Test FinalizationRound."""

    def test_run(
        self,
    ) -> None:
        """Runs tests."""

        self.period_state = cast(
            PeriodState,
            self.period_state.update(
                most_voted_keeper_address=sorted(list(self.participants))[0]
            ),
        )

        test_round = FinalizationRound(
            state=self.period_state,
            consensus_params=self.consensus_params,
        )

        with pytest.raises(
            ABCIAppInternalError,
            match=re.escape(
                "internal error: sender sender is not in the set of participants: ['agent_0', 'agent_1', 'agent_2', 'agent_3']"
            ),
        ):
            test_round.process_payload(
                FinalizationTxPayload(sender="sender", tx_hash=get_final_tx_hash())
            )

        with pytest.raises(
            TransactionNotValidError,
            match=re.escape(
                "sender sender is not in the set of participants: ['agent_0', 'agent_1', 'agent_2', 'agent_3']"
            ),
        ):
            test_round.check_payload(
                FinalizationTxPayload(sender="sender", tx_hash=get_final_tx_hash())
            )

        with pytest.raises(
            ABCIAppInternalError,
            match="internal error: sender agent_1 is not the elected sender: agent_0",
        ):
            test_round.process_payload(
                FinalizationTxPayload(
                    sender=sorted(list(self.participants))[1],
                    tx_hash=get_final_tx_hash(),
                )
            )

        with pytest.raises(
            TransactionNotValidError,
            match="sender agent_1 is not the elected sender: agent_0",
        ):
            test_round.check_payload(
                FinalizationTxPayload(
                    sender=sorted(list(self.participants))[1],
                    tx_hash=get_final_tx_hash(),
                )
            )

        assert not test_round.tx_hash_set
        assert test_round.end_block() is None

        test_round.process_payload(
            FinalizationTxPayload(
                sender=sorted(list(self.participants))[0], tx_hash=get_final_tx_hash()
            )
        )

        assert test_round.tx_hash_set

        with pytest.raises(
            ABCIAppInternalError,
            match="internal error: sender agent_0 has already sent the tx hash: tx_hash",
        ):
            test_round.process_payload(
                FinalizationTxPayload(
                    sender=sorted(list(self.participants))[0],
                    tx_hash=get_final_tx_hash(),
                )
            )

        with pytest.raises(
            TransactionNotValidError,
            match="sender agent_0 has already sent the tx hash: tx_hash",
        ):
            test_round.check_payload(
                FinalizationTxPayload(
                    sender=sorted(list(self.participants))[0],
                    tx_hash=get_final_tx_hash(),
                )
            )

        actual_next_state = self.period_state.update(final_tx_hash=get_final_tx_hash())
        res = test_round.end_block()
        assert res is not None
        state, event = res
        assert (
            cast(PeriodState, state).final_tx_hash
            == cast(PeriodState, actual_next_state).final_tx_hash
        )
        assert event == Event.DONE


class TestSelectKeeperARound(BaseRoundTestClass):
    """Test SelectKeeperARound"""

    def test_run(
        self,
    ) -> None:
        """Run tests."""

        test_round = SelectKeeperARound(
            state=self.period_state, consensus_params=self.consensus_params
        )

        select_keeper_payloads = get_participant_to_selection(self.participants)
        first_payload = select_keeper_payloads.pop(
            sorted(list(select_keeper_payloads.keys()))[0]
        )

        test_round.process_payload(first_payload)
        assert (
            test_round.participant_to_selection[first_payload.sender] == first_payload
        )
        assert not test_round.selection_threshold_reached
        assert test_round.end_block() is None

        with pytest.raises(ABCIAppInternalError, match="keeper has not enough votes"):
            _ = test_round.most_voted_keeper_address

        with pytest.raises(ABCIAppInternalError):
            test_round.process_payload(SelectKeeperPayload(sender="sender", keeper=""))

        with pytest.raises(
            ABCIAppInternalError,
            match="internal error: sender agent_0 has already sent the selection: keeper",
        ):
            test_round.process_payload(first_payload)

        with pytest.raises(
            TransactionNotValidError,
            match="sender agent_0 has already sent the selection: keeper",
        ):
            test_round.check_payload(first_payload)

        with pytest.raises(
            TransactionNotValidError,
            match=re.escape(
                "sender sender is not in the set of participants: ['agent_0', 'agent_1', 'agent_2', 'agent_3']"
            ),
        ):
            test_round.check_payload(
                SelectKeeperPayload(sender="sender", keeper="keeper")
            )

        for payload in select_keeper_payloads.values():
            test_round.process_payload(payload)
        assert test_round.selection_threshold_reached
        assert test_round.most_voted_keeper_address == "keeper"

        actual_next_state = self.period_state.update(
            participant_to_selection=MappingProxyType(
                dict(get_participant_to_selection(self.participants))
            )
        )

        res = test_round.end_block()
        assert res is not None
        state, event = res
        assert (
            cast(PeriodState, state).participant_to_selection.keys()
            == cast(PeriodState, actual_next_state).participant_to_selection.keys()
        )
        assert event == Event.DONE


class TestSelectKeeperBRound(BaseRoundTestClass):
    """Test SelectKeeperBRound."""

    def test_run(
        self,
    ) -> None:
        """Run tests."""

        test_round = SelectKeeperBRound(
            state=self.period_state, consensus_params=self.consensus_params
        )

        select_keeper_payloads = get_participant_to_selection(self.participants)
        first_payload = select_keeper_payloads.pop(
            sorted(list(select_keeper_payloads.keys()))[0]
        )

        test_round.process_payload(first_payload)
        assert (
            test_round.participant_to_selection[first_payload.sender] == first_payload
        )
        assert not test_round.selection_threshold_reached
        assert test_round.end_block() is None

        with pytest.raises(ABCIAppInternalError, match="keeper has not enough votes"):
            _ = test_round.most_voted_keeper_address

        with pytest.raises(ABCIAppInternalError):
            test_round.process_payload(SelectKeeperPayload(sender="sender", keeper=""))

        with pytest.raises(
            ABCIAppInternalError,
            match="internal error: sender agent_0 has already sent the selection: keeper",
        ):
            test_round.process_payload(first_payload)

        with pytest.raises(
            TransactionNotValidError,
            match="sender agent_0 has already sent the selection: keeper",
        ):
            test_round.check_payload(first_payload)

        with pytest.raises(
            TransactionNotValidError,
            match=re.escape(
                "sender sender is not in the set of participants: ['agent_0', 'agent_1', 'agent_2', 'agent_3']"
            ),
        ):
            test_round.check_payload(
                SelectKeeperPayload(sender="sender", keeper="keeper")
            )

        for payload in select_keeper_payloads.values():
            test_round.process_payload(payload)
        assert test_round.selection_threshold_reached
        assert test_round.most_voted_keeper_address == "keeper"

        actual_next_state = self.period_state.update(
            participant_to_selection=MappingProxyType(
                dict(get_participant_to_selection(self.participants))
            )
        )

        res = test_round.end_block()
        assert res is not None
        state, event = res
        assert (
            cast(PeriodState, state).participant_to_selection.keys()
            == cast(PeriodState, actual_next_state).participant_to_selection.keys()
        )
        assert event == Event.DONE


class TestConsensusReachedRound(BaseRoundTestClass):
    """Test ConsensusReachedRound."""

    def test_runs(
        self,
    ) -> None:
        """Runs tests."""

        test_round = ConsensusReachedRound(
            state=self.period_state, consensus_params=self.consensus_params
        )

        assert test_round.end_block() is None

        with pytest.raises(
            TransactionNotValidError, match="this round does not accept transactions"
        ):
            test_round.check_payload(MagicMock())

        with pytest.raises(
            ABCIAppInternalError, match="this round does not accept transactions"
        ):
            test_round.process_payload(MagicMock())


class TestValidateSafeRound(BaseRoundTestClass):
    """Test ValidateSafeRound."""

    def test_positive_votes(
        self,
    ) -> None:
        """Test ValidateRound."""

        test_round = ValidateSafeRound(
            state=self.period_state, consensus_params=self.consensus_params
        )

        with pytest.raises(
            TransactionNotValidError,
            match=re.escape(
                "sender sender is not in the set of participants: ['agent_0', 'agent_1', 'agent_2', 'agent_3']"
            ),
        ):
            test_round.check_payload(ValidatePayload(sender="sender", vote=True))

        with pytest.raises(
            TransactionNotValidError,
            match=re.escape(
                "sender sender is not in the set of participants: ['agent_0', 'agent_1', 'agent_2', 'agent_3']"
            ),
        ):
            test_round.check_payload(ValidatePayload(sender="sender", vote=True))

        participant_to_votes_payloads = get_participant_to_votes(self.participants)
        first_payload = participant_to_votes_payloads.pop(
            sorted(list(participant_to_votes_payloads.keys()))[0]
        )
        test_round.process_payload(first_payload)

        with pytest.raises(
            ABCIAppInternalError,
            match="internal error: sender agent_0 has already sent its vote: True",
        ):
            test_round.process_payload(first_payload)

        assert test_round.participant_to_votes[first_payload.sender] == first_payload

        with pytest.raises(
            TransactionNotValidError,
            match="sender agent_0 has already sent its vote: True",
        ):
            test_round.check_payload(first_payload)

        assert test_round.end_block() is None
        assert not test_round.positive_vote_threshold_reached
        for payload in participant_to_votes_payloads.values():
            test_round.process_payload(payload)

        assert test_round.positive_vote_threshold_reached

        actual_next_state = self.period_state.update(
            participant_to_votes=MappingProxyType(
                dict(get_participant_to_votes(self.participants))
            )
        )
        res = test_round.end_block()
        assert res is not None
        state, event = res
        assert (
            cast(PeriodState, state).participant_to_votes.keys()
            == cast(PeriodState, actual_next_state).participant_to_votes.keys()
        )
        assert event == Event.DONE

    def test_negative_votes(
        self,
    ) -> None:
        """Test ValidateRound."""

        test_round = ValidateSafeRound(
            state=self.period_state, consensus_params=self.consensus_params
        )

        participant_to_votes_payloads = get_participant_to_votes(
            self.participants, vote=False
        )
        first_payload = participant_to_votes_payloads.pop(
            sorted(list(participant_to_votes_payloads.keys()))[0]
        )
        test_round.process_payload(first_payload)

        assert test_round.participant_to_votes[first_payload.sender] == first_payload
        assert test_round.end_block() is None
        assert not test_round.negative_vote_threshold_reached
        for payload in participant_to_votes_payloads.values():
            test_round.process_payload(payload)

        assert test_round.negative_vote_threshold_reached

        res = test_round.end_block()
        assert res is not None
        state, event = res
        assert event == Event.EXIT_A
        with pytest.raises(
            AEAEnforceError, match="'participant_to_votes' field is None"
        ):
            _ = cast(PeriodState, state).participant_to_votes


class TestValidateTransactionRound(BaseRoundTestClass):
    """Test ValidateRound."""

    def test_positive_votes(
        self,
    ) -> None:
        """Test ValidateRound."""

        test_round = ValidateTransactionRound(
            state=self.period_state, consensus_params=self.consensus_params
        )

        with pytest.raises(
            TransactionNotValidError,
            match=re.escape(
                "sender sender is not in the set of participants: ['agent_0', 'agent_1', 'agent_2', 'agent_3']"
            ),
        ):
            test_round.process_payload(ValidatePayload(sender="sender", vote=True))

        with pytest.raises(
            TransactionNotValidError,
            match=re.escape(
                "sender sender is not in the set of participants: ['agent_0', 'agent_1', 'agent_2', 'agent_3']"
            ),
        ):
            test_round.check_payload(ValidatePayload(sender="sender", vote=True))

        participant_to_votes_payloads = get_participant_to_votes(self.participants)
        first_payload = participant_to_votes_payloads.pop(
            sorted(list(participant_to_votes_payloads.keys()))[0]
        )
        test_round.process_payload(first_payload)

        with pytest.raises(
            ABCIAppInternalError,
            match="internal error: sender agent_0 has already sent its vote: True",
        ):
            test_round.process_payload(first_payload)

        assert test_round.participant_to_votes[first_payload.sender] == first_payload

        with pytest.raises(
            TransactionNotValidError,
            match="sender agent_0 has already sent its vote: True",
        ):
            test_round.check_payload(first_payload)

        assert test_round.end_block() is None
        assert not test_round.positive_vote_threshold_reached
        for payload in participant_to_votes_payloads.values():
            test_round.process_payload(payload)

        assert test_round.positive_vote_threshold_reached

        actual_next_state = self.period_state.update(
            participant_to_votes=MappingProxyType(
                dict(get_participant_to_votes(self.participants))
            )
        )
        res = test_round.end_block()
        assert res is not None
        state, event = res
        assert (
            cast(PeriodState, state).participant_to_votes.keys()
            == cast(PeriodState, actual_next_state).participant_to_votes.keys()
        )
        assert event == Event.DONE

    def test_negative_votes(
        self,
    ) -> None:
        """Test ValidateRound."""

        test_round = ValidateTransactionRound(
            state=self.period_state, consensus_params=self.consensus_params
        )

        participant_to_votes_payloads = get_participant_to_votes(
            self.participants, vote=False
        )
        first_payload = participant_to_votes_payloads.pop(
            sorted(list(participant_to_votes_payloads.keys()))[0]
        )
        test_round.process_payload(first_payload)

        assert test_round.participant_to_votes[first_payload.sender] == first_payload
        assert test_round.end_block() is None
        assert not test_round.negative_vote_threshold_reached
        for payload in participant_to_votes_payloads.values():
            test_round.process_payload(payload)

        assert test_round.negative_vote_threshold_reached

        res = test_round.end_block()
        assert res is not None
        state, event = res

        assert event == Event.EXIT_B
        with pytest.raises(
            AEAEnforceError, match="'participant_to_votes' field is None"
        ):
            _ = cast(PeriodState, state).participant_to_votes


def test_rotate_list_method() -> None:
    """Test `rotate_list` method."""

    ex_list = [1, 2, 3, 4, 5]
    assert rotate_list(ex_list, 2) == [3, 4, 5, 1, 2]


def test_period_state() -> None:
    """Test PeriodState."""

    participants = get_participants()
    participant_to_randomness = get_participant_to_randomness(participants, 1)
    most_voted_randomness = get_most_voted_randomness()
    participant_to_selection = get_participant_to_selection(participants)
    most_voted_keeper_address = get_most_voted_keeper_address()
    safe_contract_address = get_safe_contract_address()
    participant_to_votes = get_participant_to_votes(participants)
    participant_to_observations = get_participant_to_observations(participants)
    participant_to_estimate = get_participant_to_estimate(participants)
    estimate = get_estimate()
    most_voted_estimate = get_most_voted_estimate()
    participant_to_tx_hash = get_participant_to_tx_hash(participants)
    most_voted_tx_hash = get_most_voted_tx_hash()
    participant_to_signature = get_participant_to_signature(participants)
    final_tx_hash = get_final_tx_hash()

    period_state = PeriodState(
        participants=participants,
        participant_to_randomness=participant_to_randomness,
        most_voted_randomness=most_voted_randomness,
        participant_to_selection=participant_to_selection,
        most_voted_keeper_address=most_voted_keeper_address,
        safe_contract_address=safe_contract_address,
        participant_to_votes=participant_to_votes,
        participant_to_observations=participant_to_observations,
        participant_to_estimate=participant_to_estimate,
        estimate=estimate,
        most_voted_estimate=most_voted_estimate,
        participant_to_tx_hash=participant_to_tx_hash,
        most_voted_tx_hash=most_voted_tx_hash,
        participant_to_signature=participant_to_signature,
        final_tx_hash=final_tx_hash,
    )

    actual_keeper_randomness = float(
        (int(most_voted_randomness, base=16) // 10 ** 0 % 10) / 10
    )
    assert period_state.keeper_randomness == actual_keeper_randomness
    assert period_state.participant_to_randomness == participant_to_randomness
    assert period_state.most_voted_randomness == most_voted_randomness
    assert period_state.participant_to_selection == participant_to_selection
    assert period_state.most_voted_keeper_address == most_voted_keeper_address
    assert period_state.safe_contract_address == safe_contract_address
    assert period_state.participant_to_votes == participant_to_votes
    assert period_state.participant_to_observations == participant_to_observations
    assert period_state.participant_to_estimate == participant_to_estimate
    assert period_state.estimate == estimate
    assert period_state.most_voted_estimate == most_voted_estimate
    assert period_state.most_voted_tx_hash == most_voted_tx_hash
    assert period_state.participant_to_signature == participant_to_signature
    assert period_state.final_tx_hash == final_tx_hash

    assert period_state.encoded_most_voted_estimate == encode_float(most_voted_estimate)
