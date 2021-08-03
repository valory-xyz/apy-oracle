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
"""Decode AEA messages from Tendermint protobuf messages."""
from typing import Callable, Optional, Union

from packages.valory.connections.abci.tendermint.abci.types_pb2 import Request, Response
from packages.valory.connections.abci.tendermint.crypto.proof_pb2 import Proof
from packages.valory.protocols.abci import AbciMessage
from packages.valory.protocols.abci.custom_types import (
    BlockID,
    BlockParams,
    CheckTxType,
    CheckTxTypeEnum,
    ConsensusParams,
    Duration,
    Evidence,
    EvidenceParams,
    Evidences,
    Header,
    LastCommitInfo,
    PartSetHeader,
    Timestamp,
    Validator,
    ValidatorParams,
    ValidatorUpdates,
    VersionParams,
    VoteInfo,
)


class _TendermintProtocolDecoder:
    """
    Decoder called by the server to process requests from the TCP connection with Tendermint.

    It translates from Tendermint's ABCI Protobuf messages into the AEA's ABCI protocol messages.
    """

    @classmethod
    def process(
        cls, message_type: str, message: Union[Request, Response]
    ) -> Optional[AbciMessage]:
        """Process an ABCI request or response."""
        message_type = (
            f"request_{message_type}"
            if isinstance(message, Request)
            else f"response_{message_type}"
        )
        handler: Callable[[Request], AbciMessage] = getattr(
            cls, message_type, cls.no_match
        )
        return handler(message)

    @classmethod
    def request_flush(cls, _request: Request) -> AbciMessage:
        """
        Decode a flush request.

        :param _request: the request.
        :return: the AbciMessage request.
        """
        return AbciMessage(
            performative=AbciMessage.Performative.REQUEST_FLUSH,
        )

    @classmethod
    def request_info(cls, request: Request) -> AbciMessage:
        """
        Decode a info request.

        :param request: the request.
        :return: the AbciMessage request.
        """
        info = request.info
        return AbciMessage(
            performative=AbciMessage.Performative.REQUEST_INFO,
            version=info.version,
            block_version=info.block_version,
            p2p_version=info.p2p_version,
        )

    @classmethod
    def request_init_chain(cls, request: Request) -> AbciMessage:
        """
        Decode a init_chain request.

        :param request: the request.
        :return: the AbciMessage request.
        """
        init_chain = request.init_chain
        timestamp = cls._decode_timestamp(init_chain.time)
        chain_id = init_chain.chain_id
        consensus_params = (
            cls._decode_consensus_params(init_chain.consensus_params)
            if init_chain.consensus_params is not None
            else None
        )
        validators = ValidatorUpdates(init_chain.validators)
        app_state_bytes = init_chain.app_state_bytes
        initial_height = init_chain.initial_height

        result = AbciMessage(
            performative=AbciMessage.Performative.REQUEST_INIT_CHAIN,
            time=timestamp,
            chain_id=chain_id,
            validators=validators,
            app_state_bytes=app_state_bytes,
            initial_height=initial_height,
        )
        if consensus_params is not None:
            result.set("consensus_params", consensus_params)
        return result

    @classmethod
    def request_begin_block(cls, request: Request) -> AbciMessage:
        """
        Decode a begin_block request.

        :param request: the request.
        :return: the AbciMessage request.
        """
        begin_block = request.begin_block
        hash_ = begin_block.hash
        header = cls._decode_header(begin_block.header)
        last_commit_info = cls._decode_last_commit_info(begin_block.last_commit_info)
        evidences = [
            cls._decode_evidence(byzantine_validator)
            for byzantine_validator in list(begin_block.byzantine_validators)
        ]
        return AbciMessage(
            performative=AbciMessage.Performative.REQUEST_BEGIN_BLOCK,
            hash=hash_,
            header=header,
            last_commit_info=last_commit_info,
            byzantine_validators=Evidences(evidences),
        )

    @classmethod
    def request_check_tx(cls, request: Request) -> AbciMessage:
        """
        Decode a check_tx request.

        :param request: the request.
        :return: the AbciMessage request.
        """
        check_tx = request.check_tx
        tx = check_tx.tx
        check_tx_type = CheckTxType(CheckTxTypeEnum(check_tx.type))
        return AbciMessage(
            performative=AbciMessage.Performative.REQUEST_CHECK_TX,
            tx=tx,
            type=check_tx_type,
        )

    @classmethod
    def request_deliver_tx(cls, request: Request) -> AbciMessage:
        """
        Decode a deliver_tx request.

        :param request: the request.
        :return: the AbciMessage request.
        """
        return AbciMessage(
            performative=AbciMessage.Performative.REQUEST_DELIVER_TX,
            tx=request.deliver_tx.tx,
        )

    @classmethod
    def request_query(cls, request: Request) -> AbciMessage:
        """
        Decode a query request.

        :param request: the request.
        :return: the AbciMessage request.
        """
        return AbciMessage(
            performative=AbciMessage.Performative.REQUEST_QUERY,
            query_data=request.query.data,
            path=request.query.path,
            height=request.query.height,
            prove=request.query.prove,
        )

    @classmethod
    def request_commit(cls, _request: Request) -> AbciMessage:
        """
        Decode a commit request.

        :param _request: the request.
        :return: the AbciMessage request.
        """
        return AbciMessage(performative=AbciMessage.Performative.REQUEST_COMMIT)

    @classmethod
    def request_end_block(cls, request: Request) -> AbciMessage:
        """
        Decode an end_block request.

        :param request: the request.
        :return: the AbciMessage request.
        """
        return AbciMessage(
            performative=AbciMessage.Performative.REQUEST_END_BLOCK,
            height=request.end_block.height,
        )

    @classmethod
    def request_list_snapshots(cls, request: Request) -> AbciMessage:
        raise NotImplementedError

    @classmethod
    def request_offer_snapshot(cls, request: Request) -> AbciMessage:
        raise NotImplementedError

    @classmethod
    def request_load_snapshot_chunk(cls, request: Request) -> AbciMessage:
        raise NotImplementedError

    @classmethod
    def request_apply_snapshot_chunk(cls, request: Request) -> AbciMessage:
        raise NotImplementedError

    @classmethod
    def no_match(cls, _request: Request) -> None:
        return None

    @classmethod
    def _decode_timestamp(cls, timestamp_tendermint_pb) -> Timestamp:
        """Decode a timestamp object."""
        return Timestamp(
            timestamp_tendermint_pb.seconds,
            timestamp_tendermint_pb.nanos,
        )

    @classmethod
    def _decode_consensus_params(
        cls, consensus_params_tendermint_pb
    ) -> ConsensusParams:
        """Decode a ConsensusParams object."""
        return ConsensusParams.decode(consensus_params_tendermint_pb)

    @classmethod
    def _decode_header(cls, header_tendermint_pb) -> Header:
        """Decode a Header object."""
        return Header.decode(header_tendermint_pb)

    @classmethod
    def _decode_block_id(cls, block_id_pb) -> BlockID:
        """Decode a Block ID object."""
        part_set_header_pb = block_id_pb.part_set_header
        part_set_header = PartSetHeader(
            part_set_header_pb.index, part_set_header_pb.bytes
        )
        return BlockID(block_id_pb.hash, part_set_header)

    @classmethod
    def _decode_last_commit_info(cls, last_commit_info_tendermint_pb) -> LastCommitInfo:
        """Decode a LastCommitInfo object."""
        return LastCommitInfo.decode(last_commit_info_tendermint_pb)

    @classmethod
    def _decode_proof(cls, proof_pb) -> Proof:
        """Decode a Proof object."""
        return Proof(
            proof_pb.total,
            proof_pb.index,
            proof_pb.leaf_hash,
            proof_pb.aunts,
        )

    @classmethod
    def _decode_vote_info(cls, vote_pb) -> VoteInfo:
        """Decode a VoteInfo object."""
        validator = cls._decode_validator(vote_pb.validator)
        signed_last_block = vote_pb.signed_last_block
        return VoteInfo(validator, signed_last_block)

    @classmethod
    def _decode_validator(cls, validator_pb) -> Validator:
        """Decode a Validator object."""
        return Validator(validator_pb.address, validator_pb.power)

    @classmethod
    def _decode_evidence(cls, evidence_pb):
        """Decode an Evidence object."""
        return Evidence.decode(evidence_pb)

    @classmethod
    def _decode_block_params(cls, block_params_tendermint_pb) -> BlockParams:
        return BlockParams(
            block_params_tendermint_pb.max_bytes, block_params_tendermint_pb.max_gas
        )

    @classmethod
    def _decode_evidence_params(cls, evidence_params_tendermint_pb) -> EvidenceParams:
        duration = Duration.decode(evidence_params_tendermint_pb.max_age_duration)
        return EvidenceParams(
            evidence_params_tendermint_pb.max_age_num_blocks,
            duration,
            evidence_params_tendermint_pb.max_bytes,
        )

    @classmethod
    def _decode_validator_params(
        cls, validator_params_tendermint_pb
    ) -> ValidatorParams:
        pub_key_types = list(validator_params_tendermint_pb.pub_key_types)
        return ValidatorParams(
            pub_key_types,
        )

    @classmethod
    def _decode_version_params(cls, version_params_tendermint_pb) -> VersionParams:
        pass
