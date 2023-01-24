# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2021-2022 Valory AG
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

"""This module contains all the storing operations of the APY behaviour."""


from enum import Enum, auto
from io import BytesIO
from typing import Any, Callable, Dict, Optional, TypeVar, Union, cast

import joblib
import pandas as pd
from pmdarima.pipeline import Pipeline

from packages.valory.skills.abstract_round_abci.io_.store import (
    AbstractStorer,
    StoredJSONType,
)
from packages.valory.skills.abstract_round_abci.io_.store import Storer as BaseStorer


NativelySupportedSingleObjectType = Union[StoredJSONType, Pipeline, pd.DataFrame]
NativelySupportedMultipleObjectsType = Dict[str, NativelySupportedSingleObjectType]
NativelySupportedObjectType = Union[
    NativelySupportedSingleObjectType, NativelySupportedMultipleObjectsType
]
NativelySupportedStorerType = Callable[[str, NativelySupportedObjectType, Any], None]
CustomObjectType = TypeVar("CustomObjectType")
CustomStorerType = Callable[[str, CustomObjectType, Any], None]
SupportedSingleObjectType = Union[NativelySupportedObjectType, CustomObjectType]
SupportedMultipleObjectsType = Dict[str, SupportedSingleObjectType]
SupportedObjectType = Union[SupportedSingleObjectType, SupportedMultipleObjectsType]
SupportedStorerType = Union[NativelySupportedStorerType, CustomStorerType]
NativelySupportedJSONStorerType = Callable[
    [str, Union[StoredJSONType, Dict[str, StoredJSONType]], Any], None
]
NativelySupportedPipelineStorerType = Callable[
    [str, Union[Pipeline, Dict[str, Pipeline]], Any], None
]
NativelySupportedDfStorerType = Callable[
    [str, Union[pd.DataFrame, Dict[str, pd.DataFrame]], Any], None
]


class ExtendedSupportedFiletype(Enum):
    """Enum for the supported filetypes of the IPFS interacting methods."""

    PM_PIPELINE = auto()
    CSV = auto()


class CSVStorer(AbstractStorer):
    """A CSV file storer."""

    def serialize_object(
        self, filename: str, obj: NativelySupportedSingleObjectType, **kwargs: Any
    ) -> Dict[str, str]:
        """Store a pandas dataframe."""
        if not isinstance(obj, pd.DataFrame):
            raise ValueError(  # pragma: no cover
                f"`JSONStorer` cannot be used with a {type(obj)}! Only with a {pd.DataFrame}"
            )

        index = kwargs.get("index", False)

        try:
            return {filename: obj.to_csv(index=index)}
        except (TypeError, OSError) as e:  # pragma: no cover
            raise IOError(str(e)) from e


class ForecasterStorer(AbstractStorer):
    """A pmdarima Pipeline storer."""

    def serialize_object(
        self, filename: str, obj: NativelySupportedSingleObjectType, **kwargs: Any
    ) -> Dict[str, str]:
        """Store a pmdarima Pipeline."""
        if not isinstance(obj, Pipeline):
            raise ValueError(  # pragma: no cover
                f"`JSONStorer` cannot be used with a {type(obj)}! Only with a {Pipeline}"
            )

        bytes_container = BytesIO()
        try:
            joblib.dump(obj, bytes_container)
            # set the reference point at the beginning of the container
            bytes_container.seek(0)
            bytes_content = bytes_container.read()
            serialized = bytes_content.hex()
            return {filename: serialized}
        except (ValueError, OSError) as e:  # pragma: no cover
            raise IOError(str(e)) from e


class Storer(BaseStorer):
    """Class which stores files."""

    def __init__(
        self,
        filetype: Optional[ExtendedSupportedFiletype],
        custom_storer: Optional[CustomStorerType],
        path: str,
    ):
        """Initialize a `Storer`."""
        super().__init__(filetype, custom_storer, path)
        self._filetype_to_storer: Dict[Enum, SupportedStorerType]
        self._filetype_to_storer[ExtendedSupportedFiletype.PM_PIPELINE] = cast(
            NativelySupportedPipelineStorerType,
            ForecasterStorer(path).serialize_object,
        )
        self._filetype_to_storer[ExtendedSupportedFiletype.CSV] = cast(
            NativelySupportedDfStorerType, CSVStorer(path).serialize_object
        )
