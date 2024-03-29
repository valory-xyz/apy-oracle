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

"""This module contains all the loading operations of the APY behaviour."""

from io import BytesIO, StringIO
from typing import Callable, Dict, Optional

import joblib
import pandas as pd

from packages.valory.skills.abstract_round_abci.io_.load import AbstractLoader
from packages.valory.skills.abstract_round_abci.io_.load import Loader as BaseLoader
from packages.valory.skills.apy_estimation_abci.io_.store import (
    CustomObjectType,
    ExtendedSupportedFiletype,
    NativelySupportedSingleObjectType,
    SupportedSingleObjectType,
)


CustomLoaderType = Optional[Callable[[str], CustomObjectType]]
SupportedLoaderType = Callable[[str], SupportedSingleObjectType]


class CSVLoader(AbstractLoader):
    """A csv files Loader."""

    def load_single_object(
        self, serialized_object: str
    ) -> NativelySupportedSingleObjectType:
        """Read a pandas dataframe from a csv object.

        :param serialized_object: the object serialized into a CSV string.
        :return: the pandas dataframe.
        """
        buffer = StringIO(serialized_object)

        try:
            return pd.read_csv(buffer)
        except pd.errors.EmptyDataError as e:  # pragma: no cover
            raise IOError("The provided csv was empty!") from e


class ForecasterLoader(AbstractLoader):
    """A `pmdarima` forecaster loader."""

    def load_single_object(
        self, serialized_object: str
    ) -> NativelySupportedSingleObjectType:
        """Load a `pmdarima` forecaster.

        :param serialized_object: the object serialized into a hex string of a `joblib`'s bytes dump.
        :return: a `pmdarima.pipeline.Pipeline`.
        """
        bytes_content = bytes.fromhex(serialized_object)
        bytes_container = BytesIO(bytes_content)

        try:
            return joblib.load(bytes_container)
        except (ValueError, UnicodeDecodeError) as e:  # pragma: no cover
            raise IOError(
                f"The serialized object {serialized_object} cannot be loaded: {e}!"
            ) from e


class Loader(BaseLoader):
    """Class which loads files."""

    def __init__(
        self,
        filetype: Optional[ExtendedSupportedFiletype],
        custom_loader: CustomLoaderType,
    ):
        """Initialize a `Loader`."""
        super().__init__(filetype, custom_loader)

        self.__filetype_to_loader: Dict[ExtendedSupportedFiletype, SupportedLoaderType]
        self.__filetype_to_loader[
            ExtendedSupportedFiletype.PM_PIPELINE
        ] = ForecasterLoader().load_single_object
        self.__filetype_to_loader[
            ExtendedSupportedFiletype.CSV
        ] = CSVLoader().load_single_object
