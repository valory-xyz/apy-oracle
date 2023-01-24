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

"""IO operations for the APY skill."""

from io import StringIO

import pandas as pd
from pandas._libs.tslibs.np_datetime import (  # type: ignore # pylint: disable=E0611
    OutOfBoundsDatetime,
)

from packages.valory.skills.apy_estimation_abci.tools.etl import TRANSFORMED_HIST_DTYPES


def load_hist(serialized_hist: str) -> pd.DataFrame:
    """Load the already fetched and transformed historical data.

    :param serialized_hist: the historical data, serialized to a csv string.
    :return: a dataframe with the historical data.
    """
    try:
        buffer = StringIO(serialized_hist)
        pairs_hist = pd.read_csv(buffer).astype(TRANSFORMED_HIST_DTYPES)

        # Convert the `blockTimestamp` to a pandas datetime.
        pairs_hist["blockTimestamp"] = pd.to_datetime(
            pairs_hist["blockTimestamp"], unit="s"
        )
    except (OutOfBoundsDatetime, KeyError) as e:
        raise IOError("The provided csv is not well formatted!") from e
    except pd.errors.EmptyDataError as e:
        raise IOError("The provided csv was empty!") from e

    return pairs_hist
