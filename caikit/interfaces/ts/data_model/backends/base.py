# Copyright The Caikit Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Base classes to share between data model backends
"""

# Standard
from typing import Any, Iterable, Tuple, Type
import abc

# Third Party
import pandas as pd

# First Party
import alog

# Local
from caikit.core.data_model.data_backends import DataModelBackendBase
from caikit.core.exceptions import error_handler

log = alog.use_channel("DMBCK")
error = error_handler.get(log)


class UncachedBackendMixin(DataModelBackendBase):
    """Intermediate base class that disables attribute caching"""

    def cache_attribute(self, *_, **__) -> bool:
        """Never cache attributes"""
        return False


class StrictFieldBackendMixin(DataModelBackendBase):
    """Intermediate base class that raises attribute errors for unknown fields"""

    def get_attribute(self, data_model_class: Type, name: str) -> Any:
        """Base implementation that raises an AttributeError on bad attr names.
        It should be called after object-specific logic.
        """
        if name not in data_model_class.fields:
            error(
                "<COR81128387E>",
                AttributeError(
                    f"No such attribute [{name}] on [{data_model_class.__name__}]"
                ),
            )


class TimeSeriesBackendBase(UncachedBackendMixin, StrictFieldBackendMixin):
    """Abstract base class for all backends of the central TimeSeries data model
    type
    """

    @abc.abstractmethod
    def as_pandas(self) -> Tuple[pd.DataFrame, str, Iterable[str]]:
        """All backends must implement the ability to coerce their underlying
        data into a pandas DataFrame and provide the pointers to the timeseries
        source and value source(s)

        Returns:
            df:  pd.DataFrame
                The data frame itself
            timestamp_source:  str
                The column name (or None) indicating where the
                timestamp sequence can be found
            value_source:  Iterable[str]
                The names of the columns holding value sequences
        """


class MultiTimeSeriesBackendBase(UncachedBackendMixin, StrictFieldBackendMixin):
    """Abstract base class for all backends of the central MultiTimeSeries data model
    type
    """

    @abc.abstractmethod
    def as_pandas(self) -> Tuple[pd.DataFrame, Iterable[str], str, Iterable[str]]:
        """All backends must implement the ability to coerce their underlying
        data into a pandas DataFrame and provide the pointers to the timeseries
        source and value source(s)

        Returns:
            df:  pd.DataFrame
                The data frame itself
            key_source: Iterable[str]
                the names of the columns holding key values
            timestamp_source:  str
                The column name (or None) indicating where the
                timestamp sequence can be found
            value_source:  Iterable[str]
                The names of the columns holding value sequences
        """
