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
"""Utilities related to manageing spark DataFrame caching"""

# Third Party
from pyspark.sql import DataFrame


class EnsureCached:
    """Will ensure that a given dataframe is cached.
    If dataframe is already cached it does nothing. If it's not
    cached, it will cache it and then uncache the object when
    the EnsureCached object container goes out of scope. Users
    must utilize the with pattern of access.

    Example:
    ```python
        with EnsureCached(df) as _:
            # do dataframey sorts of things on df
            # it's guarenteed to be cached
            # inside this block
        # that's it, you're done.
        # df remains cached if it already was
        # or it's no longer cached if it wasn't
        # before entering the with block above.
    ```
    """

    def __init__(self, dataframe: DataFrame):
        self._did_cache = False
        self._df = dataframe
        if hasattr(dataframe, "cache") and not self._df.is_cached:
            self._df.cache()
            self._did_cache = True

    def __enter__(self):
        return self._df

    def __exit__(self, exc_type, exc_value, traceback):
        if self._did_cache:
            self._df.unpersist()
