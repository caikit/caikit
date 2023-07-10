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


"""The base class for data model object backends"""

# Standard
from typing import Any, Type, Union
import abc

# Local
from ..base import DataBase

# DataModelBackendBase #########################################################


class DataModelBackendBase(abc.ABC):
    """A base interface class for accessing data from within a given backend
    data layout
    """

    @abc.abstractmethod
    def get_attribute(
        self,
        data_model_class: Type[DataBase],
        name: str,
    ) -> Union[Any, DataBase.OneofFieldVal]:
        """A data model backend must implement this in order to provide the
        frontend view the functionality needed to lazily extract data.

        Args:
            data_model_class (Type[DataBase]): The frontend data model class
                that is accessing this attribute
            name (str): The name of the attribute to access

        Returns:
            value:  Union[Any, OneofFieldVal]
                The extracted attribute value or a OneofFieldVal that wraps the
                field val with an indicator about the oneof field that is set.
        """

    # pylint: disable=unused-argument
    def cache_attribute(self, name: str, value: Any) -> bool:
        """Determine whether or not to cache the given attribute's result on the
        wrapping data model object.

        The base implementation always returns True. Derived classes may opt to
        always return False to fully disable caching, or cache conditionally
        based on the name/value of the individual field.

        Args:
            name (str): The name of the attribute to check
            value (Any): The extracted value

        Returns:
            should_cache:  bool
                True if the value should be cached, False otherwise
        """
        return True
