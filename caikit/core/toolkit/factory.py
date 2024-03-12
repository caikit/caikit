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
This toolkit utility provides common factory construction semantics and a common
base class for classes that can be constructed via caikit config
"""

# Standard
from typing import Optional, Type
import abc
import importlib

# First Party
import aconfig
import alog

# Local
from ..exceptions import error_handler

log = alog.use_channel("FCTRY")
error = error_handler.get(log)


class FactoryConstructible(abc.ABC):
    """A class can be constructed by a factory if its constructor takes exactly
    one argument that is an aconfig.Config object and it has a name to identify
    itself with the factory.
    """

    @property
    @classmethod
    @abc.abstractmethod
    def name(cls) -> str:
        """This is the name of this constructible type that will be used by
        the factory to identify this class
        """

    @abc.abstractmethod
    def __init__(self, config: aconfig.Config, instance_name: str):
        """A FactoryConstructible object must be constructed with a config
        object that it uses to pull in all configuration
        """


class Factory:
    """The base factory class implements all common factory functionality to
    read a designated portion of config and instantiate an instance of the
    registered classes.
    """

    # The keys in the instance config
    TYPE_KEY = "type"
    CONFIG_KEY = "config"

    def __init__(self, name: str):
        """Construct with the path in the global config where this factory's
        configuration lives.
        """
        self._name = name
        self._registered_types = {}

    @property
    def name(self) -> str:
        return self._name

    def register(self, constructible: Type[FactoryConstructible]):
        """Register the given constructible"""
        current = self._registered_types.get(constructible.name)
        error.value_check(
            "<COR27473932E>",
            current is None or current is constructible,
            "Conflicting registration of {}",
            constructible.name,
        )
        self._registered_types[constructible.name] = constructible

    def construct(
        self,
        instance_config: dict,
        instance_name: Optional[str] = None,
    ) -> FactoryConstructible:
        """Construct an instance of the given type"""
        inst_type = instance_config.get(self.__class__.TYPE_KEY)
        inst_cfg = aconfig.Config(
            instance_config.get(self.__class__.CONFIG_KEY, {}),
            override_env_vars=False,
        )
        inst_cls = self._registered_types.get(inst_type)
        error.value_check(
            "<COR41162423E>",
            inst_cls is not None,
            "No {} class registered for type {}",
            self.name,
            inst_type,
        )
        instance_name = instance_name or inst_cls.name
        return inst_cls(inst_cfg, instance_name)


class ImportableFactory(Factory):
    """An ImportableFactory extends the base Factory to allow the construction
    to specify an "import_class" field that will be used to import and register
    the implementation class before attempting to initialize it.
    """

    IMPORT_CLASS_KEY = "import_class"

    def construct(
        self,
        instance_config: dict,
        instance_name: Optional[str] = None,
    ):
        # Look for an import_class and import and register it if found
        import_class_val = instance_config.get(self.__class__.IMPORT_CLASS_KEY)
        if import_class_val:
            error.type_check(
                "<COR85108801E>",
                str,
                **{self.__class__.IMPORT_CLASS_KEY: import_class_val},
            )
            module_name, class_name = import_class_val.rsplit(".", 1)
            try:
                imported_module = importlib.import_module(module_name)
            except ImportError:
                error(
                    "<COR46837141E>",
                    ValueError(
                        "Invalid {}: Module cannot be imported [{}]".format(
                            self.__class__.IMPORT_CLASS_KEY,
                            module_name,
                        )
                    ),
                )
            try:
                imported_class = getattr(imported_module, class_name)
            except AttributeError:
                error(
                    "<COR46837142E>",
                    ValueError(
                        "Invalid {}: No such class [{}] on module [{}]".format(
                            self.__class__.IMPORT_CLASS_KEY,
                            class_name,
                            module_name,
                        )
                    ),
                )
            error.subclass_check("<COR52306423E>", imported_class, FactoryConstructible)

            self.register(imported_class)
        return super().construct(instance_config, instance_name)
