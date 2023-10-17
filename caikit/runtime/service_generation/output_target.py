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
"""This file builds the data model for the `output_target` field, which contains
 all the output target types for any plugged-in model savers"""

# Standard
from typing import List, Optional, Type, Union
import abc
import typing

# First Party
from py_to_proto.dataclass_to_proto import Annotated, FieldNumber, OneofField
import aconfig
import alog

# Local
from caikit import get_config
from caikit.core.data_model import DataBase
from caikit.core.data_model.dataobject import make_dataobject
from caikit.core.exceptions import error_handler
from caikit.core.model_management import LocalPathModelSaver, ModelSaver
from caikit.core.toolkit.factory import FactoryConstructible, ImportableFactory
from caikit.interfaces.common.data_model.stream_sources import PathReference

log = alog.use_channel("MDSV-PLUG")
error = error_handler.get(log)


## Plugin Bases ################################################################


class ModelSaverPluginBase(FactoryConstructible):
    """An ModelSaverPlugin is a pluggable source that defines the shape of
    the data object that defines an output location, as well as the code for
    saving a trained model to that location.
    """

    def __init__(self, config: aconfig.Config, instance_name: str):
        """Construct with the basic factor constructible interface and store the
        args for use by the child
        """
        self._config = config
        self._instance_name = instance_name

    ## Abstract Interface ##

    @abc.abstractmethod
    def get_model_saver_class(self) -> Type[ModelSaver]:
        """Return the type of model saver built by this plugin"""

    @abc.abstractmethod
    def get_field_number(self) -> int:
        """Allow plugins to return a static field number"""

    @abc.abstractmethod
    def make_model_saver(self, target: DataBase) -> ModelSaver:
        """Given an output target, build a model saver"""

    ## Public Methods ##

    def get_field_name(self) -> str:
        """The name of the field that this plugin will use in the output target oneof"""
        return self.get_output_target_type().__name__.lower()

    # Default impls
    def get_output_target_type(self) -> Type[DataBase]:
        """Return the output target message expected by your model saver"""
        # Gorp here to get the generic type T of the target class
        # Could just be overriden directly by your plugin
        bases = self.get_model_saver_class().__orig_bases__
        output_target_base = [b for b in bases if typing.get_origin(b) == ModelSaver][0]
        return typing.get_args(output_target_base)[0]


## Target Plugins ##############################################################


class LocalModelSaverPlugin(ModelSaverPluginBase):
    """Plugin for a local model saver"""

    name = "Local"

    def get_model_saver_class(self) -> Type[ModelSaver]:
        return LocalPathModelSaver

    def make_model_saver(self, target: DataBase) -> ModelSaver:
        error.type_check("<RUN37386095E>", PathReference, target=target)
        target: PathReference
        save_with_id = self._config.get("save_with_id", None)
        return LocalPathModelSaver(target=target, save_with_id=save_with_id)

    def get_field_number(self) -> int:
        return 1


## ModelSaverPluginFactory ####################################################


class ModelSaverPluginFactory(ImportableFactory):
    """The ModelSaverPluginFactory is responsible for holding a registry of
    plugin instances that will be used to create and manage model savers
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._plugins = None

    def get_plugins(
        self, plugins_config: Optional[aconfig.Config] = None
    ) -> List[ModelSaverPluginBase]:
        """Builds the set of plugins to use for a model saver of type element_type"""
        if self._plugins is None:
            self._plugins = []
            if plugins_config is None:
                plugins_config = get_config().runtime.training.model_saver_plugins
            for name, cfg in plugins_config.items():
                self._plugins.append(self.construct(cfg, name))

            # Make sure field numbers are unique
            field_numbers = [plugin.get_field_number() for plugin in self._plugins]
            duplicates_field_numbers = [
                plugin.name
                for plugin in self._plugins
                if field_numbers.count(plugin.get_field_number()) > 1
            ]
            error.value_check(
                "<RUN13216546E>",
                not duplicates_field_numbers,
                "Duplicate plugin field numbers found for plugins: {}",
                duplicates_field_numbers,
            )
        return self._plugins


# Single default instance
PluginFactory = ModelSaverPluginFactory("ModelSaver")
PluginFactory.register(LocalModelSaverPlugin)

## OutputTargetOneOf
class OutputTargetOneOf:
    """Class that implements output target api for runtime requests"""

    # TODO: memoize?
    @property
    def name_to_plugin_map(self):
        if not hasattr(self, "_name_to_plugin_map"):
            self._name_to_plugin_map = {
                plugin.get_field_name(): plugin for plugin in self.PLUGINS
            }
        return self._name_to_plugin_map

    def get_model_saver(self):
        """Find plugin for field and make model saver"""
        # Determine which of the names is set
        set_field = None
        for field_name in self.get_proto_class().DESCRIPTOR.fields_by_name:
            if getattr(self, field_name) is not None:
                error.value_check(
                    "<RUN40182782E>",
                    set_field is None,
                    "Found OutputTargetOneOf with multiple sources set: {} and {}",
                    set_field,
                    field_name,
                )
                error.value_check(
                    "<RUN48659820E>",
                    field_name in self.name_to_plugin_map,
                    "No model saver plugin found for field: {}",
                    field_name,
                )
                set_field = field_name

        # If no field is set - default to no model saver
        # Could consider default to one place e.g. one high level training_output_dir
        if set_field is None:
            log.error("<RUN00550784E>", "No model saver set")
            return None

        plugin = self.name_to_plugin_map[set_field]
        return plugin.make_model_saver(getattr(self, set_field))


## make_output_target_message #####################################################

# TODO: Ask Scott or Dean about correct multiple inheritance type hinting
# OutputTargetAndDataBase = typing.TypeVar("OutputTargetAndDataBase", bound=Union[DataBase, OutputTargetOneOf])
OutputTargetDataModel = typing.TypeVar("OutputTargetDataModel", DataBase, OutputTargetOneOf)


def make_output_target_message(
    plugin_factory: ModelSaverPluginFactory = PluginFactory,
    plugins_config: Optional[aconfig.Config] = None,
) -> Type[OutputTargetDataModel]:
    """Dynamically create the output target message"""

    # Get the required plugins
    plugins = plugin_factory.get_plugins(plugins_config)

    # Make sure there are no field name duplicates
    plug_to_name = {plugin: plugin.get_field_name() for plugin in plugins}
    all_field_names = list(plug_to_name.values())
    duplicates = {
        plugin.name: field_name
        for plugin, field_name in plug_to_name.items()
        if all_field_names.count(field_name) > 1
    }
    error.value_check(
        "<RUN40793078E>",
        not duplicates,
        "Duplicate plugin field names found for output_target: {}",
        duplicates,
    )

    annotation_list = [
        Annotated[
            plugin.get_output_target_type(),
            OneofField(plugin.get_field_name()),
            FieldNumber(plugin.get_field_number()),
        ]
        for plugin in plugins
    ]

    output_target_type_union = Union[tuple(annotation_list)]

    data_object = make_dataobject(
        package="some_package",
        name="OutputTarget",
        bases=(OutputTargetOneOf,),
        attrs={"PLUGINS": plugins},
        annotations={"output_target": output_target_type_union},
    )

    return data_object
