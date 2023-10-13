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
import abc
import typing
from typing import Type, Optional, List

import aconfig
import alog

from caikit import get_config
from caikit.core.exceptions import error_handler

from caikit.core.model_management import ModelSaver, LocalFileModelSaver
from caikit.core.data_model import DataBase
from caikit.core.toolkit.factory import FactoryConstructible, ImportableFactory
from caikit.interfaces.common.data_model.stream_sources import File

log = alog.use_channel("DSTRM-SRC")
error = error_handler.get(log)


## Plugin Bases ################################################################

class ModelSaverPluginBase(FactoryConstructible):
    """An OutputTargetPlugin is a pluggable source that defines the shape of
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
        """Return the type of model saver built by this plugin
        """

    @abc.abstractmethod
    def get_field_number(self) -> int:
        """Allow plugins to return a static field number"""
        pass

    @abc.abstractmethod
    def make_model_saver(self, target: Type[DataBase]) -> ModelSaver:
        """Given an output target, build a model saver"""
        pass


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

    def get_model_saver_class(self) -> Type[ModelSaver]:
        return LocalFileModelSaver

    def make_model_saver(self, target: Type[DataBase]) -> ModelSaver:
        error.type_check("<COR13362169E>", File, target=target)
        target: File
        save_with_id = self._config.get("save_with_id", None)
        return LocalFileModelSaver(target=target, save_with_id=save_with_id)

    def get_field_number(self) -> int:
        return 1


## OutputTargetRegistry ####################################################


class ModelSaverPluginFactory(ImportableFactory):
    """The DataStreamSourceRegistry is responsible for holding a registry of
    plugin instances that will be used to create and manage data stream sources
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._plugins = None

    def get_plugins(
        self, plugins_config: Optional[aconfig.Config] = None
    ) -> List[ModelSaverPluginBase]:
        """Builds the set of plugins to use for a data stream source of type element_type"""
        if self._plugins is None:
            self._plugins = []
            if plugins_config is None:
                # TODO: where to configure
                plugins_config = get_config().data_streams.source_plugins
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
                "<COR69189361E>",
                not duplicates_field_numbers,
                "Duplicate plugin field numbers found for plugins: {}",
                duplicates_field_numbers,
            )
        return self._plugins


# Single default instance
PluginFactory = ModelSaverPluginFactory("ModelSaver")
PluginFactory.register(LocalModelSaverPlugin)




def make_output_target_message() -> typing.Type[DataBase]:
    """Do the magic!"""

    plugins = [LocalFileOutputPlugin()]

    annotation_list = [
        typing.Annotated[
            plugin.get_message_type(),
            OneofField(plugin.get_field_name()),
            FieldNumber(plugin.get_field_number()),
        ]
        for plugin in plugins
    ]
    # data_stream_type_union = typing.Union[tuple(annotation_list)]
    # You need more than one thing in the union...
    data_stream_type_union = typing.Union[typing.Annotated[File, OneofField("file"), 1], typing.Annotated[File, OneofField("filezz"), 2]]

    data_object = make_dataobject(
        package="some_package",
        name="OutputTarget",
        bases=(OutputTargetOneOfThing,),
        attrs={"PLUGINS": plugins},
        annotations={"output_target": data_stream_type_union},
    )

    return data_object