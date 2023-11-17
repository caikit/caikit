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
from typing import Optional, Type, Union

# First Party
from py_to_proto.dataclass_to_proto import Annotated, OneofField
import aconfig
import alog

# Local
from caikit import get_config
from caikit.core.data_model import DataBase
from caikit.core.data_model.dataobject import make_dataobject
from caikit.core.exceptions import error_handler
from caikit.runtime import service_generation
import caikit

log = alog.use_channel("MDSV-PLUG")
error = error_handler.get(log)


def make_output_target_message(
    model_savers_config: Optional[aconfig.Config] = None,
) -> Type[DataBase]:
    """Dynamically create the output target message"""

    if not model_savers_config:
        model_savers_config = get_config().model_management.savers

    annotation_list = []

    field_number = 1
    for saver_name in model_savers_config:
        saver = caikit.core.MODEL_MANAGER.get_saver(saver_name)
        output_target_type = saver.output_target_type()
        annotation_list.append(
            Annotated[output_target_type, OneofField(saver_name), field_number]
        )
        field_number += 1

    output_target_type_union = Union[tuple(annotation_list)]

    data_object = make_dataobject(
        package=service_generation.get_runtime_service_package(),
        name="OutputTarget",
        annotations={"output_target": output_target_type_union},
    )

    return data_object
