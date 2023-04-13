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

# Local
from caikit.runtime.service_generation.create_service import (
    create_inference_rpcs,
    create_training_rpcs,
)
import sample_lib

## Setup ########################################################################

widget_class = sample_lib.blocks.sample_task.SampleBlock
untrainable_module_class = sample_lib.blocks.sample_task.SamplePrimitiveBlock

## Tests ########################################################################

### create_inference_rpcs tests #################################################


def test_create_inference_rpcs():
    rpcs = create_inference_rpcs([widget_class])
    assert len(rpcs) == 1
    assert widget_class in rpcs[0].module_list


def test_create_inference_rpcs_for_multiple_modules_of_same_type():
    module_list = [
        sample_lib.blocks.sample_task.SampleBlock,
        sample_lib.blocks.sample_task.SamplePrimitiveBlock,
        sample_lib.blocks.other_task.OtherBlock,
    ]
    rpcs = create_inference_rpcs(module_list)

    # only 2 RPCs, Widget and Gadget because SampleWidget and AnotherWidget are of the same module type Widget
    assert len(rpcs) == 2
    assert sample_lib.blocks.sample_task.SampleBlock in rpcs[0].module_list
    assert sample_lib.blocks.sample_task.SamplePrimitiveBlock in rpcs[0].module_list
    assert sample_lib.blocks.other_task.OtherBlock in rpcs[1].module_list


def test_create_inference_rpcs_with_block_and_workflow():
    module_list = [
        sample_lib.blocks.sample_task.SampleBlock,
        sample_lib.workflows.sample_task.SampleWorkflow,
    ]
    rpcs = create_inference_rpcs(module_list)
    # only 1 RPC
    assert len(rpcs) == 1
    assert sample_lib.blocks.sample_task.SampleBlock in rpcs[0].module_list
    assert sample_lib.workflows.sample_task.SampleWorkflow in rpcs[0].module_list


def test_create_inference_rpcs_remove_non_primitive_modules():
    # NOTE: This requires Gadget to be in config since other modules do not have methods - TODO fix?
    module_list = [
        sample_lib.blocks.sample_task.SampleBlock,  # is a primitive module
        sample_lib.blocks.sample_task.InnerBlock,  # not a primitive module
    ]
    rpcs = create_inference_rpcs(module_list)

    # only 1 RPC, fidget is not valid
    assert len(rpcs) == 1
    assert sample_lib.blocks.sample_task.SampleBlock in rpcs[0].module_list


### create_training_rpcs tests #################################################


def test_create_training_rpcs():
    rpcs = create_training_rpcs([widget_class])
    assert len(rpcs) == 1
    assert widget_class in rpcs[0].module_list


def test_create_training_rpcs_without_train():
    rpcs = create_training_rpcs([untrainable_module_class])
    assert not rpcs
