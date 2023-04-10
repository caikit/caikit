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
This test suite ensures that any model metadata (things inside a config.yml file) is persisted when
a module is loaded and then re-saved
"""

# Standard
from typing import Any, Dict, Set
import os
import tempfile

# Third Party
# pylint: disable=import-error
from sample_lib.blocks.sample_task import SampleBlock

# pylint: disable=import-error
from sample_lib.workflows.sample_task import SampleWorkflow
import pytest

# Local
from caikit.core import toolkit

# Unit Test Infrastructure
from tests.base import TestCaseBase
import caikit.core

# scratch:
# - might want to support like a `saved` date?
# - want a unique id
# - do we want to easily support editing metadata in the case that somebody:
#   - loads a model
#   - changes the python model object
#   - re-saves the model?
# - What about the `version` field? It's unused now but if it does change should we allow it to
# re-save with a new version?


def _load_model_metadata(model_path: str) -> Dict[str, Any]:
    return toolkit.load_yaml(os.path.join(model_path, "config.yml"))


def _check_dicts_equal(
    this: Dict[str, Any], that: Dict[str, Any], fields_to_not_check: Set[str]
):
    these_fields = set(this.keys()) - fields_to_not_check
    those_fields = set(that.keys()) - fields_to_not_check
    assert these_fields == those_fields

    for field in these_fields:
        assert this.get(field) == that.get(field)


@pytest.fixture
def fixtures_dir():
    return TestCaseBase.fixtures_dir


@pytest.fixture
# pylint: disable=redefined-outer-name
def sample_model_path(fixtures_dir):
    return os.path.join(fixtures_dir, "sample_block")


# pylint: disable=redefined-outer-name
def test_block_metadata_is_persisted(sample_model_path):
    """Make sure that if you load and then re-save any model, the metadata in the config.yml is
    persisted."""
    # Get the block metadata:
    initial_metadata = _load_model_metadata(sample_model_path)
    model = caikit.core.load(sample_model_path)

    with tempfile.TemporaryDirectory() as tempdir:
        model.save(tempdir)
        resaved_metadata = _load_model_metadata(tempdir)

    fields_to_not_check = {"saved", "sample_lib_version", "train"}
    _check_dicts_equal(initial_metadata, resaved_metadata, fields_to_not_check)

    # Fun little extra check, the `train` field _should_ be modified by the code
    assert "optim_method" not in resaved_metadata["train"]


# pylint: disable=redefined-outer-name
def test_loaded_modules_have_metadata(sample_model_path):
    """Make sure a model has metadata after being loaded"""
    expected_metadata = _load_model_metadata(sample_model_path)
    model_loaded_with_core = caikit.core.load(sample_model_path)
    model_directly_loaded = SampleBlock.load(sample_model_path)

    # TODO: figure out "module_id" and "model_path" as well...

    _check_dicts_equal(
        expected_metadata,
        model_loaded_with_core.metadata,
        {"module_id", "model_path"},
    )
    _check_dicts_equal(
        expected_metadata,
        model_directly_loaded.metadata,
        {"module_id", "model_path"},
    )


def test_block_has_saved_field():
    """Make sure that if you load a model and then save it multiple times,
    the "saved" field should track each timestamp when you save, and should be different
    """
    with tempfile.TemporaryDirectory() as tempdir:
        model1 = SampleBlock()
        path1 = os.path.join(tempdir, "test1")
        model1.save(path1)
        resaved_metadata1 = _load_model_metadata(path1)

        model2 = caikit.core.load(path1)
        path2 = os.path.join(tempdir, "test1")
        model2.save(path2)
        resaved_metadata2 = _load_model_metadata(path2)

    assert "saved" in resaved_metadata1
    assert "saved" in resaved_metadata2
    assert resaved_metadata1["saved"] != resaved_metadata2["saved"]


def test_block_has_tracking_id_field():
    with tempfile.TemporaryDirectory() as tempdir:
        model1 = SampleBlock()
        path1 = os.path.join(tempdir, "test1")
        model1.save(path1)
        resaved_metadata1 = _load_model_metadata(path1)

        model2 = caikit.core.load(path1)
        path2 = os.path.join(tempdir, "test1")
        model2.save(path2)
        resaved_metadata2 = _load_model_metadata(path2)

    assert "tracking_id" in resaved_metadata1
    assert "tracking_id" in resaved_metadata2
    assert resaved_metadata1["tracking_id"] == resaved_metadata2["tracking_id"]


# pylint: disable=redefined-outer-name
def test_block_metadata_is_saved_into_a_workflow(sample_model_path):
    """Make sure that if you load a model and then re-package it as part of yet another model
    (in this case a block is packaged inside a workflow), the metadata of the original model is
    persisted"""
    # Get the block metadata:
    initial_metadata = _load_model_metadata(sample_model_path)
    model = caikit.core.load(sample_model_path)

    # Create a new workflow containing our block
    workflow = SampleWorkflow(model)

    with tempfile.TemporaryDirectory() as tempdir:
        workflow.save(tempdir)
        # The block's .yaml should live under workflow_root/dummy_model/config.yml
        resaved_metadata = _load_model_metadata(os.path.join(tempdir, "dummy_model"))

    # assert resaved_metadata == initial_metadata
    fields_to_not_check = {"saved", "SampleBlock_version", "train"}
    _check_dicts_equal(resaved_metadata, initial_metadata, fields_to_not_check)


def test_models_with_funky_load_do_not_throw():
    @caikit.core.block("00110203-baad-beef-0809-0a0b0c0d0e0f", "FunkyBlock", "0.0.1")
    class _FunkyModel(SampleBlock):
        @classmethod
        def load(cls, model_path):
            return (super().load(model_path), "something else")

    model = _FunkyModel()
    with tempfile.TemporaryDirectory() as tempdir:
        model.save(tempdir)
        caikit.core.load(tempdir)


# pylint: disable=redefined-outer-name
def test_load_can_be_called_directly_with_non_standard_kwargs(sample_model_path):
    initial_metadata = _load_model_metadata(sample_model_path)
    # note that
    # - no positional arguments given
    # - path is `model_path` not `module_path`
    model = SampleBlock.load(foo="bar", test_kw="arg", model_path=sample_model_path)

    assert len(model.metadata) > 0
    _check_dicts_equal(initial_metadata, model.metadata, {"module_id", "model_path"})

    # Write a class that doesn't have a `xxx_path` arg for load
    @caikit.core.block("00110203-0809-beef-baad-0a0b0c0d0e0f", "FunkyBlock", "0.0.1")
    class _FunkyModel(SampleBlock):
        @classmethod
        def load(cls, some_really_odd_param_name):
            return super().load(some_really_odd_param_name)

    # check this doesn't raise
    # (it won't extract metadata though...)
    _FunkyModel.load(some_really_odd_param_name=sample_model_path)


# pylint: disable=redefined-outer-name
def test_parent_class_loads_work(sample_model_path):
    """This test ensures that our metadata injector works on blocks that inherit from other
    classes"""
    model = SampleBlock.load(sample_model_path)

    assert isinstance(model, SampleBlock)


def test_workflows_save_correct_module_paths():
    with tempfile.TemporaryDirectory() as tempdir:
        SampleWorkflow(SampleBlock()).save(tempdir)
        workflow = caikit.core.load(tempdir)

    # Should only be the path to the one block
    assert len(workflow.metadata["module_paths"]) == 1
    # Add a fake module path
    module_paths = workflow.metadata["module_paths"]
    module_paths["some_module"] = "some_path"
    assert len(workflow.metadata["module_paths"]) == 2

    # Assert that re-loading works fine and fake path not in new model's metadata
    with tempfile.TemporaryDirectory() as tempdir:
        workflow.save(tempdir)
        reloaded_workflow = caikit.core.load(tempdir)
    assert len(reloaded_workflow.metadata["module_paths"]) == 1
