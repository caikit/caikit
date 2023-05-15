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

# Standard
import os
import tempfile

# Local
from caikit.core.workflows import workflow
from caikit.core.workflows.base import WorkflowLoader, WorkflowSaver

# pylint: disable=import-error
from sample_lib.blocks.sample_task import SampleBlock
from sample_lib.data_model.sample import SampleInputType, SampleTask
from sample_lib.workflows.sample_task import SampleWorkflow

# Unit Test Infrastructure
from tests.base import TestCaseBase
import caikit.core


class SampleWorkflowLoader(caikit.core.module.ModuleLoader):
    pass


class TestWorkflowBase(TestCaseBase):
    def setUp(self):
        self.DUMMY_WORKFLOW = caikit.core.WorkflowBase()
        self.NO_OP_WORKFLOW = SampleWorkflow()
        self.NO_OP_WORKFLOW_CLASS = SampleWorkflow

    def test_init_available(self):
        model = caikit.core.WorkflowBase()
        self.assertIsInstance(model, caikit.core.WorkflowBase)

    def test_load_not_implemented(self):
        workflow_path = os.path.join(self.fixtures_dir, "dummy_workflow")
        with self.assertRaises(NotImplementedError):
            caikit.core.WorkflowBase.load(workflow_path)

    def test_run_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.DUMMY_WORKFLOW.run()

    def test_save_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.DUMMY_WORKFLOW.save("dummy_path")

    def test_train_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            caikit.core.WorkflowBase.train()

    def test_timed_run_seconds(self):
        # with seconds
        num_seconds = 0.01
        time_passed, iterations_passed, _ = self.NO_OP_WORKFLOW.timed_run(
            sample_input=SampleInputType(name="Gabe"), num_seconds=num_seconds
        )
        self.assertIsInstance(time_passed, float)
        self.assertLess(num_seconds, time_passed)
        self.assertIsInstance(iterations_passed, int)

    def test_timed_run_iterations(self):
        # with iterations
        num_iterations = 1
        time_passed, iterations_passed, _ = self.NO_OP_WORKFLOW.timed_run(
            sample_input=SampleInputType(name="Gabe"), num_iterations=num_iterations
        )
        self.assertIsInstance(time_passed, float)
        self.assertIsInstance(iterations_passed, int)
        self.assertEqual(num_iterations, iterations_passed)

    def test_timed_load(self):
        time_passed, model = self.NO_OP_WORKFLOW_CLASS.timed_load(
            module_path=os.path.join(self.fixtures_dir, "dummy_workflow")
        )
        self.assertIsInstance(time_passed, float)
        self.assertIsInstance(model, self.NO_OP_WORKFLOW_CLASS)

    def test_validate_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            SampleWorkflow.validate_training_data("not a file")


class TestWorkflowAnnotation(TestCaseBase):
    def test_workflow_annotation_adds_metadata_to_class(self):
        # Declare a new dummy workflow
        @workflow("ABCDE", "MyNewWorkflow", "0.0.1", SampleTask)
        class MyNewWorkflow(caikit.core.WorkflowBase):
            # pylint: disable=no-method-argument,super-init-not-called
            def __init__():
                pass

        self.assertEqual(MyNewWorkflow.WORKFLOW_ID, "ABCDE")
        self.assertEqual(MyNewWorkflow.MODULE_ID, "ABCDE")
        self.assertEqual(MyNewWorkflow.WORKFLOW_NAME, "MyNewWorkflow")
        self.assertEqual(MyNewWorkflow.WORKFLOW_VERSION, "0.0.1")
        self.assertEqual(
            MyNewWorkflow.WORKFLOW_CLASS,
            MyNewWorkflow.__module__ + "." + MyNewWorkflow.__qualname__,
        )
        self.assertIsNotNone(MyNewWorkflow.PRODUCER_ID)
        self.assertEqual(MyNewWorkflow.PRODUCER_ID.name, "MyNewWorkflow")
        self.assertEqual(MyNewWorkflow.PRODUCER_ID.version, "0.0.1")

    def test_workflow_annotation_registers_workflow_in_module_registry(self):
        # Declare a new dummy workflow
        @workflow("ABCDE-1", "MyNewWorkflow2", "0.0.2", SampleTask)
        # pylint: disable=unused-variable
        class MyNewWorkflow2(caikit.core.WorkflowBase):
            # pylint: disable=no-method-argument,super-init-not-called
            def __init__():
                pass

        self.assertIsNotNone(caikit.core.MODULE_REGISTRY.get("ABCDE-1"))

    def test_workflow_annotation_registers_workflow_in_workflow_registry(self):
        # Declare a new dummy workflow
        @workflow("ABCDE-2", "MyNewWorkflow3", "0.0.2", SampleTask)
        # pylint: disable=unused-variable
        class MyNewWorkflow3(caikit.core.WorkflowBase):
            # pylint: disable=no-method-argument,super-init-not-called
            def __init__():
                pass

        self.assertIsNotNone(caikit.core.WORKFLOW_REGISTRY.get("ABCDE-2"))

    def test_workflow_annotation_registers_will_not_register_duplicate_workflow_ids(
        self,
    ):
        # Declare a new dummy workflow
        def declare_workflow():
            @workflow("ABCDE-3", "MyNewWorkflow4", "0.0.2", SampleTask)
            # pylint: disable=unused-variable
            class MyNewWorkflow4(caikit.core.WorkflowBase):
                # pylint: disable=no-method-argument,super-init-not-called
                def __init__():
                    pass

        declare_workflow()  # Should succeed

        # Verify the fist workflow declaration is in the registry
        self.assertIsNotNone(caikit.core.MODULE_REGISTRY.get("ABCDE-3"))

        with self.assertRaises(RuntimeError):
            declare_workflow()  # Should fail (workflow was already registered)


class TestWorkflowLoader(TestCaseBase):
    @classmethod
    def setUpClass(cls):
        @caikit.core.workflow(
            "A32D68FA-E5E6-41BD-BAAE-77A880EB6878",
            "SampleWorkflow",
            "0.0.1",
            SampleTask,
        )
        class TestSampleWorkflow(caikit.core.WorkflowBase):
            pass

        workflow_path = os.path.join(cls.fixtures_dir, "dummy_workflow")
        cls.dummy_workflow = TestSampleWorkflow()
        cls.loader = WorkflowLoader(SampleWorkflow(), workflow_path)

    def test_missing_arg(self):
        self.assertIsNone(self.loader.load_arg("invalid"))

    def test_missing_module_key(self):
        with self.assertRaisesRegex(KeyError, "Missing required"):
            self.loader.load_module("invalid")

    def test_module_load(self):
        self.assertIsInstance(
            self.loader.load_module("dummy_model"),
            SampleBlock,
        )

    def test_module_load_list_invalid(self):
        with self.assertRaises(KeyError):
            self.loader.load_module_list("invalid")

    def test_module_load_list_bad(self):
        with self.assertRaises(TypeError):
            self.loader.load_module_list("dummy_model")

    def test_module_load_list(self):
        loaded_modules = self.loader.load_module_list("dummy_models")
        self.assertIsInstance(loaded_modules, list)

        for module in loaded_modules:
            self.assertIsInstance(module, SampleBlock)


class TestWorkflowSaver(TestCaseBase):
    @classmethod
    def setUpClass(cls):
        cls.dummy_workflow = SampleWorkflow()

    def test_workflow_saver_attribs(self):
        # make sure the saver has the desired config attrs
        with tempfile.TemporaryDirectory() as tempdir:
            with WorkflowSaver(
                self.dummy_workflow,
                model_path=tempdir,
            ) as workflow_saver:
                # name
                self.assertIsInstance(workflow_saver.config.get("name"), str)
                self.assertEqual(
                    workflow_saver.config.get("name"), self.dummy_workflow.WORKFLOW_NAME
                )

                # workflow version
                self.assertIsInstance(workflow_saver.config.get("version"), str)
                self.assertEqual(
                    workflow_saver.config.get("version"),
                    self.dummy_workflow.WORKFLOW_VERSION,
                )

                # workflow class
                self.assertIsInstance(workflow_saver.config.get("workflow_class"), str)
                self.assertEqual(
                    workflow_saver.config.get("workflow_class"),
                    self.dummy_workflow.WORKFLOW_CLASS,
                )

                # workflow id
                self.assertIsInstance(workflow_saver.config.get("workflow_id"), str)
                self.assertEqual(
                    workflow_saver.config.get("workflow_id"),
                    self.dummy_workflow.WORKFLOW_ID,
                )

                # sample_lib_version
                self.assertIsInstance(
                    workflow_saver.config.get("sample_lib_version"), str
                )
                self.assertEqual(
                    workflow_saver.config.get("sample_lib_version"),
                    "1.2.3",
                )

                # creation date
                self.assertIsInstance(workflow_saver.config.get("created"), str)

            # and that the config gets written
            self.assertTrue(os.path.isfile(os.path.join(tempdir, "config.yml")))

    def test_save_module_saves_a_block_to_model_subdirectory(self):
        dummy_path = os.path.join(self.fixtures_dir, "dummy_workflow", "dummy_block")
        dummy_block = caikit.core.load(dummy_path)

        with tempfile.TemporaryDirectory() as tempdir:
            with WorkflowSaver(
                self.dummy_workflow,
                model_path=tempdir,
            ) as workflow_saver:
                workflow_saver.save_module(dummy_block, "dummy")

                self.assertTrue(os.path.exists(os.path.join(tempdir, "dummy")))
                self.assertIsNotNone(workflow_saver.config.get("module_paths"))
                self.assertEqual(
                    workflow_saver.config.get("module_paths"), {"dummy": "./dummy"}
                )

    def test_save_module_saves_multiple_blocks_to_model_subdirectories(self):
        dummy_path = os.path.join(self.fixtures_dir, "dummy_workflow", "dummy_block")
        dummy_block = caikit.core.load(dummy_path)

        with tempfile.TemporaryDirectory() as tempdir:
            with WorkflowSaver(
                self.dummy_workflow,
                model_path=tempdir,
            ) as workflow_saver:
                workflow_saver.save_module(dummy_block, "dummy")
                workflow_saver.save_module(dummy_block, "dummy2")
                workflow_saver.save_params(test2="val2", test3="val3")
                self.assertTrue(os.path.exists(os.path.join(tempdir, "dummy")))
                self.assertTrue(os.path.exists(os.path.join(tempdir, "dummy2")))
                self.assertIsNotNone(workflow_saver.config.get("module_paths"))
                self.assertEqual(
                    workflow_saver.config.get("module_paths"),
                    {"dummy": "./dummy", "dummy2": "./dummy2"},
                )
                self.assertEqual("val2", workflow_saver.config.get("test2"))
                self.assertEqual("val3", workflow_saver.config.get("test3"))

    def test_save_module_list_invalid_module(self):
        """Test when the input module for the save is invalid."""
        dummy_models_with_rel_path = {"dummy": "invalid"}

        with tempfile.TemporaryDirectory() as tempdir:
            with WorkflowSaver(
                self.dummy_workflow,
                model_path=tempdir,
            ) as workflow_saver:
                with self.assertRaises(TypeError):
                    workflow_saver.save_module_list(
                        dummy_models_with_rel_path, "dummy_model"
                    )

    def test_save_module_list_invalid_dict(self):
        """Test when the input dict for the save is invalid."""
        dummy_models_with_rel_path = "invalid"

        with tempfile.TemporaryDirectory() as tempdir:
            with WorkflowSaver(
                self.dummy_workflow,
                model_path=tempdir,
            ) as workflow_saver:
                with self.assertRaises(TypeError):
                    workflow_saver.save_module_list(dummy_models_with_rel_path, "dummy")

    def test_save_module_list_invalid_config_key(self):
        """Test when the config key is invalid."""
        dummy_path = os.path.join(self.fixtures_dir, "dummy_workflow", "dummy_block")
        dummy_block = caikit.core.load(dummy_path)
        dummy_models_with_rel_path = {"dummy_sad": dummy_block}

        with tempfile.TemporaryDirectory() as tempdir:
            with WorkflowSaver(
                self.dummy_workflow,
                model_path=tempdir,
            ) as workflow_saver:
                with self.assertRaises(TypeError):
                    workflow_saver.save_module_list(dummy_models_with_rel_path, {})

    def test_save_module_list(self):
        """Test valid module list save with a list of syntax blocks."""
        dummy_path = os.path.join(self.fixtures_dir, "dummy_workflow", "dummy_block")
        dummy_block = caikit.core.load(dummy_path)
        dummy_models_with_rel_path = {"dummy_path": dummy_block}

        with tempfile.TemporaryDirectory() as tempdir:
            with WorkflowSaver(
                self.dummy_workflow,
                model_path=tempdir,
            ) as workflow_saver:
                list_of_rel_path, _ = workflow_saver.save_module_list(
                    dummy_models_with_rel_path, "dummy"
                )
                # Ensure that we only get one relative path back
                self.assertEqual(len(list_of_rel_path), 1)
                # And that if we split our path components up, get 2 parts...
                saved_subpath = os.path.split(list_of_rel_path[0])
                self.assertTrue(len(saved_subpath) == 2)
                # ...Where the first resolves to the relative path [no hierarchy expected here]
                self.assertTrue(not saved_subpath[0].strip("."))
                # ...And the second is the dummy_path we specified
                self.assertEqual(saved_subpath[1], "dummy_path")
