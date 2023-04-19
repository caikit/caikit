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
import shutil
import tempfile

# Local
from caikit.config import get_config
from caikit.core.blocks import block
from caikit.core.blocks.base import BlockSaver
from caikit.core.toolkit.serializers import JSONSerializer

# pylint: disable=import-error
from sample_lib.blocks.sample_task import SampleBlock
from sample_lib.data_model.sample import SampleInputType

# Unit Test Infrastructure
from tests.base import TestCaseBase
import caikit.core


class TestBlockBase(TestCaseBase):
    @classmethod
    def setUpClass(cls):
        cls.dummy_block_class = SampleBlock

    def setUp(self):
        self.base_block_instance = caikit.core.BlockBase()
        self.dummy_model_path = os.path.join(self.fixtures_dir, "dummy_block")
        self.dummy_block_instance = self.dummy_block_class(self.dummy_model_path)

    def test_init_available(self):
        model = caikit.core.BlockBase([0, 1, 2], kw1=0, kw2=1, kw3=2)
        self.assertIsInstance(model, caikit.core.BlockBase)

    def test_load_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            caikit.core.BlockBase.load()

    def test_run_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.base_block_instance.run()

    def test_save_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.base_block_instance.save("dummy_path")

    def test_train_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            caikit.core.BlockBase.train()

    def test_timed_run_seconds(self):
        # with seconds
        num_seconds = 0.01
        time_passed, iterations_passed, _ = self.dummy_block_instance.timed_run(
            sample_input=SampleInputType(name="Gabe"), num_seconds=num_seconds
        )
        self.assertIsInstance(time_passed, float)
        self.assertLess(num_seconds, time_passed)
        self.assertIsInstance(iterations_passed, int)

    def test_timed_run_iterations(self):
        # with iterations
        num_iterations = 1
        time_passed, iterations_passed, _ = self.dummy_block_instance.timed_run(
            sample_input=SampleInputType(name="Gabe"), num_iterations=num_iterations
        )
        self.assertIsInstance(time_passed, float)
        self.assertIsInstance(iterations_passed, int)
        self.assertEqual(num_iterations, iterations_passed)

    def test_timed_load(self):
        time_passed, model = self.dummy_block_class.timed_load(
            module_path=self.dummy_model_path
        )
        self.assertIsInstance(time_passed, float)
        self.assertIsInstance(model, self.dummy_block_class)


class TestBlockAnnotation(TestCaseBase):
    def test_block_annotation_adds_metadata_to_class(self):
        # Declare a new dummy block
        @block("12345", "MyNewBlock", "0.0.1")
        class MyNewBlock(caikit.core.BlockBase):
            # pylint: disable=no-method-argument,super-init-not-called
            def __init__():
                pass

        self.assertEqual(MyNewBlock.BLOCK_ID, "12345")
        self.assertEqual(MyNewBlock.MODULE_ID, "12345")
        self.assertEqual(MyNewBlock.BLOCK_NAME, "MyNewBlock")
        self.assertEqual(MyNewBlock.BLOCK_VERSION, "0.0.1")
        self.assertEqual(
            MyNewBlock.BLOCK_CLASS,
            MyNewBlock.__module__ + "." + MyNewBlock.__qualname__,
        )
        self.assertIsNotNone(MyNewBlock.PRODUCER_ID)
        self.assertEqual(MyNewBlock.PRODUCER_ID.name, "MyNewBlock")
        self.assertEqual(MyNewBlock.PRODUCER_ID.version, "0.0.1")

    def test_block_annotation_registers_block_in_module_registry(self):
        # Declare a new dummy block
        @block("12345-A", "MyNewBlock2", "0.0.2")
        # pylint: disable=unused-variable
        class MyNewBlock2(caikit.core.BlockBase):
            # pylint: disable=no-method-argument,super-init-not-called
            def __init__():
                pass

        self.assertIsNotNone(caikit.core.MODULE_REGISTRY.get("12345-A"))

    def test_block_annotation_registers_block_in_block_registry(self):
        # Declare a new dummy block
        @block("12345-B", "MyNewBlock3", "0.0.2")
        # pylint: disable=unused-variable
        class MyNewBlock3(caikit.core.BlockBase):
            # pylint: disable=no-method-argument,super-init-not-called
            def __init__():
                pass

        self.assertIsNotNone(caikit.core.BLOCK_REGISTRY.get("12345-B"))

    def test_block_annotation_registers_will_not_register_duplicate_block_ids(self):
        # Declare a new dummy block
        def declare_block():
            @block("12345-C", "MyNewBlock4", "0.0.2")
            # pylint: disable=unused-variable
            class MyNewBlock4(caikit.core.BlockBase):
                # pylint: disable=no-method-argument,super-init-not-called
                def __init__():
                    pass

        declare_block()  # Should succeed

        # Verify the fist block declaration is in the registry
        self.assertIsNotNone(caikit.core.MODULE_REGISTRY.get("12345-C"))

        with self.assertRaises(RuntimeError):
            declare_block()  # Should fail (block was already registered)


class TestBlockSaver(TestCaseBase):
    @classmethod
    def setUpClass(cls):
        cls.dummy_block = SampleBlock()

    def test_block_saver_attribs(self):
        # make sure the saver has the desired config attrs
        with tempfile.TemporaryDirectory() as tempdir:
            with BlockSaver(
                self.dummy_block,
                model_path=tempdir,
            ) as block_saver:
                # name
                self.assertIsInstance(block_saver.config.get("name"), str)
                self.assertEqual(
                    block_saver.config.get("name"), self.dummy_block.BLOCK_NAME
                )

                # block version
                self.assertIsInstance(block_saver.config.get("version"), str)
                self.assertEqual(
                    block_saver.config.get("version"), self.dummy_block.BLOCK_VERSION
                )

                # block class
                self.assertIsInstance(block_saver.config.get("block_class"), str)
                self.assertEqual(
                    block_saver.config.get("block_class"), self.dummy_block.BLOCK_CLASS
                )

                # block id
                self.assertIsInstance(block_saver.config.get("block_id"), str)
                self.assertEqual(
                    block_saver.config.get("block_id"), self.dummy_block.BLOCK_ID
                )

                # sample_lib_version
                self.assertIsInstance(block_saver.config.get("sample_lib_version"), str)
                self.assertEqual(
                    block_saver.config.get("sample_lib_version"),
                    "1.2.3",
                )

                # creation date
                self.assertIsInstance(block_saver.config.get("created"), str)

            # and that the config gets written
            self.assertTrue(os.path.isfile(os.path.join(tempdir, "config.yml")))

    def test_add_dir(self):
        with tempfile.TemporaryDirectory() as tempdir:
            with BlockSaver(
                self.dummy_block,
                model_path=tempdir,
            ) as block_saver:
                # add a directory called `a`
                a_rel, a_abs = block_saver.add_dir("a")
                self.assertEqual(os.path.normpath(a_rel), os.path.basename(a_abs))
                self.assertTrue(os.path.isdir(a_abs))

                # add `b/c` inside `a`
                bc_rel, bc_abs = block_saver.add_dir("b/c", "a")
                self.assertTrue(os.path.isdir(bc_abs))
                self.assertTrue(bc_abs.endswith(bc_rel))

            # verify `a/b/c` was created
            self.assertTrue(os.path.isdir(os.path.join(tempdir, "a/b/c")))

    def test_update_config(self):
        # verify that we can add some config options with `saver.update_config`
        with tempfile.TemporaryDirectory() as tempdir:
            with BlockSaver(
                self.dummy_block,
                model_path=tempdir,
            ) as block_saver:
                block_saver.update_config(
                    {
                        "training_mode": "stochastic",
                        "training_epochs": 1000,
                    }
                )

                self.assertEqual(block_saver.config.get("training_mode"), "stochastic")
                self.assertEqual(block_saver.config.get("training_epochs"), 1000)

    def test_copy_file(self):
        # verify that we can copy a file into the model with `saver.copy_file`
        with tempfile.TemporaryDirectory() as tempdir:
            with BlockSaver(
                self.dummy_block,
                model_path=tempdir,
            ) as block_saver:
                block_saver.copy_file(
                    os.path.join(self.fixtures_dir, "linux.txt"), "artifacts"
                )
            self.assertTrue(
                os.path.isfile(os.path.join(tempdir, "artifacts/linux.txt"))
            )

    def test_remove_on_error(self):
        # we cannot use `TemporaryDirectory` context manager because the model saver will
        # remove it in the middle of the test
        tempdir = tempfile.mkdtemp()

        try:
            # verify that exceptions are re-raised from the context manager
            with self.assertRaises(RuntimeError):
                with BlockSaver(
                    self.dummy_block,
                    model_path=tempdir,
                ) as block_saver:
                    block_saver.add_dir("artifacts")
                    raise RuntimeError("test")

            # verify that our error caused the model tree to be removed
            self.assertFalse(os.path.exists(os.path.join(tempdir, "config.yml")))
            self.assertFalse(os.path.exists(os.path.join(tempdir, "artifacts")))

        finally:
            shutil.rmtree(tempdir, ignore_errors=True)

    def test_save_object_saves_a_json_object_to_model_root_dir(self):
        with tempfile.TemporaryDirectory() as tempdir:
            with BlockSaver(
                self.dummy_block,
                model_path=tempdir,
            ) as block_saver:
                serializer = JSONSerializer()

                block_saver.save_object({"foo": "bar"}, "test.json", serializer)

                self.assertTrue(os.path.exists(os.path.join(tempdir, "test.json")))

    def test_save_object_saves_a_json_object_to_model_sub_dir(self):
        with tempfile.TemporaryDirectory() as tempdir:
            with BlockSaver(
                self.dummy_block,
                model_path=tempdir,
            ) as block_saver:
                serializer = JSONSerializer()

                block_saver.save_object(
                    {"foo": "bar"}, "test.json", serializer, "artifacts"
                )

                self.assertTrue(
                    os.path.exists(os.path.join(tempdir, "artifacts/test.json"))
                )
