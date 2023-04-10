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
from types import ModuleType
import os

# Local
from caikit.core.toolkit import extension_utils

# Unit Test Infrastructure
from tests.base import TestCaseBase
import caikit.core


class TestExtensionUtils(TestCaseBase):
    @classmethod
    def setUpClass(cls):
        cls.caikit_lib = caikit.core
        cls.extension_like_lib = caikit.core
        cls.nonextension_like_lib = os
        cls.extension_like_lib_name = cls.extension_like_lib.__name__
        cls.nonextension_like_lib_name = cls.nonextension_like_lib.__name__

    def test_is_extension_like_happy(self):
        """Ensure caikit core looks like a caikit * library"""
        self.assertTrue(extension_utils.is_extension_like(self.extension_like_lib))

    def test_is_extension_like_sad(self):
        """Ensure something like os does NOT look like a caikit * library"""
        self.assertFalse(extension_utils.is_extension_like(self.nonextension_like_lib))

    def test_enable_extension_binding(self):
        """Ensure that enabling extension binding creates an extensions subpackage"""
        self.assertFalse(hasattr(self.caikit_lib, extension_utils.EXTENSIONS_ATTR_NAME))
        extension_utils.enable_extension_binding(self.caikit_lib)
        self.assertTrue(hasattr(self.caikit_lib, extension_utils.EXTENSIONS_ATTR_NAME))
        self.assertIsInstance(self.caikit_lib, ModuleType)

    def test_enable_extension_binding_non_extension_like(self):
        """Ensure that we need to pass a ModuleType to extension binding enablement"""
        with self.assertRaises(TypeError):
            extension_utils.enable_extension_binding(42)

    def test_bind_extension_like(self):
        """Ensure we can bind caikit / ext like libs, and skip other modules"""
        extension_utils.enable_extension_binding(self.caikit_lib)
        extension_utils.bind_extensions(
            [self.extension_like_lib_name, self.nonextension_like_lib_name],
            self.caikit_lib,
        )
        # Grab our extensions that have been bound
        exts = getattr(self.caikit_lib, extension_utils.EXTENSIONS_ATTR_NAME)
        # Ensure only extension like libs are bound
        self.assertTrue(hasattr(exts, self.extension_like_lib_name))
        self.assertFalse(hasattr(exts, self.nonextension_like_lib_name))
        # Ensure that the bound lib is a ModuleType object
        bound_lib = getattr(exts, self.extension_like_lib_name)
        self.assertIsInstance(bound_lib, ModuleType)

    def test_bind_requires_enablement(self):
        """Ensure that we can't bind prior to initializing the extensions subpackage"""
        with self.assertRaises(ValueError):
            extension_utils.bind_extensions(
                [self.extension_like_lib_name], self.caikit_lib
            )

    def test_bind_invalid_extensions(self):
        """Ensure we type check our lib extensions properly"""
        extension_utils.enable_extension_binding(self.caikit_lib)
        # Ensure that we catch bad types for for the arg itself
        with self.assertRaises(TypeError):
            extension_utils.bind_extensions([42], self.caikit_lib)
        # And also within the nested type of the iterable
        with self.assertRaises(TypeError):
            extension_utils.bind_extensions(42, self.caikit_lib)

    def test_bind_invalid_lib_handle_type(self):
        """Ensure we type check our lib handle properly"""
        extension_utils.enable_extension_binding(self.caikit_lib)
        with self.assertRaises(TypeError):
            extension_utils.bind_extensions([self.extension_like_lib_name], 42)

    def tearDown(self):
        # Disable extension binding on the caikit library between tests
        if hasattr(self.caikit_lib, extension_utils.EXTENSIONS_ATTR_NAME):
            delattr(self.caikit_lib, extension_utils.EXTENSIONS_ATTR_NAME)
