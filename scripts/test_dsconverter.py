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

# Third Party
# Import the script code
from dsconverter import CustomDocstringConverter
import pytest


# Test case for convert_to_google_style function
def test_get_docstrings():
    converter = CustomDocstringConverter()
    test_path = "/Users/mwjohnson/Code/caikit/caikit/core/modules/config.py"
    docstrings = converter.extract_docstrings_from_file(test_path)
    for docstring in docstrings:
        print("foo:\n", docstring)
    assert 1 == 2


def test_basic_custom_docstring():
    converter = CustomDocstringConverter()
    test1 = '''
    """Method for extracting pred set from dataset. Implemented in subclass.

        Args:
            dataset:  object
                In-memory version of whatever
            preprocess_func:  method
                Function used as proxy for any preliminary steps that need to be taken to run the
                model on the input text. This helper function ultimately leads to the input to this
                module and may involve executing other modules.
            foo
                Description
            *args, **kwargs: dict
                Optional keyword arguments for prediction set extraction.
        Returns:
            list
                List of labels in the format of the module_type that is being called.
        """
    '''

    converted = converter.convert_to_google_style(test1)
    print("Pre-conversion:\n", test1, "\nPost-conversion:\n", converted)

    assert 1 == 2


# Test case for convert_to_google_style function
def test_complex_custom_docstring():
    converter = CustomDocstringConverter()
    test1 = '''
    """Time a model `load` call.

        Args:
            *args: list
                Will be passed to `self.load`.
            **kwargs:  dict
                Will be passed to `self.load` -- the only way to pass arbitrary arguments to
                `self.load` from this function.

        Returns:
            int, caikit.core._ModuleBase
                The first return value is the total time spent in the `self.load` call. The second
                return value is the loaded model.

        Notes:
            You can pass everything that should go to the run function normally using args/kwargs.
            Example: `model.timed_load("/model/path/dir")`
        """
    '''

    converted = converter.convert_to_google_style(test1)
    print("Pre-conversion:\n", test1, "\nPost-conversion:\n", converted)

    assert 1 == 2


def test_no_arg_desc_docstring():
    converter = CustomDocstringConverter()
    test1 = '''
    """Time a model `load` call.

        Args:
            *args: list
            **kwargs:  dict
            hello: foo

        Returns:
            int, caikit.core._ModuleBase
                The first return value is the total time spent in the `self.load` call. The second
                return value is the loaded model.

        Notes:
            You can pass everything that should go to the run function normally using args/kwargs.
            Example: `model.timed_load("/model/path/dir")`
        """
    '''

    converted = converter.convert_to_google_style(test1)
    print("Pre-conversion:\n", test1, "\nPost-conversion:\n", converted)

    assert 1 == 2


def test_no_arg_description_returns():
    converter = CustomDocstringConverter()
    test1 = '''
    """Extract the core list of predictions that is needed for quality evaluation

        Args:
            pred_set: list
        Returns:
            pred_annotations: list
        """
    '''
    converted = converter.convert_to_google_style(test1)
    print("Pre-conversion:\n", test1, "\nPost-conversion:\n", converted)

    assert 1 == 2


def test_already_converted():
    converter = CustomDocstringConverter()
    test1 = '''
    """Time a model `load` call.

        Args:
             *args (list): Will be passed to `self.load`.
             **kwargs (dict): Will be passed to `self.load` -- the only way to
                pass arbitrary arguments to `self.load` from this function.

        Returns:
             int, caikit.core._ModuleBase: The first return value is the
               total time spent in the `self.load` call. The second return value is the
               loaded model.

        Notes:
            You can pass everything that should go to the run function normally using args/kwargs.
            Example: `model.timed_load("/model/path/dir")`
        """
        '''
    converted = converter.convert_to_google_style(test1)
    print("Pre-conversion:\n", test1, "\nPost-conversion:\n", converted)

    assert 1 == 2


# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])
