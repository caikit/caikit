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


def test_examples_in_docstring():
    converter = CustomDocstringConverter()
    test1 = '''
    """Create a new data stream from a python iterable, such as a list or tuple.  This data
        stream produces a single data item for each element of the iterable..

        Args:
            data:  iterable
                A list or tuple or other python iterable used to construct a new data stream where
                each data item contains a single data item.

        Returns:
            DataStream
                A new data stream that produces data items from the elements of `data`.

        Examples:
            >>> list_stream = DataStream.from_iterable([1, 2, 3])
            >>> for data_item in list_stream:
            >>>     print(data_item)
            1
            2
            3
        """
        '''
    converted = converter.convert_to_google_style(test1)
    print("Pre-conversion:\n", test1, "\nPost-conversion:\n", converted)

    assert 1 == 2


def test_no_data_type_arg():
    converter = CustomDocstringConverter()
    test1 = '''
    """Create a new data stream from a python iterable, such as a list or tuple.  This data
        stream produces a single data item for each element of the iterable..

        Args:
            proto:
                A protocol buffer to be populated.
        
        Returns:
            DataStream
                A new data stream that produces data items from the elements of `data`.

    """
    '''
    converted = converter.convert_to_google_style(test1)
    print("Pre-conversion:\n", test1, "\nPost-conversion:\n", converted)

    assert 1 == 2


def test_no_data_type_arg_same_line_desc():
    converter = CustomDocstringConverter()
    test1 = '''
    """Create a new data stream from a python iterable, such as a list or tuple.  This data
        stream produces a single data item for each element of the iterable..

        Args:
            proto: A protocol buffer to be populated.
        
        Returns:
            DataStream
                A new data stream that produces data items from the elements of `data`.

    """
    '''
    converted = converter.convert_to_google_style(test1)
    print("Pre-conversion:\n", test1, "\nPost-conversion:\n", converted)

    assert 1 == 2


def test_no_data_type_arg_multi_line_desc():
    converter = CustomDocstringConverter()
    test1 = '''
    """Create a new data stream from a python iterable, such as a list or tuple.  This data
        stream produces a single data item for each element of the iterable..

        Args:
            proto: 
                A protocol buffer to be populated.
                More description here
                More here!
        
        Returns:
            DataStream
                A new data stream that produces data items from the elements of `data`.

    """
    '''
    converted = converter.convert_to_google_style(test1)
    print("Pre-conversion:\n", test1, "\nPost-conversion:\n", converted)

    assert 1 == 2


def test_buncha_ors():
    converter = CustomDocstringConverter()
    test1 = '''
    """Create a new data stream from a python iterable, such as a list or tuple.  This data
        stream produces a single data item for each element of the iterable..

        Args:
            proto: int | int | int | int | int | int | int 
                A protocol buffer to be populated.
                More description here
                More here!
        
        Returns:
            DataStream
                A new data stream that produces data items from the elements of `data`.

    """
    '''
    converted = converter.convert_to_google_style(test1)
    print("Pre-conversion:\n", test1, "\nPost-conversion:\n", converted)

    assert 1 == 2


def test_fewa_ors():
    converter = CustomDocstringConverter()
    test1 = '''
    """Create a new data stream from a python iterable, such as a list or tuple.  This data
        stream produces a single data item for each element of the iterable..

        Args:
            obj: str | caikit.core.data_model.DataBase
                Object to be augmented.
        Returns:
            str | caikit.core.data_model.DataBase
                Augmented object of same type as input obj.
        """
    '''
    converted = converter.convert_to_google_style(test1)
    print("Pre-conversion:\n", test1, "\nPost-conversion:\n", converted)

    assert 1 == 2


def test_data_type_same_line_returns():
    converter = CustomDocstringConverter()
    test1 = '''
    """Configure caikit for your usage!
    Merges into the internal aconfig.Config object with overrides from multiple sources.
    Sources, last takes precedence:
        1. The existing configuration from calls to `caikit.configure()`
        2. The config from `config_yml_path`
        3. The config from `config_dict`
        4. The config files specified in the `config_files` configuration
            (NB: This may be set by the `CONFIG_FILES` environment variable)
        5. Environment variables, in ALL_CAPS_SNAKE_FORMAT
    Args:
        config_yml_path (Optional[str]): The path to the base configuration yaml
            with overrides for your usage.
        config_dict (Optional[Dict]): Config overrides in dictionary form
    Returns: None
        This only sets the config object that is returned by `caikit.get_config()`
    """
    '''
    converted = converter.convert_to_google_style(test1)
    print("Pre-conversion:\n", test1, "\nPost-conversion:\n", converted)

    assert 1 == 2

def test_example_usage():
    converter = CustomDocstringConverter()
    test1 = '''
    """The decorator for AI Task classes.

    This defines an output data model type for the task, and a minimal set of required inputs
    that all public models implementing this task must accept.

    As an example, the `caikit.interfaces.nlp.SentimentTask` might look like::

        @task(
            required_inputs={
                "raw_document": caikit.interfaces.nlp.RawDocument
            },
            output_type=caikit.interfaces.nlp.SentimentPrediction
        )
        class SentimentTask(caikit.TaskBase):
            pass

    and a public model that implements this task might have a .run function that looks like::

        def run(raw_document: caikit.interfaces.nlp.RawDocument,
                inference_mode: str = "fast",
                device: caikit.interfaces.common.HardwareEnum) ->
                    caikit.interfaces.nlp.SentimentPrediction:
            # impl

    Note the run function may include other arguments beyond the minimal required inputs for
    the task.

    Args:
        required_parameters (Dict[str, ValidInputTypes]): The required parameters that all public
            models' .run functions must contain. A dictionary of parameter name to parameter
            type, where the types can be in the set of:
                - Python primitives
                - Caikit data models
                - Iterable containers of the above
                - Caikit model references (maybe?)
        output_type (Type[DataBase]): The output type of the task, which all public models'
            .run functions must return. This must be a caikit data model type.

    Returns:
        A decorator function for the task class, registering it with caikit's core registry of
            tasks.
    """
    '''
    converted = converter.convert_to_google_style(test1)
    print("Pre-conversion:\n", test1, "\nPost-conversion:\n", converted)

    assert 1 == 2

def test_example_usage():
    converter = CustomDocstringConverter()
    test1 = '''
    """Decorator that can be used to mark a function
    or a class as "work in progress". It will result in a warning being emitted
    when the function / class is used.

    Args:
        category: WipCategory
            Enum specifying what category of message you want to throw
        action: Action
            Enum specifying what type of action you want to take.
            Example: ERROR or WARNING

    Example Usage:

    ### Decorating class

    1. No configuration:
        @work_in_progress
        class Foo:
            pass

    2. Action and category configuration:
        @work_in_progress(action=Action.WARNING, category=WipCategory.BETA)
        class Foo:
            pass

    ### Decorating Function:

    1. No configuration:
        @work_in_progress
        def foo(*args, **kwargs):
            pass

    2. Action and category configuration:
        @work_in_progress(action=Action.WARNING, category=WipCategory.BETA)
         def foo(*args, **kwargs):
            pass

    ### Sample message:

    foo is still in the BETA phase and subject to change!

    """
    '''
    converted = converter.convert_to_google_style(test1)
    print("Pre-conversion:\n", test1, "\nPost-conversion:\n", converted)

    assert 1 == 2




# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])
