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

"""Shared functionality and interfaces used by *all* moduless.
"""

# TODO: Review to see if the lines in this file can be reduced or
# broken out into another file (e.g. Python sub-modules)
# pylint: disable=too-many-lines

# abstract methods may define arguments but not use them
# pylint: disable=unused-argument

# Standard
from io import BytesIO
from typing import Any, Dict, List, Optional, Union
import collections
import os
import shutil
import tempfile
import time
import types

# First Party
import alog

# Local
from ..data_model import DataBase, DataStream
from ..toolkit import fileio
from ..toolkit.errors import DataValidationError, error_handler
from .loader import ModuleLoader
from .meta import _ModuleBaseMeta
from caikit import core

log = alog.use_channel("MODULE")
error = error_handler.get(log)


class ModuleBase(metaclass=_ModuleBaseMeta):
    """Abstract base class from which all modules should inherit."""

    def __init__(self):
        """Construct a new model."""
        # Set up an empty metadata dictionary, to be:
        # - populated with metadata from `config.yml` files on `load`, and
        # - saved back to `config.yml` files on `save`
        self._metadata = {}
        # Keep an indicator of the backend used to load this model
        self._load_backend = None

    #############
    ## Utility ##
    #############

    @property
    def metadata(self) -> Dict[str, Any]:
        """This module's metadata.

        Returns:
            Dict[str, Any]: A dictionary of this module's metadata

            TODO: Can this be a `ModuleConfig` object instead? (or aconfig.Config)?
        """
        if not hasattr(self, "_metadata") or self._metadata is None:
            self._metadata = {}
        return self._metadata

    def set_load_backend(self, load_backend):
        """Method used by the model manager to indicate the load backend that
        was used to load this module
        """
        self._load_backend = load_backend

    @classmethod
    def get_inference_signature(
        cls, input_streaming: bool, output_streaming: bool
    ) -> Optional["caikit.core.signature_parsing.CaikitMethodSignature"]:
        """Returns the inference method signature that is capable of running the module's task
        for the given flavors of input and output streaming
        """
        for in_streaming, out_streaming, signature in cls._INFERENCE_SIGNATURES:
            if in_streaming == input_streaming and out_streaming == output_streaming:
                return signature
        return None

    @property
    def load_backend(self):
        """Get the backend instance used to load this module. This can be used
        in module implementations that require use of a specific backend at
        inference time.
        """
        return self._load_backend

    ###################
    ## Instantiation ##
    ###################

    @classmethod
    @alog.logged_function(log.debug)
    def bootstrap(cls, *args, **kwargs):
        """Bootstrap a module. This method can be used to initialize the module
        from artifacts created outside of a particular caikit library
        """
        error(
            "<COR92634438E>",
            NotImplementedError("This is not available in this module."),
        )

    @classmethod
    @alog.logged_function(log.debug)
    def load(cls, model_path, *args, **kwargs):
        """Load a new instance of workflow from a given model_path

        Args:
            model_path (str): Path to workflow
        Returns:
            caikit.core.workflows.base.WorkflowBase: A new instance of any given
                implementation of WorkflowBase
        """
        return cls._load(ModuleLoader(model_path), *args, **kwargs)

    @classmethod
    def _load(cls, module_loader, *args, **kwargs):
        """Load a model."""
        error(
            "<COR88356566E>",
            NotImplementedError("This is not available in this module."),
        )

    @classmethod
    def timed_load(cls, *args, **kwargs):
        """Time a model `load` call.

        Args:
            *args (list): Will be passed to `self.load`.
            **kwargs (dict): Will be passed to `self.load` -- the only way to
                pass arbitrary arguments to `self.load` from this function.

        Returns:
            int, caikit.core._ModuleBase: The first return value is the total
                time spent in the `self.load` call. The second return value is
                the loaded model.

        Notes:
            You can pass everything that should go to the run function normally using args/kwargs.
            Example: `model.timed_load("/model/path/dir")`
        """
        # get initial values
        start_time = time.time()
        # We are calling caikit.core over cls.load because we need to figure out
        # what instance the model belongs to
        model = core.load(*args, **kwargs)
        time_passed = time.time() - start_time
        return time_passed, model

    def validate_loaded_model(self, *args):
        """Validate a loaded model."""
        error(
            "<COR56275627E>",
            NotImplementedError("This is not available in this module."),
        )

    ###################
    ## Serialization ##
    ###################

    @alog.logged_function(log.debug)
    def save(self, model_path, *args, **kwargs):
        """Save a model.

        Args:
            model_path (str): Path on disk to export the model to.
        """
        error(
            "<COR58632237E>",
            NotImplementedError("This is not available in this module."),
        )

    @alog.logged_function(log.debug)
    def as_file_like_object(self, *args, **kwargs):
        """Produces a file-like object corresponding to a zip archive affiliated with a given
        model. This method wraps is functionally similar to .save() - it saves a model into
        a temporary directory and produces a zip archive, then loads the result as a io.BytesIO
        object. The result of this function is also compatible with .load(), and cleanup is
        handled automatically.

        Args:
            *args, **kwargs (dict): Optional keyword arguments for saving.
        Returns:
            io.BytesIO: File like object holding an exported model in memory as
                a io.BytesIO object.
        """
        return BytesIO(self.as_bytes(*args, **kwargs))

    @alog.logged_function(log.debug)
    def as_bytes(self, *args, **kwargs):
        """Produces a bytes object corresponding to a zip archive affiliated with a given
        model. This method wraps is functionally similar to .save() - it saves a model into
        a temporary directory and produces a zip archive, then loads the result as a bytes
        object. The result of this function is also compatible with .load(), and cleanup is
        handled automatically.

        Args:
            *args, **kwargs (dict): Optional keyword arguments for saving.
        Returns:
            bytes: bytes object holding an exported model in memory.
        """
        # Open a temporary directory & do all operations relative to that temporary directory.
        with tempfile.TemporaryDirectory() as ephemeral_model_path:
            # Save the model to the temporary directory
            model_path = os.path.join(ephemeral_model_path, ".model")
            zip_path = os.path.join(ephemeral_model_path, ".archive")
            zip_path_with_ext = zip_path + ".zip"
            self.save(model_path, *args, **kwargs)
            try:
                # Compress the model to a zip archive in the temporary directory
                shutil.make_archive(zip_path, "zip", model_path)
                # Load the zip archive bytes into memory as a file-like object and clean up any disk
                # objects (NOTE: it is safe to delete the archive once we extract the bytes).
                with open(zip_path_with_ext, "rb") as handle:
                    in_memory_archive = handle.read()
            except PermissionError:
                error(
                    "<COR80051233E>",
                    PermissionError(
                        "Unable to create archive to be loaded into memory."
                    ),
                )
            return in_memory_archive

    ###############
    ## Inference ##
    ###############

    @alog.logged_function(log.debug)
    def run(self, *args, **kwargs):
        """Run a model - this typically makes a single prediction and returns an object from the
        data model.
        """
        error(
            "<COR80640037E>",
            NotImplementedError("This is not available in this module."),
        )

    @alog.logged_function(log.debug)
    def run_batch(self, *args, **kwargs):
        """Run a model in batch mode - this typically ingests an iterable of inputs that can be
        applied to run & returns a list of data model objects that run ordinarily returns. A module
        may override this method to provide faster evaluation capabilities, e.g., by leveraging
        vectorization during prediction.

        All provided args and kwargs that should be expanded with the batch should be provided as
        prebatched iterables. If a provided arg/kwarg is not provided as an iterable, it will be
        passed as is to all self contained run calls, which may be the case in some rare cases,
        such as runtime explanability enablement.

        This function is intentionally kept as simple as possible. In order to maintain its
        simplicity, all argument iterables must be the same length, where the length of every
        provided iterable is presumed to be the batch size. If an iterable must be passed as
        arg to each run call, batch run must be called by wrapping it in another iterable and
        duplicating the iterable arg to match the size, or (ideally) overridden in the subclass
        as necessary.

        Args:
            *args: Variable length argument list to be passed directly to run().
            **kwargs: Arbitrary keyword arguments to be passed directly to run().
        Returns:
            tuple: Iterable of prediction outputs, run as a batch.
        """
        predictions = []
        fixed_args = {}
        expanded_args = {}
        fixed_kwargs = {}
        expanded_kwargs = {}
        batch_size = self._validate_and_extract_batch_size(*args, **kwargs)
        # Keep map args to indices - mark iterable nonstrings as expandable args
        for idx, arg in enumerate(args):
            if self._is_expandable_iterable(arg):
                expanded_args[idx] = arg
            else:
                fixed_args[idx] = arg
        # And do the same with kwargs
        for kwarg_key, kwarg_value in kwargs.items():
            if self._is_expandable_iterable(kwarg_value):
                expanded_kwargs[kwarg_key] = kwarg_value
            else:
                fixed_kwargs[kwarg_key] = kwarg_value

        for idx in range(batch_size):
            run_args = self._build_args_for_default_run_with_batch(
                fixed_args, expanded_args, idx
            )
            run_kwargs = self._build_kwargs_for_default_run_with_batch(
                fixed_kwargs, expanded_kwargs, idx
            )
            run_out = self.run(*run_args, **run_kwargs)
            predictions.append(run_out)
        return tuple(predictions)

    def timed_run(self, *args, num_seconds=None, num_iterations=None, **kwargs):
        """Time a number of runs over set seconds or iterations.

        Args:
            *args (list): Will be passed to `self.run`.
            num_seconds (int): Minimum number of seconds to run timed_run over.
                Will most likely be more than this value due to its waiting for
                the each call to `self.run` to finish.
            num_iterations (int): Minimum number of iterations to run timed_run
                over. Will run exactly this many times.
            **kwargs (dict): Will be passed to `self.run`.

        Returns:
            int, int, caikit.core.data_model.DataBase: The first return value is
                the total time spent in the `self.run` loop. The second return
                value is the total number of calls to `self.run` were made. The
                return value is the output of the module's run method

        Notes:
            You can pass everything that should go to the run function normally using args/kwargs.
            Example: `model.timed_run("some example text", num_seconds=60)`

        By default it will run for greater than or equal to 120 seconds.
        """
        # default to running for 120 seconds
        if not (num_seconds or num_iterations):
            num_seconds = 120

        # get initial values
        start_time = time.time()
        iterations_passed = 0
        time_passed = time.time() - start_time

        # stop on seconds or iterations depending on input arguments
        # pylint: disable=unnecessary-lambda-assignment
        continue_condition = (
            lambda t_p, i_p: t_p <= num_seconds if num_seconds else i_p < num_iterations
        )
        response = None

        while continue_condition(time_passed, iterations_passed):
            # use model's run method
            response = self.run(*args, **kwargs)

            # increment output values
            time_passed = time.time() - start_time
            iterations_passed += 1

        return time_passed, iterations_passed, response

    def stream(self, data_stream, *args, **kwargs):
        """Lazily evaluate a run() on a given model by constructing a new data stream generator
        from the results. Note that we do not allow datastreams in args/kwargs. In rare cases,
        this may mean that stream() is not available, e.g., for keywords extraction. In these
        cases, stream() should be overridden in the subclass (module implementation) to allow
        and expand along multiple data streams.

        Args:
            data_stream (caikit.core.data_model.DataStream): Datastream to be
                lazily sequentially processed by the module under consideration.
            *args: Variable length argument list to be passed directly to run().
            **kwargs: Arbitrary keyword arguments to be passed directly to run().
        Returns:
            protobufs: A DataBase object.
        """
        error.type_check("<COR98214589E>", DataStream, data_stream=data_stream)
        # Ensure that no args/kwargs are DataStreams, since these get passed to stream()
        run_argvals = list(args) + list(kwargs.values())
        if any(isinstance(arg, DataStream) for arg in run_argvals):
            error(
                "<COR28828273E>",
                ValueError(
                    "Only one DataStream may be passed when invoking module stream()"
                ),
            )
        # TODO: Add .run_batch() integration
        return DataStream(
            lambda: (self.run(data_item, *args, **kwargs) for data_item in data_stream)
        )

    ##############
    ## Training ##
    ##############

    @classmethod
    @alog.logged_function(log.debug)
    def train(cls, *args, **kwargs):
        """Train a model."""
        error(
            "<COR44977721E>",
            NotImplementedError("This is not available in this module."),
        )

    @classmethod
    def validate_training_data(
        cls, training_data: Union[str, DataStream], limit: int = -1
    ) -> List[DataValidationError]:
        """Validate a set of training data, passed as a filename or as a data stream.
        Return up to `limit` number of DataValidationErrors
        """
        error(
            "<COR56285627E>",
            NotImplementedError("This is not available in this module."),
        )

    ################
    ## Evaluation ##
    ################

    # will be used to evaluate modules; defined in sub-classes
    evaluation_type = None
    # first arg is "self", unfortunately; TODO: get rid of that somehow
    evaluator = None

    @staticmethod
    def find_label_func(*_args, **_kwargs):
        """Function used to extract "label" from a prediction/result of a module's .run method.
        Define if you wish to have more specific evaluation metrics. Implemented in subclass.
        """
        raise NotImplementedError("Func not implemented")

    @staticmethod
    def find_label_data_func(*_args, **_kwargs):
        """Function used to extract data belonging to class "label" from a prediction/result
        of a module's .run method. Define if you wish to have more specific evaluation metrics.
        Implemented in subclass.
        """
        raise NotImplementedError("Func not implemented")

    def evaluate_quality(
        self,
        dataset_path,
        *args,
        preprocess_func=None,
        detailed_metrics=False,
        labels=None,
        partial_match_metrics=False,
        max_hierarchy_levels=3,
        **kwargs,
    ):
        """Run quality evaluation for instance of module

        Args:
            dataset_path (str): Path to where the input "gold set" dataset
                lives. Most often this is .json file.
            preprocess_func (method): Function used as proxy for any preliminary
                steps that need to be taken to run the model on the input text.
                This helper function ultimately leads to the input to this
                module and may involve executing other modules.
            detailed_metrics: boolean (Optional, defaults to False)
                Only for 'keywords'. Include partial scores and scores over every text in document.
            labels: list (Optional, defaults to None)
                Optional list of class labels to evaluate quality on. By default evaluation is done
                over all class labels. Using this, you can explicitly mention only a subset of
                labels to include in the quality evaluation.
            partial_match_metrics: boolean (Optional, defaults to False)
                Include partial match micro avg F1.
            max_hierarchy_levels (int): Used in hierarchical multilabel
                multiclass evaluation only. The number of levels in the
                hierarchy to run model evaluation on, in addition to complete
                matches.
            *args, **kwargs: Optional arguments which can be used by goldset/prediction
                set extraction.
                Nonekeyword arguments: `block_level`: str
                    For any module that has pre processing steps in the
                    middle of raw text and actual module input, use the input from gold standard
                    labels instead of a pre-process function. Useful for measuring quality for the
                    'block' alone (instead of the module + pre_process pipeline)
        Returns:
            dict: Dictionary of results provided by the `self.evaluator.run`
                function, depending on the associated `evaluation_type`. Reports
                things like precision, recall, and f1.
        """
        # 1) load dataset
        dataset = self._load_evaluation_dataset(dataset_path)

        # 2) verify dataset
        error.type_check("<COR14030040E>", collections.abc.Iterable, dataset=dataset)

        # 3) extract gold set predictions
        # pylint: disable=assignment-from-no-return
        gold_set = self._extract_gold_set(dataset)
        gold_annos = self._extract_gold_annotations(gold_set)

        # 4) obtain pred set predictions
        # pylint: disable=assignment-from-no-return
        pred_set = self._extract_pred_set(
            dataset, preprocess_func=preprocess_func, *args, **kwargs
        )
        pred_annos = self._extract_pred_annotations(pred_set)

        # 5) initialize evaluator
        # pylint: disable=not-callable
        evaluator = self.evaluator(gold_annos, pred_annos)

        # 6) run evaluator
        results = evaluator.run(
            self.evaluation_type,
            self.find_label_func,
            self.find_label_data_func,
            detailed_metrics,
            labels,
            partial_match_metrics,
            max_hierarchy_levels,
        )

        # 7) generate report
        report = self._generate_report(results, gold_set)

        # 8) return report
        return report

    ## Implementation Details ##################################################

    @staticmethod
    def _is_expandable_iterable(arg):
        """Check to see if something is a list / tuple of data model objects or strings. If it is,
        we consider it "expandable", meaning that one element of the iterable to one run call. In
        contrast, if something is not expandable, it will be passed as is to each call.

        Args:
            arg (any): Argument to run_batch being considered.
        Returns:
            bool: True if the argument is a compatible iterable, False
                otherwise.
        """
        # Throw if generators are passed - can't imagine any situation (for now) where this is
        # something that someone is doing on purpose, so we are a bit specific about this error.
        if isinstance(arg, types.GeneratorType):
            error(
                "<COR28071103E>",
                ValueError("Generator types are incompatible with .run_batch"),
            )
        if isinstance(arg, DataStream):
            error(
                "<COR75305604E>",
                ValueError("Data streams are incompatible with .run_batch"),
            )
        if isinstance(arg, (tuple, list)):
            return all(isinstance(obj, (str, DataBase)) for obj in arg)
        return False

    def _validate_and_extract_batch_size(self, *args, **kwargs):
        """Check to ensure that there's at least one iterable whose length is well defined,
        i.e., no generators, and that if multiple iterable arg/kwarg values are provided,
        they are all the same length.

        Args:
            *args: Variable length argument list to be passed directly to run().
            **kwargs: Arbitrary keyword arguments to be passed directly to run().
        Returns:
            int: Inferred batch size based on expandable iterables.
        """
        batch_size = None
        for _, arg in enumerate(args):
            batch_size = self._validate_arg_and_verify_batch_size(arg, batch_size)
        for _, arg_value in kwargs.items():
            batch_size = self._validate_arg_and_verify_batch_size(arg_value, batch_size)
        if batch_size is None:
            error("<COR36507545E>", ValueError("No iterable passed to batch predict"))
        return batch_size

    def _validate_arg_and_verify_batch_size(self, val, current_batch_size):
        """Check an arg value from args/kwargs. If we find that it's an expandable iterable, see
        if it conflicts with what we know about the inferred batch size so far.

        args:
            val (any): Argument / keyword argument value being inspected.
            current_batch_size (None | int): Current inferred batch size from
                previous args/kwargs, or None if no inferences have been made on
                other expandable iterables yet.
        Returns:
            None | inferred batch size.
        """
        if self._is_expandable_iterable(val):
            iter_batch_size = len(val)
            # Set the batch size if it's not set already. Raise if we have conflicting iterator
            # sizes. This will happen if the arg of a module run call has an iterable value. In
            # such cases, the subclass should override run_batch.
            if current_batch_size is None:
                return iter_batch_size
            if current_batch_size != iter_batch_size:
                err_str = "Iterables of varying length may not be passed to default batch predict."
                error("<COR98338604E>", ValueError(err_str))
        if current_batch_size:
            return current_batch_size

    @staticmethod
    def _build_args_for_default_run_with_batch(fixed_args, expanded_args, idx):
        """Build the non keyword arguments for run_batch's default implementation by expanding
        iterable args where possible, and grouping them with repeated noniterable arguments. The
        index correspondes to the current document under consideration.

        Args:
            fixed_args (dict): Noniterable args - common across all documents.
            expanded_args (dict): Iterable args - we'll need to index into this
                to get our doc arg.
            idx (int): Index of the document being considered.
        Returns:
            list: Args to be run for document [idx].
        """
        constructed_args = []
        if not expanded_args and not fixed_args:
            return constructed_args
        # Keys in arg dicts are positional arg number - get the biggest number arg we have
        max_exarg_idx = 0 if not expanded_args.keys() else max(expanded_args.keys())
        max_fixarg_idx = 0 if not fixed_args.keys() else max(fixed_args.keys())
        arg_count = max(max_exarg_idx, max_fixarg_idx)
        for arg_no in range(arg_count + 1):
            if arg_no in fixed_args:
                constructed_args.append(fixed_args[arg_no])
            elif arg_no in expanded_args:
                try:
                    constructed_args.append(expanded_args[arg_no][idx])
                except IndexError:
                    error(
                        "<COR94219537E>",
                        ValueError("idx {} exceeds extracted batch size".format(idx)),
                    )
            else:
                error(
                    "<COR68021708E>",
                    ValueError(
                        "arg number [{}] is missing from run_batch call".format(arg_no)
                    ),
                )
        return constructed_args

    @staticmethod
    def _build_kwargs_for_default_run_with_batch(fixed_kwargs, expanded_kwargs, idx):
        """Similar to the previous function, but for kwargs. Note that we can just clone our fixed
        kwargs instead of cycling through them, because order doesn't matter here.

        Args:
            fixed_args (dict): Noniterable valued kwargs - common across all
                documents.
            expanded_args (dict): Iterable valued kwargs - we'll need to index
                into these to get our doc kwarg.
        Returns:
            dict: Kwargs to be run for document [idx].
        """
        constructed_kwargs = fixed_kwargs.copy()
        for arg_name, iterable_arg_val in expanded_kwargs.items():
            try:
                constructed_kwargs[arg_name] = iterable_arg_val[idx]
            except IndexError:
                error(
                    "<COR51120124E>",
                    ValueError("idx {} exceeds extracted batch size".format(idx)),
                )
        return constructed_kwargs

    def _extract_gold_set(self, dataset):
        """Method for extracting gold set from dataset. Implemented in subclass.

        Args:
            dataset (object): In-memory version of whatever is loaded from on-
                disk. May be json, txt, etc.

        Returns:
            list: List of labels in the format of the module_type that is being
                called.
        """
        error(
            "<COR01455940E>",
            NotImplementedError("This is not available in this module."),
        )

    def _extract_pred_set(self, dataset, *args, preprocess_func=None, **kwargs):
        """Method for extracting pred set from dataset. Implemented in subclass.

        Args:
            dataset (object): In-memory version of whatever is loaded from on-
                disk. May be json, txt, etc.
            preprocess_func (method): Function used as proxy for any preliminary
                steps that need to be taken to run the model on the input text.
                This helper function ultimately leads to the input to this
                module and may involve executing other modules.
            *args, **kwargs (dict): Optional keyword arguments for prediction set extraction.
        Returns:
            list: List of labels in the format of the module_type that is being
                called.
        """
        error(
            "<COR95693719E>",
            NotImplementedError("This is not available in this module."),
        )

    @staticmethod
    def _load_evaluation_dataset(dataset_path):
        """Helper specifically for dataset loading.

        Args:
            dataset_path (str): Path to where the input 'gold set' dataset
                lives. Most often this is .json file.

        Returns:
            object: list, dict, or other python object, depending on the input
                dataset_path extension. Currently only supports `.json` and uses
                fileio from toolkit.
        """
        error.type_check("<COR33285197E>", str, dataset_path=dataset_path)

        if dataset_path.endswith(".json"):
            return fileio.load_json(dataset_path)

        # if all else fails
        error(
            "<COR81451234E>",
            ValueError("Unsure of how to load: {0}".format(dataset_path)),
        )

    @staticmethod
    def _extract_gold_annotations(gold_set):
        """Extract the core list of annotations that is needed for quality evaluation

        Args:
            gold_set (list)
        Returns:
            gold_annotations: list
        """
        return gold_set

    @staticmethod
    def _extract_pred_annotations(pred_set):
        """Extract the core list of predictions that is needed for quality evaluation

        Args:
            pred_set (list)
        Returns:
            pred_annotations: list
        """
        return pred_set

    @staticmethod
    def _generate_report(report, gold_set):
        """Generate the quality report output
        Args:
            report (dict)
            gold_set (list(dict))
        """
        return report
