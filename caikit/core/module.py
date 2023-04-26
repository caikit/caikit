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

"""Shared functionality and interfaces used by *all* modules, i.e., all blocks and workflows.
"""

# TODO: Review to see if the lines in this file can be reduced or
# broken out into another file (e.g. Python sub-modules)
# pylint: disable=too-many-lines

# abstract methods may define arguments but not use them
# pylint: disable=unused-argument

# Standard
from importlib import metadata
from io import BytesIO
from typing import Any, Dict, List, Optional, Type, Union
import collections
import datetime
import os
import shutil
import tempfile
import time
import types
import uuid

# Third Party
import semver

# First Party
import alog

# Local
from .. import core
from . import data_model as dm
from .data_model import DataStream
from .module_backends import backend_types
from .module_config import ModuleConfig
from .module_meta import _ModuleBaseMeta
from .toolkit import ObjectSerializer, fileio
from .toolkit.errors import DataValidationError, error_handler
from .toolkit.wip_decorator import TempDisableWIP, WipCategory, work_in_progress
from caikit.config import get_config

log = alog.use_channel("MODULE")
error = error_handler.get(log)

# This file is `*` imported so we need to override what is exposed
__all__ = [
    "_MODULE_TYPES",
    "MODULE_BACKEND_REGISTRY",
    "MODULE_REGISTRY",
    "ModuleBase",
    "ModuleLoader",
    "ModuleSaver",
]

# This private registry is used to define types of modules that have been
# defined using the @module_type decorator
_MODULE_TYPES = []

# Single base global registry of all modules
MODULE_REGISTRY = {}

# Global module backend registry dict
MODULE_BACKEND_REGISTRY = {}

# This is used by individual caikit.module implementations, like a block
# to define what backend type models they can support at load time.
SUPPORTED_LOAD_BACKENDS_VAR_NAME = "SUPPORTED_LOAD_BACKENDS"


class ModuleBase(metaclass=_ModuleBaseMeta):
    """Abstract base class from which all Blocks and Workflows should inherit."""

    # will be used to evaluate blocks; defined in sub-classes
    evaluation_type = None
    # first arg is "self", unfortunately; TODO: get rid of that somehow
    evaluator = None

    @staticmethod
    def find_label_func(*_args, **_kwargs):
        """Function used to extract "label" from a prediction/result of a block's .run method.
        Define if you wish to have more specific evaluation metrics. Implemented in subclass.
        """
        raise NotImplementedError("Func not implemented")

    @staticmethod
    def find_label_data_func(*_args, **_kwargs):
        """Function used to extract data belonging to class "label" from a prediction/result
        of a block's .run method. Define if you wish to have more specific evaluation metrics.
        Implemented in subclass.
        """
        raise NotImplementedError("Func not implemented")

    def __init__(self, *args, **kwargs):
        """Construct a new model."""
        # Set up an empty metadata dictionary, to be:
        # - populated with metadata from `config.yml` files on `load`, and
        # - saved back to `config.yml` files on `save`
        self._metadata = {}

    @property
    @work_in_progress(category=WipCategory.WIP)
    def metadata(self) -> Dict[str, Any]:
        """This module's metadata.

        Returns:
            Dict[str, Any]: A dictionary of this module's metadata

            TODO: Can this be a `ModuleConfig` object instead? (or aconfig.Config)?
        """
        if not hasattr(self, "_metadata") or self._metadata is None:
            self._metadata = {}
        return self._metadata

    @classmethod
    @alog.logged_function(log.debug)
    def bootstrap(cls, *args, **kwargs):
        """Bootstrap a block. This method can be used to initialize the block
        from artifacts created outside of a particular caikit library
        """
        error(
            "<COR92634438E>",
            NotImplementedError("This is not available in this module."),
        )

    @classmethod
    @alog.logged_function(log.debug)
    def load(cls, *args, **kwargs):
        """Load a model."""
        error(
            "<COR92634437E>",
            NotImplementedError("This is not available in this module."),
        )

    @classmethod
    def _load(cls, module_loader, *args, **kwargs):
        """Load a model."""
        error(
            "<COR88356566E>",
            NotImplementedError("This is not available in this module."),
        )

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
            tuple
                Iterable of prediction outputs, run as a batch.
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

    @staticmethod
    def _is_expandable_iterable(arg):
        """Check to see if something is a list / tuple of data model objects or strings. If it is,
        we consider it "expandable", meaning that one element of the iterable to one run call. In
        contrast, if something is not expandable, it will be passed as is to each call.

        Args:
            arg: any
                Argument to run_batch being considered.
        Returns:
            bool
                True if the argument is a compatible iterable, False otherwise.
        """
        # Throw if generators are passed - can't imagine any situation (for now) where this is
        # something that someone is doing on purpose, so we are a bit specific about this error.
        if isinstance(arg, types.GeneratorType):
            error(
                "<COR28071103E>",
                ValueError("Generator types are incompatible with .run_batch"),
            )
        if isinstance(arg, dm.DataStream):
            error(
                "<COR75305604E>",
                ValueError("Data streams are incompatible with .run_batch"),
            )
        if isinstance(arg, (tuple, list)):
            return all(isinstance(obj, (str, dm.base.DataBase)) for obj in arg)
        return False

    def _validate_and_extract_batch_size(self, *args, **kwargs):
        """Check to ensure that there's at least one iterable whose length is well defined,
        i.e., no generators, and that if multiple iterable arg/kwarg values are provided,
        they are all the same length.

        Args:
            *args: Variable length argument list to be passed directly to run().
            **kwargs: Arbitrary keyword arguments to be passed directly to run().
        Returns:
            int
                Inferred batch size based on expandable iterables.
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
            val: any
                Argument / keyword argument value being inspected.
            current_batch_size: None | int
                Current inferred batch size from previous args/kwargs, or None if no inferences
                have been made on other expandable iterables yet.
        Returns:
            None | inferred batch size.
        """
        if self._is_expandable_iterable(val):
            iter_batch_size = len(val)
            # Set the batch size if it's not set already. Raise if we have conflicting iterator
            # sizes. This will happen if the arg of a block run call has an iterable value. In
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
            fixed_args: dict
                Noniterable args - common across all documents.
            expanded_args: dict
                Iterable args - we'll need to index into this to get our doc arg.
            idx: int
                Index of the document being considered.
        Returns:
            list
                Args to be run for document [idx].
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
            fixed_args: dict
                Noniterable valued kwargs - common across all documents.
            expanded_args: dict
                Iterable valued kwargs - we'll need to index into these to get our doc kwarg.
        Returns:
            dict
                Kwargs to be run for document [idx].
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

    def stream(self, data_stream, *args, **kwargs):
        """Lazily evaluate a run() on a given model by constructing a new data stream generator
        from the results. Note that we do not allow datastreams in args/kwargs. In rare cases,
        this may mean that stream() is not available, e.g., for keywords extraction. In these
        cases, stream() should be overridden in the subclass (module implementation) to allow
        and expand along multiple data streams.

        Args:
            data_stream: caikit.core.data_model.DataStream
                Datastream to be lazily sequentially processed by the module under consideration.
            *args: Variable length argument list to be passed directly to run().
            **kwargs: Arbitrary keyword arguments to be passed directly to run().
        Returns:
            protobufs
                A DataBase object.
        """
        error.type_check("<COR98214589E>", dm.DataStream, data_stream=data_stream)
        # Ensure that no args/kwargs are DataStreams, since these get passed to stream()
        run_argvals = list(args) + list(kwargs.values())
        if any(isinstance(arg, dm.DataStream) for arg in run_argvals):
            error(
                "<COR28828273E>",
                ValueError(
                    "Only one DataStream may be passed when invoking module stream()"
                ),
            )
        # TODO: Add .run_batch() integration
        return dm.DataStream(
            lambda: (self.run(data_item, *args, **kwargs) for data_item in data_stream)
        )

    @alog.logged_function(log.debug)
    def save(self, model_path, *args, **kwargs):
        """Save a model.

        Args:
            model_path: str
                Path on disk to export the model to.
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
            *args, **kwargs: dict
                Optional keyword arguments for saving.
        Returns:
            io.BytesIO
                File like object holding an exported model in memory as a io.BytesIO object.
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
            *args, **kwargs: dict
                Optional keyword arguments for saving.
        Returns:
            bytes
                bytes object holding an exported model in memory.
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

    @classmethod
    @alog.logged_function(log.debug)
    def train(cls, *args, **kwargs):
        """Train a model."""
        error(
            "<COR44977721E>",
            NotImplementedError("This is not available in this module."),
        )

    def _extract_gold_set(self, dataset):
        """Method for extracting gold set from dataset. Implemented in subclass.

        Args:
            dataset:  object
                In-memory version of whatever is loaded from on-disk. May be json, txt, etc.

        Returns:
            list
                List of labels in the format of the block_type that is being called.
        """
        error(
            "<COR01455940E>",
            NotImplementedError("This is not available in this module."),
        )

    def _extract_pred_set(self, dataset, *args, preprocess_func=None, **kwargs):
        """Method for extracting pred set from dataset. Implemented in subclass.

        Args:
            dataset:  object
                In-memory version of whatever is loaded from on-disk. May be json, txt, etc.
            preprocess_func:  method
                Function used as proxy for any preliminary steps that need to be taken to run the
                model on the input text. This helper function ultimately leads to the input to this
                block and may involve executing other blocks.
            *args, **kwargs: dict
                Optional keyword arguments for prediction set extraction.
        Returns:
            list
                List of labels in the format of the block_type that is being called.
        """
        error(
            "<COR95693719E>",
            NotImplementedError("This is not available in this module."),
        )

    @staticmethod
    def load_evaluation_dataset(dataset_path):
        """Helper specifically for dataset loading.

        Args:
            dataset_path:  str
                Path to where the input 'gold set' dataset lives. Most often this is .json file.

        Returns:
            object
                list, dict, or other python object, depending on the input dataset_path extension.
                Currently only supports `.json` and uses fileio from toolkit.
        """
        error.type_check("<COR33285197E>", str, dataset_path=dataset_path)

        if dataset_path.endswith(".json"):
            return fileio.load_json(dataset_path)

        # if all else fails
        error(
            "<COR81451234E>",
            ValueError("Unsure of how to load: {0}".format(dataset_path)),
        )

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
        """Run quality evaluation for instance of block or workflow.

        Args:
            dataset_path:  str
                Path to where the input "gold set" dataset lives. Most often this is .json file.
            preprocess_func:  method
                Function used as proxy for any preliminary steps that need to be taken to run the
                model on the input text. This helper function ultimately leads to the input to this
                block and may involve executing other blocks.
            detailed_metrics: boolean (Optional, defaults to False)
                Only for 'keywords'. Include partial scores and scores over every text in document.
            labels:  list (Optional, defaults to None)
                Optional list of class labels to evaluate quality on. By default evaluation is done
                over all class labels. Using this, you can explicitly mention only a subset of
                labels to include in the quality evaluation.
            partial_match_metrics: boolean (Optional, defaults to False)
                Include partial match micro avg F1.
            max_hierarchy_levels: int
                Used in hierarchical multilabel multiclass evaluation only. The number of levels in
                the hierarchy to run model evaluation on,
                in addition to complete matches.
            *args, **kwargs:
                Optional arguments which can be used by goldset/prediction set extraction.
                keyword arguments:
                `block_level`: str (Applicable to block_type relations)
                    For any block that has pre processing steps in the
                    middle of raw text and actual block input, use the input from gold standard
                    labels instead of a pre-process function. Useful for measuring quality for the
                    'block' alone (instead of the block + pre_process pipeline)
        Returns:
            dict
                Dictionary of results provided by the `self.evaluator.run` function, depending on
                the associated `evaluation_type`. Reports things like precision, recall, and f1.
        """
        # 1) load dataset
        dataset = self.load_evaluation_dataset(dataset_path)

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

    @staticmethod
    def _extract_gold_annotations(gold_set):
        """Extract the core list of annotations that is needed for quality evaluation

        Args:
            gold_set: list
        Returns:
            gold_annotations: list
        """
        return gold_set

    @staticmethod
    def _extract_pred_annotations(pred_set):
        """Extract the core list of predictions that is needed for quality evaluation

        Args:
            pred_set: list
        Returns:
            pred_annotations: list
        """
        return pred_set

    @staticmethod
    def _generate_report(report, gold_set):
        """Generate the quality report output
        Args:
            report: dict
            gold_set: list(dict)
        """
        return report

    @classmethod
    def timed_load(cls, *args, **kwargs):
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
        # get initial values
        start_time = time.time()
        # We are calling caikit.core over cls.load because we need to figure out
        # what instance the model belongs to
        model = core.load(*args, **kwargs)
        time_passed = time.time() - start_time
        return time_passed, model

    def timed_run(self, *args, num_seconds=None, num_iterations=None, **kwargs):
        """Time a number of runs over set seconds or iterations.

        Args:
            *args: list
                Will be passed to `self.run`.
            num_seconds:  int
                Minimum numer of seconds to run timed_run over. Will most likely be more than this
                value due to its waiting for the each call to `self.run` to finish.
            num_iterations:  int
                Minimum numer of iterations to run timed_run over. Will run exactly this many times.
            **kwargs:  dict
                Will be passed to `self.run`.

        Returns:
            int, int, caikit.core.data_model.DataBase
                The first return value is the total time spent in the `self.run` loop. The second
                return value is the total number of calls to `self.run` were made.
                The return value is the output of the block's run method

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

    def validate_loaded_model(self, *args):
        """Validate a loaded model."""
        error(
            "<COR56275627E>",
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


class ModuleLoader:
    def __init__(self, model_path):
        """Construct a new module loader.

        Args:
            model_path:  str
                The path to the directory where the model is to be loaded from.
        """
        self.model_path = os.path.normpath(model_path)
        error.dir_check("<COR43014802E>", model_path)
        self.config = ModuleConfig.load(model_path)

    def load_arg(self, arg):
        """Extract arg value from the loaded model's config"""
        return getattr(self.config, arg)

    def load_args(self, *args):
        """Extract values from the loaded model's config"""
        return tuple(getattr(self.config, arg) for arg in args)


class ModuleSaver:
    """A module saver that provides common functionality used for saving modules and also a context
    manager that cleans up gracefully in case an error is encountered during the save process.
    """

    SAVED_KEY_NAME = "saved"
    CREATED_KEY_NAME = "created"
    TRACKING_KEY_NAME = "tracking_id"
    MODULE_VERSION_KEY_NAME = "version"

    def __init__(self, module: ModuleBase, model_path):
        """Construct a new module saver.

        Args:
            module:  caikit.core.module.Module
                The instance of the module to be saved.
            model_path:  str
                The absolute path to the directory where the model will be saved.  If this directory
                does not exist, it will be created.
        """
        self.model_path = os.path.normpath(model_path)

        # Get possibly nested caikit library path
        module_path = module.__module__
        lib_name_generator = (
            k
            for k, v in get_config().libraries.items()
            if module_path.startswith(v.module_path)
        )
        try:
            self.library_name = next(lib_name_generator)
        except StopIteration:
            # This assumes no nested module path by default
            self.library_name = module_path.split(".")[0]  # tests

        try:
            self.library_version = metadata.version(self.library_name)
        except metadata.PackageNotFoundError:
            log.debug("<COR25991305D>", "No library version found")
            if (
                self.library_name in get_config().libraries
                and "version" in get_config().libraries[self.library_name]
            ):
                self.library_version = get_config().libraries[self.library_name].version
            else:
                self.library_version = "0.0.0"

        self.config = {
            self.library_name + "_version": self.library_version,
            self.CREATED_KEY_NAME: str(datetime.datetime.now()),
            self.SAVED_KEY_NAME: str(datetime.datetime.now()),
            "name": module.MODULE_NAME,
            self.TRACKING_KEY_NAME: str(uuid.uuid4()),
        }

        # Add the sub-type specific fields
        for subtype in _MODULE_TYPES:
            subtype_id = getattr(module, f"{subtype}_ID", None)
            if subtype_id is not None:
                self.config.update(
                    {
                        f"{subtype.lower()}_id": subtype_id,
                        f"{subtype.lower()}_class": getattr(module, f"{subtype}_CLASS"),
                        "version": getattr(module, f"{subtype}_VERSION"),
                    }
                )

        # Temp disable wip for following invocation to not log warnings for downstream
        # usage of ModuleSaver
        with TempDisableWIP():
            # Get metadata back about this module and add it to the config
            stored_config = module.metadata
        # Sanitize some things off of the config:
        # Remove the old `saved` timestamp:
        stored_config.pop(self.SAVED_KEY_NAME, None)
        # Remove any reserved keys, these will be set by the `ModuleConfig` class
        for key in ModuleConfig.reserved_keys:
            if key in stored_config:
                stored_config.pop(key)

        # Run some extremely silly metadata sanitization stuff to _not_ save metadata that was
        # explicitly removed from some certain modules
        ModuleSaver._provide_backwards_compatibility(module, stored_config)

        self.config.update(stored_config)

    @staticmethod
    def _provide_backwards_compatibility(
        module: ModuleBase, stored_config: Dict[str, Any]
    ) -> None:
        """Updates the stored_config in-place to remove any metadata keys that some certain
        existing models expect to _not_ exist"""

        # BERT entity mentions blocks no longer save the "type_map.json" file
        if module.MODULE_ID == "f7e4208f-daee-4c5d-8268-b010929dd247":
            stored_config.pop("type_map_path", None)

    def add_dir(self, relative_path, base_relative_path=""):
        """Create a directory inside the `model_path` for this saver.

        Args:
            relative_path:  str
                A path relative to this saver's `model_path` denoting the directory to create.
            base_relative_path:  str
                A path, relative to this saver's `model_path`, in which `relative_path` will be
                created.

        Returns:
            str, str
                A tuple containing both the `relative_path` and `absolute_path` to the
                directory created.

        Examples:
            >>> with ModelSaver('/path/to/model') as saver:
            >>>     rel_path, abs_path = saver.add_dir('word_embeddings', 'model_data')
            >>> print(rel_path)
            model_data/word_embeddings
            >>> print(abs_path)
            /path/to/model/model_data/word_embeddings
        """
        base_relative_path = os.path.normpath(base_relative_path)
        relative_path = os.path.normpath(relative_path)

        relative_path = os.path.join(base_relative_path, relative_path)
        absolute_path = os.path.join(self.model_path, relative_path)

        os.makedirs(absolute_path, exist_ok=True)

        return relative_path, absolute_path

    def copy_file(self, file_path, relative_path=""):
        """Copy an external file into a subdirectory of the `model_path` for this saver.

        Args:
            file_path:  str
                Absolute path to the external file to copy.
            relative_path:  str
                The relative path inside of `model_path` where the file will be copied to.
                If set to the empty string (default) then the file will be placed directly in
                the `model_path` directory.

        Returns:
            str, str
                A tuple containing both the `relative_path` and `absolute_path` to the copied file.
        """
        file_path = os.path.normpath(file_path)

        if not os.path.isfile(file_path):
            error(
                "<COR80954473E>",
                FileNotFoundError(
                    "Attempted to add `{}` but is not a regular file.".format(file_path)
                ),
            )

        filename = os.path.basename(os.path.normpath(file_path))

        relative_path, absolute_path = self.add_dir(relative_path)

        relative_file_path = os.path.join(relative_path, filename)
        absolute_file_path = os.path.join(absolute_path, filename)

        shutil.copyfile(file_path, absolute_file_path)

        return relative_file_path, absolute_file_path

    def save_object(self, obj, filename, serializer, relative_path=""):
        """Save a Python object using the provided ObjectSerializer.

        Args:
            obj:  any
                The Python object to save
            filename: str
                The filename to use for the saved object
            serializer: ObjectSerializer
                An ObjectSerializer instance (e.g., YAMLSerializer) that should be used to serialize
                the object
            relative_path:  str
                The relative path inside of `model_path` where the object will be saved
        """
        if not issubclass(serializer.__class__, ObjectSerializer):
            error(
                "<COR85655282E>",
                TypeError(
                    "`{}` does not extend `ObjectSerializer`".format(
                        serializer.__class__.__name__
                    )
                ),
            )

        relative_path, absolute_path = self.add_dir(relative_path)

        # Normalize any '././' structure that may come from relative paths
        relative_file_path = os.path.normpath(os.path.join(relative_path, filename))
        absolute_file_path = os.path.normpath(os.path.join(absolute_path, filename))

        serializer.serialize(obj, absolute_file_path)

        return relative_file_path, absolute_file_path

    def update_config(self, additional_config):
        """Add items to this saver's config dictionary.

        Args:
            additional_config:  dict
                A dictionary of config options to add the this saver's configuration.

        Notes:
            The behavior of this method matches `dict.update` and is equivalent to calling
            `saver.config.update`.  The `saver.config` dictionary may be accessed directly for
            more sophisticated manipulation of the configuration.
        """
        self.config.update(additional_config)

    def __enter__(self):
        """Enter the module saver context.  This creates the `model_path` directory.  If this
        context successfully exits, then the model configuration and all files it contains will
        be written and saved to disk inside the `model_path` directory.  If any uncaught exceptions
        are thrown inside this context, then `model_path` will be removed.
        """
        os.makedirs(self.model_path, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the module saver context. If this context successfully exits, then the model
        configuration and all files it contains will be written and saved to disk inside the
        `model_path` directory.  If any uncaught exceptions are thrown inside this context, then
        `model_path` will be removed.
        """
        if exc_type is not None:
            shutil.rmtree(self.model_path, ignore_errors=True)
            return

        ModuleConfig(self.config).save(self.model_path)


def register_module_implementation(
    implementation_class: type,
    backend_type: int,
    module_id: str,
    backend_config_override: Dict = None,
):
    """This function will register the mapping for the given module_id and
    backend_type to the implementation class

    Args:
        implementation_class:  type
            The class that is used to implement this backend type for the given
            module_id
        backend_type:  int
            Value from MODULE_BACKEND_TYPES that indicates the backend
            that this class implements
        module_id:  str
            The module_id from the caikit.core module registry that this class
            overloads
        backend_config_override: Dict
            Dictionary containing essential overrides for the backend config.
            This will get stored with the implementation_class class name and will automatically
            get picked up and merged with other such configs for a specific backend

    """

    backend_config_override = backend_config_override or {}

    log.debug(
        "Registering backend [%s] implementation for module [%s]",
        backend_type,
        module_id,
    )

    error.value_check(
        "<COR86780140E>",
        backend_type in backend_types.MODULE_BACKEND_TYPES,
        "Cannot override implementation of {} for unkonwn backend type {}",
        module_id,
        backend_type,
    )

    core_class = MODULE_REGISTRY.get(module_id)
    if core_class is None:
        # TODO! Inject a dummy entry that will raise on usage
        pass  # pragma: no cover

    # Do the registration!
    module_type_mapping = MODULE_BACKEND_REGISTRY.setdefault(module_id, {})

    # Make sure this is not an overwrite of an existing registration
    existing_type = module_type_mapping.get(backend_type)
    assert (
        existing_type is None or existing_type is implementation_class
    ), f"Registration conflict! ({module_id}, {backend_type}) already registered as {existing_type}"

    BackendConfig = collections.namedtuple(
        "BackendConfig", "impl_class backend_config_override"
    )

    module_type_mapping[backend_type] = BackendConfig(
        impl_class=implementation_class, backend_config_override=backend_config_override
    )


# pylint: disable=redefined-outer-name
def module_type(module_type):
    """This encapsulates the logic of creating a derived module subtype
    (e.g. block). It is intended to decorate a class which inherits from
    ModuleBase. The wrapped class is augmented in the following ways:

    * A new class-attribute is defined named `module_type` which is itself a
        decorator that concrete implementations of this module type can use to
        bind a module id, description, and version (e.g. @BlockBase.block).
    * A new class attribute is added as a global registry that the above
        decorator will use to store all registered concrete implementations of
        the module type

    These above class attributes can be further hoisted to free attributes in
    the python module where the module type is defined (e.g. @block).
    """

    # Add this module type to the global list of module types
    # pylint: disable=global-variable-not-assigned
    global _MODULE_TYPES

    module_type_name = module_type.upper()

    _MODULE_TYPES.append(module_type_name)

    def module_type_decorator(cls):
        # Perform top-level imports here for places that the the new module type
        # will need to be added. This is done to avoid circular dependencies
        # since these top-level imports are only needed where the decorator is
        # used.
        # Local
        # pylint C0415: Import outside toplevel (import-outside-toplevel)
        # pylint: disable=import-outside-toplevel
        from caikit.core import model_manager

        # Add the registry dict and decorate the top-level module
        registry_name = f"{module_type_name}_REGISTRY"
        impl_registry = {}
        error.value_check(
            "<COR68254470E>",
            not hasattr(core, registry_name),
            "Cannot re-declare module type {}",
            module_type,
        )
        setattr(core, registry_name, impl_registry)
        setattr(model_manager, registry_name, impl_registry)
        setattr(cls, "REGISTRY", impl_registry)

        # Define the module implementation decorator
        # pylint: disable=redefined-builtin,pointless-statement
        def _module_impl_decorator(
            id=None,
            name=None,
            version=None,
            backend_type=backend_types.LOCAL,
            base_module: Union[str, Type[ModuleBase]] = None,
            backend_config_override: Optional[Dict] = None,
        ):
            f"""Apply this decorator to any class that should be treated as a {module_type} (i.e.,
            extends
            `{cls.__name__}) and registered with caikit.core so that the library "knows" the class
            is a
            {module_type} and is capable of loading instances of the {module_type}.

            Args:
                id:  str
                    A UUID to use when registering this {module_type} with caikit.core
                    Not required if based on another caikit module using `base_module`
                name:  str
                    A human-readable name for the {module_type}
                    Not required if based on another caikit module using `base_module`
                version:  str
                    A SemVer for the {module_type}
                    Not required if based on another caikit module using `base_module`
                backend_type: backend_type
                    Associated backend type for the module.
                    Default: `LOCAL`
                base_module: str | ModuleBase
                    If this module is based on a different caikit module, provide name
                    of the base module.
                    Default: None
                backend_config_override: Dict
                    Dictionary containing configuration required for the specific backend.
                    Default: None

            Returns:
                A decorated version of the class to which it was applied, after registering the
                class as a valid {module_type} with caikit.core
            """
            base_module_class = None
            # Flag to store if the current module is a backend implementation
            # of an existing module or not
            backend_module_impl = False

            # No mutable default
            backend_config_override = backend_config_override or {}

            if any([id is None, version is None or name is None]):
                error.type_check(
                    "<COR87944440E>",
                    str,
                    type(ModuleBase),
                    allow_none=False,
                    base_module=base_module,
                )
                error.type_check(
                    "<COR60584425E>",
                    dict,
                    allow_none=True,
                    backend_config_override=backend_config_override,
                )

                # If the base_module is a string, assume that it is the module_id of the
                # base module
                if isinstance(base_module, str):
                    module_id = base_module
                    error.value_check(
                        "<COR09479833E>",
                        module_id in MODULE_REGISTRY,
                        "Unknown base module id: {}",
                        module_id,
                    )
                    base_module_class = MODULE_REGISTRY[module_id]

                # If base_module is a type, validate that it derives from ModuleBase and
                # use its MODULE_ID
                elif isinstance(base_module, type):
                    if not issubclass(base_module, ModuleBase):
                        error(
                            "<COR20161747E>",
                            f"base_module [{base_module}] does not derive from ModuleBase",
                        )

                    base_module_class = base_module

                # TODO: Add support for inheritance of backend implementation
                # i.e if a module inherits from base_module

                id = base_module_class.MODULE_ID
                version = base_module_class.MODULE_VERSION
                name = base_module_class.MODULE_NAME
                backend_module_impl = True

            error.type_check("<COR54118928E>", str, id=id, name=name, version=version)

            semver.VersionInfo.parse(version)  # Make sure this is a valid SemVer

            def decorator(cls_):
                # Verify this is a valid module type (inherits from the wrapped
                # base class)

                if backend_module_impl and not issubclass(base_module_class, cls):
                    error(
                        "<COR32401861E>",
                        TypeError(
                            f"`{base_module_class.__name__}` does not extend `{cls.__name__}`",
                        ),
                    )
                elif not backend_module_impl and not issubclass(cls_, cls):
                    error(
                        "<COR68265482E>",
                        TypeError(
                            f"`{cls_.__name__}` does not extend `{cls.__name__}`",
                        ),
                    )

                # Add attributes to the implementation class
                setattr(cls_, f"{module_type_name}_ID", id)
                cls_.MODULE_ID = id  # Module ID == Module Type ID
                setattr(cls_, f"{module_type_name}_NAME", name)
                cls_.MODULE_NAME = name  # Module Name == Module Type Name
                setattr(cls_, f"{module_type_name}_VERSION", version)
                cls_.MODULE_VERSION = version  # Module Version == Module Type Version
                classname = f"{cls_.__module__}.{cls_.__qualname__}"
                setattr(cls_, f"{module_type_name}_CLASS", classname)
                cls_.MODULE_CLASS = classname
                cls_.PRODUCER_ID = dm.ProducerId(cls_.MODULE_NAME, cls_.MODULE_VERSION)

                # Set module type as attribute of the class
                # pylint: disable=global-variable-not-assigned
                cls_.MODULE_TYPE = module_type_name

                # If no backend support described in the class, add current backend
                # as the only backend that can load models trained by this module
                cls_.SUPPORTED_LOAD_BACKENDS = getattr(
                    cls_, SUPPORTED_LOAD_BACKENDS_VAR_NAME, [backend_type]
                )

                # Set its own backend_type as an attribute
                setattr(cls_, "BACKEND_TYPE", backend_type)

                # Verify UUID and add this block to the module and block registries
                global MODULE_REGISTRY
                current_class = MODULE_REGISTRY.get(cls_.MODULE_ID)
                if not backend_module_impl:
                    if current_class is not None:
                        error(
                            "<COR30607646E>",
                            RuntimeError(
                                "BLOCK_ID `{}` conflicts for classes `{}` and `{}`".format(
                                    cls_.MODULE_ID,
                                    cls_.__name__,
                                    MODULE_REGISTRY[cls_.MODULE_ID].__name__,
                                )
                            ),
                        )
                    MODULE_REGISTRY[cls_.MODULE_ID] = cls_
                    impl_registry[cls_.MODULE_ID] = cls_

                # Register backend
                register_module_implementation(
                    cls_, backend_type, cls_.MODULE_ID, backend_config_override
                )

                return cls_

            return decorator

        # Add this decorator to the wrapped class
        setattr(cls, module_type, _module_impl_decorator)
        return cls

    return module_type_decorator
