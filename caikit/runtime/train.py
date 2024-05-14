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
This module is a central entrypoint for running a single synchronous training
job using caikit.core.train
"""

# Standard
from typing import Type
import argparse
import importlib
import json
import os
import sys
import traceback

# Third Party
from google.protobuf import json_format

# First Party
import alog

# Local
from ..core import ModuleBase, train
from ..core.data_model import TrainingStatus
from ..core.exceptions import error_handler
from ..core.registries import module_registry
from ..core.toolkit.logging import configure as config_logging
from .names import get_service_package_name
from .service_factory import ServicePackageFactory
from .utils.servicer_util import build_caikit_library_request_dict

log = alog.use_channel("TRAIN")
error = error_handler.get(log)

# The USER_ERROR_EXIT_CODE will be thrown when the process must exit
# as result of a user input error. User-related errors should be
# >= 1 and <=127 due to how some kubernetes operators interpret them.
USER_ERROR_EXIT_CODE = 1
# The INTERNAL_ERROR_EXIT_CODE will be thrown when training
# abnormally terminates, and it is not clearly fault of the user.
# System-level errors should be >= 128 and <= 254
INTERNAL_ERROR_EXIT_CODE = 203


class ArgumentParserError(Exception):
    """Custom exception class for ArgumentParser errors."""


class TrainArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        """Error handler that raises an exception instead of exiting."""
        raise ArgumentParserError(f"{self.prog}: error: {message}")


def write_termination_log(text: str, log_file="/dev/termination-log"):
    try:
        with open(log_file, "a") as handle:
            handle.write(text)
    except Exception as e:
        log.warning(
            "<COR96300323W>",
            "Unable to write termination log due to error %s",
            e,
        )


def main() -> int:
    """Main entrypoint for running training jobs"""
    parser = TrainArgumentParser(description=__doc__)

    # Required Args
    parser.add_argument(
        "--training-kwargs",
        "-k",
        required=True,
        help="Json string or json file pointer with keyword args for the training job",
    )
    parser.add_argument(
        "--module",
        "-m",
        required=True,
        help="Module name (package.Class) or UID to train",
    )
    parser.add_argument(
        "--model-name",
        "-n",
        required=True,
        help="Name to save the model under",
    )

    # Optional args
    parser.add_argument(
        "--save-path",
        "-s",
        default=".",
        help="Path to save the output model to",
    )
    parser.add_argument(
        "--library",
        "-l",
        nargs="*",
        help="Libraries that need to be imported to register the module to train",
    )
    parser.add_argument(
        "--trainer",
        "-t",
        default=None,
        help="Trainer config name to use",
    )
    parser.add_argument(
        "--save-with-id",
        "-i",
        action="store_true",
        default=False,
        help="Include the training ID in the save path",
    )
    parser.add_argument(
        "--termination-log-file",
        "-f",
        action="store_true",
        default="/dev/termination-log",
        help="Location of where to write a termination error message",
    )

    try:
        args = parser.parse_args()
        config_logging()

        # Initialize top-level kwargs
        train_kwargs = {
            "save_path": args.save_path,
            "save_with_id": args.save_with_id,
            "model_name": args.model_name,
        }
        if args.trainer is not None:
            train_kwargs["trainer"] = args.trainer

    except Exception as e:
        message = f"Exception raised during training. This may be a problem with your input: {e}"
        log.warning(
            {
                "log_code": "<COR39662029E>",
                "message": message,
                "stack_trace": traceback.format_exc(),
            },
            exc_info=True,
        )
        write_termination_log(message)
        exit(USER_ERROR_EXIT_CODE)

    # Import libraries to register modules
    try:
        for library in args.library or []:
            log.info("<COR88091092I>", "Importing library %s", library)
            importlib.import_module(library)
    except Exception:
        message = "Unable to import module {}".format(library)
        log.warning(
            {
                "log_code": "<COR17776539E>",
                "message": message,
                "stack_trace": traceback.format_exc(),
            },
            exc_info=True,
        )
        write_termination_log(message)
        exit(USER_ERROR_EXIT_CODE)

    # Try to import the root library of the provided module. It's ok if this
    # fails since the module may be a UID
    try:
        mod_root_lib = args.module.split(".")[0]
        importlib.import_module(mod_root_lib)
    except (ImportError, ValueError):
        log.debug("Unable to import module root lib: %s", mod_root_lib)

    # Figure out the module to train
    try:
        mod_reg = module_registry()
        mod_pkg_to_mod = {
            f"{mod.__module__}.{mod.__name__}": mod for mod in mod_reg.values()
        }
        module: Type[ModuleBase] = mod_reg.get(
            args.module, mod_pkg_to_mod.get(args.module)
        )
        error.value_check(
            "<COR03876205E>",
            module is not None,
            "Unable to find module {} to train",
            args.module,
        )
    except Exception:
        message = "Unable to find module {} to train".format(args.module)
        log.warning(
            {
                "log_code": "<COR17476539E>",
                "message": message,
                "stack_trace": traceback.format_exc,
            },
            exc_info=True,
        )
        write_termination_log(message)
        exit(USER_ERROR_EXIT_CODE)

    # Read training kwargs
    try:
        if os.path.isfile(args.training_kwargs):
            with open(args.training_kwargs, encoding="utf-8") as handle:
                training_kwargs = json.load(handle)
        else:
            training_kwargs = json.loads(args.training_kwargs)

        # Convert datatypes to match the training API
        training_service = ServicePackageFactory.get_service_package(
            ServicePackageFactory.ServiceType.TRAINING,
        )
        train_rpcs = [
            rpc
            for rpc in training_service.caikit_rpcs.values()
            if rpc.module_list == [module]
        ]
        error.value_check(
            "<COR11978965E>",
            len(train_rpcs) == 1,
            "Unable to find a unique train signature",
        )
        package_name = get_service_package_name(
            ServicePackageFactory.ServiceType.TRAINING
        )
        train_rpc_req = (
            train_rpcs[0].create_request_data_model(package_name).get_proto_class()
        )
        request_proto = json_format.Parse(
            json.dumps({"parameters": training_kwargs}),
            train_rpc_req(),
        )
        req_kwargs = build_caikit_library_request_dict(
            request_proto.parameters, module.TRAIN_SIGNATURE
        )
        train_kwargs.update(req_kwargs)
        log.debug3("All train kwargs: %s", train_kwargs)
    except json.decoder.JSONDecodeError:
        message = "training-kwargs must be valid json or point to a valid json file"
        log.warning(
            {
                "log_code": "<COR65834760E>",
                "message": message,
                "stack_trace": traceback.format_exc(),
            },
            exc_info=True,
        )
        write_termination_log(message)
        exit(USER_ERROR_EXIT_CODE)
    except ValueError as e:
        message = f"Invalid value for one or more input parameters: {e}"
        log.warning(
            {
                "log_code": "<COR65474760E>",
                "message": message,
                "stack_trace": traceback.format_exc(),
            },
            exc_info=True,
        )
    except Exception:
        message = "Exception encountered when attempting to parse input parameters"
        log.warning(
            {
                "log_code": "<COR17776549E>",
                "message": message,
                "stack_trace": traceback.format_exc(),
            },
            exc_info=True,
        )
        write_termination_log(message)
        exit(USER_ERROR_EXIT_CODE)

    try:
        # Run the training
        with alog.ContextTimer(
            log.info,
            "Finished training %s in: ",
            args.model_name,
        ):
            future = train(module, wait=True, **train_kwargs)
            info = future.get_info()
            if info.status == TrainingStatus.COMPLETED:
                log.info("Training finished successfully")
                return 0
            else:
                log.error("Training finished unsuccessfully")
                for err in info.errors or []:
                    log.error(err)
                write_termination_log("Training finished unsuccessfully")
                exit(INTERNAL_ERROR_EXIT_CODE)
    except MemoryError:
        message = "OOM error during training"
        log.warning(
            {
                "log_code": "<COR04280062E>",
                "message": message,
                "stack_trace": traceback.format_exc(),
            },
            exc_info=True,
        )
        write_termination_log(message)
        exit(INTERNAL_ERROR_EXIT_CODE)
    except Exception:
        message = "Unhandled exception during training"
        log.warning(
            {
                "log_code": "<COR04280062E>",
                "message": message,
                "stack_trace": traceback.format_exc(),
            },
            exc_info=True,
        )
        write_termination_log(message)
        exit(INTERNAL_ERROR_EXIT_CODE)


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
