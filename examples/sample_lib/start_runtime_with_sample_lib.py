# Standard
from pathlib import Path
import json
import os
import shutil
import sys
import tempfile

# First Party
import alog

# Local
from caikit.config.config import get_config
from caikit.runtime.__main__ import main
from caikit.runtime.dump_services import dump_grpc_services, dump_http_services
import caikit

if __name__ == "__main__":
    try:
        with tempfile.TemporaryDirectory() as workdir:
            caikit.config.configure(
                config_dict={
                    "merge_strategy": "merge",
                    "runtime": {
                        "library": "sample_lib",
                        "local_models_dir": workdir,
                        "lazy_load_local_models": True,
                        "grpc": {"enabled": True},
                        "http": {"enabled": True},
                        "training": {"save_with_id": False, "output_dir": workdir},
                        "service_generation": {
                            "package": "caikit_sample_lib"
                        },  # This is done to avoid name collision with Caikit itself
                    },
                }
            )
            # Make sample_lib available for import
            sys.path.append(
                os.path.join(Path(__file__).parent.parent.parent, "tests/fixtures"),
            )

            # Dump protos
            shutil.rmtree("protos", ignore_errors=True)
            if get_config().runtime.grpc.enabled:
                dump_grpc_services("protos", True, True)
            if get_config().runtime.http.enabled:
                dump_http_services("protos")

            # create a sample.json file for training
            with open(
                os.path.join("protos", "sample.json"), "w", encoding="utf-8"
            ) as handle:
                handle.write(
                    json.dumps(
                        [
                            {"number": 1},
                            {"number": 2},
                        ]
                    )
                )

            alog.configure(default_level="debug")
            main()
    finally:
        # remove generated protos
        shutil.rmtree("protos", ignore_errors=True)
        os.remove("modules.json")
