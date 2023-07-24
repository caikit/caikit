# Standard
from pathlib import Path
import json
import os
import shutil
import sys

# Local
from caikit.runtime import grpc_server
from caikit.runtime.dump_services import dump_grpc_services

# Need to set env before importing grpc_server
os.environ["ENVIRONMENT"] = "test"

# Make sample_lib available for import
sys.path.append(
    os.path.join(Path(__file__).parent.parent, "tests/fixtures"),
)

# dump protos
shutil.rmtree("protos", ignore_errors=True)
dump_grpc_services("protos")

# create a sample.json file for training
with open(os.path.join("protos", "sample.json"), "w", encoding="utf-8") as handle:
    handle.write(
        json.dumps(
            [
                {"number": 1},
                {"number": 2},
            ]
        )
    )

grpc_server.main()

# remove generated protos
shutil.rmtree("protos", ignore_errors=True)
