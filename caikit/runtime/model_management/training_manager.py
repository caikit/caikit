# Standard
from concurrent.futures import Future
from typing import Dict

# Third Party
import grpc

# Local
from caikit.interfaces.runtime.data_model import TrainingStatus
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException


class TrainingManager:
    __instance: "TrainingManager" = None

    training_futures: Dict[str, Future]

    def __init__(self):
        # Re-instantiating this is a programming error
        assert self.__class__.__instance is None, "This class is a singleton!"
        TrainingManager.__instance = self

        self.training_futures = {}

    def get_training_status(self, model_id: str) -> TrainingStatus:
        if model_id not in self.training_futures:
            raise CaikitRuntimeException(
                grpc.StatusCode.NOT_FOUND,
                f"{model_id} not found in the list of currently running training jobs",
            )
        future = self.training_futures[model_id]

        if future.running():
            return TrainingStatus.PROCESSING

        if future.done():
            if future.exception() is not None:
                return TrainingStatus.FAILED
            return TrainingStatus.COMPLETED

        raise RuntimeError("Unexpected error")

    @classmethod
    def get_instance(cls) -> "TrainingManager":
        """This method returns the instance of Training Manager"""
        if not cls.__instance:
            cls.__instance = TrainingManager()
        return cls.__instance
