# Standard
import os
import shutil

# First Party
import alog

# Local
from caikit.runtime.model_management.model_manager import ModelManager
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException

log = alog.use_channel("TEST-FIXTURE")


class Fixtures:
    """
    This class contains test fixtures that can be used to
    perform model manager operations with input models for tests
    """

    @staticmethod
    def load_model(model_id, local_model_path, model_type):
        """Load a model using model manager load model implementation"""
        try:
            model_manager = ModelManager.get_instance()
            model_manager.load_model(model_id, local_model_path, model_type)
        except CaikitRuntimeException as e:
            log.error({"message": e.message, "model_id": model_id, "error_id": e.id})
            raise e

    @staticmethod
    def unload_all_models():
        """Unload all loaded models using model manager unload models implementation"""
        try:
            model_manager = ModelManager.get_instance()
            model_manager.unload_all_models()
        except CaikitRuntimeException as e:
            log.error({"message": e.message, "error_id": e.id})
            raise e

    @staticmethod
    def build_context(model_id="test-any-model-id"):
        """Build a gRPC context object containing the specified model ID

        Args:
            model_id(string): The model ID

        Returns:
            context(grpc.ServicerContext): Context object with mm-model-id
        """

        # Create a dummy class for mimicking ServicerContext invocation
        # metadata storage
        class TestContext:
            def __init__(self, model_id):
                self.model_id = model_id
                self.callbacks = []

            def invocation_metadata(self):
                return [("mm-model-id", self.model_id)]

            def add_callback(self, some_function, *args, **kwargs):
                self.callbacks.append(
                    {"func": some_function, "args": args, "kwargs": kwargs}
                )
                return True

            def cancel(self):
                [f["func"](*f["args"], **f["kwargs"]) for f in self.callbacks]

        return TestContext(model_id)

    @staticmethod
    def get_good_model_path():
        return os.path.join(os.path.dirname(__file__), "models", "foo")

    @staticmethod
    def get_other_good_model_path():
        return os.path.join(os.path.dirname(__file__), "models", "bar")

    @staticmethod
    def get_good_model_archive_path():
        return os.path.join(os.path.dirname(__file__), "models", "foo_archive.zip")

    @staticmethod
    def get_bad_model_archive_path():
        return os.path.join(os.path.dirname(__file__), "models", "bad_model.zip")

    @staticmethod
    def get_invalid_model_archive_path():
        return os.path.join(os.path.dirname(__file__), "models", "bad_archive.zip")

    @staticmethod
    def get_good_model_type():
        return "fake_module"
