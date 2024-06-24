# Standard
import os
import shutil

# Third Party
import grpc

# First Party
import alog

# Local
from caikit.runtime.model_management.model_manager import ModelManager
from caikit.runtime.names import MODEL_MESH_MODEL_ID_KEY
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
    def build_context(model_id="test-any-model-id", **metadata):
        """Build a gRPC context object containing the specified model ID

        Args:
            model_id(string): The model ID

        Returns:
            context(grpc.ServicerContext): Context object with mm-model-id
        """

        # Create a dummy class for mimicking ServicerContext invocation
        # metadata storage
        class TestContext(grpc.ServicerContext):
            def __init__(self, model_id):
                self.model_id = model_id
                self.metadata = metadata
                self.metadata[MODEL_MESH_MODEL_ID_KEY] = self.model_id
                self.callbacks = []
                self.canceled = False

            # Define the abstract methods to do nothing
            def abort(self, *_, **__):
                pass

            def abort_with_status(self, *_, **__):
                pass

            def auth_context(self, *_, **__):
                pass

            def is_active(self, *_, **__):
                pass

            def peer(self, *_, **__):
                pass

            def peer_identities(self, *_, **__):
                pass

            def peer_identity_key(self, *_, **__):
                pass

            def send_initial_metadata(self, *_, **__):
                pass

            def set_code(self, *_, **__):
                pass

            def set_details(self, *_, **__):
                pass

            def set_trailing_metadata(self, *_, **__):
                pass

            def time_remaining(self, *_, **__):
                pass

            def invocation_metadata(self):
                return list(self.metadata.items())

            def add_callback(self, some_function, *args, **kwargs):
                self.callbacks.append(
                    {"func": some_function, "args": args, "kwargs": kwargs}
                )
                # Only return true if the call has not yet canceled
                return not self.canceled

            def cancel(self):
                self.canceled = True
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
