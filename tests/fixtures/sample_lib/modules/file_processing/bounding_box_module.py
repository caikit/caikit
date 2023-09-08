"""
A sample module for sample things!
"""
# Standard

# Local
from ...data_model.sample import FileDataType, FileTask
from caikit.core.modules import ModuleLoader
import caikit.core


@caikit.core.module(
    "750cad4d-c3b8-4327-b52e-e772f0d6f311", "BoundingBoxModule", "0.0.1", FileTask
)
class BoundingBoxModule(caikit.core.ModuleBase):
    def run(
        self,
        unprocessed: FileDataType,
    ) -> FileDataType:
        filename = f"processed_{unprocessed.filename}"
        data = b"bounding|" + unprocessed.data + b"|box"
        return FileDataType(filename, data)

    @classmethod
    def load(cls, model_path, **kwargs):
        loader = ModuleLoader(model_path)
        config = loader.config
        return cls()
