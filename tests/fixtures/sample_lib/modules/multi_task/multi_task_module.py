# Local
from ...data_model.sample import OtherOutputType, SampleInputType, SampleOutputType
from caikit.core import TaskBase, module, task
from caikit.core.data_model import ProducerId
from caikit.interfaces.common.data_model import File
import caikit


@task(
    unary_parameters={"sample_input": SampleInputType},
    unary_output_type=SampleOutputType,
)
class FirstTask(TaskBase):
    pass


@task(
    unary_parameters={"file_input": File},
    unary_output_type=OtherOutputType,
)
class SecondTask(TaskBase):
    pass


@module(
    id="00110203-0123-0456-0789-0a0b02dd1eef",
    name="MultiTaskModule",
    version="0.0.1",
    tasks=[FirstTask, SecondTask],
)
class MultiTaskModule(caikit.core.ModuleBase):
    def __init__(self):
        pass

    @classmethod
    def load(cls, model_path, **kwargs):
        return cls()

    @FirstTask.taskmethod()
    def run_some_task(self, sample_input: SampleInputType) -> SampleOutputType:
        return SampleOutputType("Hello from FirstTask")

    @SecondTask.taskmethod()
    def run_other_task(self, file_input: File) -> OtherOutputType:
        return OtherOutputType(
            "Goodbye from SecondTask", ProducerId("MultiTaskModule", "0.0.1")
        )
