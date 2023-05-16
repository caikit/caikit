"""
Unit tests for the @module_type decorator
"""
# Standard
import abc
import copy
import uuid

# Third Party
import pytest

# Local
from caikit.core import module
from caikit.core.module_type import module_type
from sample_lib.modules.sample_task import SampleModule
from sample_lib.data_model.sample import SampleTask
import caikit.core

## Helpers #####################################################################


@pytest.fixture
def TestModBase():
    prev_module_types = copy.copy(module._MODULE_TYPES)

    @module_type("testmod")
    class TestModBase(module.ModuleBase):
        """Derived module type"""

    yield TestModBase

    module._MODULE_TYPES.clear()
    module._MODULE_TYPES.extend(prev_module_types)
    if hasattr(caikit.core, "TESTMOD_REGISTRY"):
        delattr(caikit.core, "TESTMOD_REGISTRY")


## Tests #######################################################################


def test_module_type_new_type(TestModBase):
    """Make sure that the newly declared module type can be used just like a
    first-class module type such as block
    """
    assert hasattr(caikit.core, "TESTMOD_REGISTRY")
    assert "TESTMOD" in module._MODULE_TYPES
    assert caikit.core.TESTMOD_REGISTRY == {}

    # Add a new derived testmod
    mod_id = str(uuid.uuid4())

    @TestModBase.testmod(
        id=mod_id, name="Sample tesmod", version="1.2.3", task=SampleTask
    )
    class SampleTestmod(TestModBase):
        """A sample test mod"""

    # Make sure that the test mod was added to the registry
    assert caikit.core.TESTMOD_REGISTRY == {mod_id: SampleTestmod}

    # Make sure module type is set as attribute of sample module
    assert hasattr(SampleTestmod, "MODULE_TYPE")
    assert SampleTestmod.MODULE_TYPE == "TESTMOD"


def test_module_type_missing_base_class(TestModBase):
    """Make sure that if a derived class misses the inheritance from the
    right base class, an exception is raised
    """
    mod_id = str(uuid.uuid4())
    with pytest.raises(TypeError):
        # pylint: disable=unused-variable
        @TestModBase.testmod(
            id=mod_id, name="Sample tesmod", version="1.2.3", task=SampleTask
        )
        class SampleBadTestmod:
            """A sample test mod that is missing the base class"""


def test_module_type_wrong_base_class(TestModBase):
    """Make sure that if a derived class inherits from the wrong module type
    an exception is raised
    """
    mod_id = str(uuid.uuid4())
    with pytest.raises(TypeError):
        # pylint: disable=unused-variable
        @TestModBase.testmod(
            id=mod_id, name="Sample tesmod", version="1.2.3", task=SampleTask
        )
        class SampleBadTestmod(caikit.core.BlockBase):
            """A sample test mod that is missing the base class"""


def test_module_no_reused_ids(TestModBase):
    """Make sure that module ids can't be reused, even across module types"""
    with pytest.raises(RuntimeError):

        @TestModBase.testmod(
            id=SampleModule.MODULE_ID,
            name="Sample tesmod",
            version="1.2.3",
            task=SampleTask,
        )
        # pylint: disable=unused-variable
        class SampleTestmod(TestModBase):
            """A sample test mod"""


def test_intermediate_metabase():
    """Make sure that an abc.ABC can be declared that derives from ModuleBase"""

    class Intermediate(caikit.core.ModuleBase, abc.ABC):
        """Sample intermediate base class"""

        @abc.abstractmethod
        def foobar(self, arg):
            """Got to define foobar!"""

    @caikit.core.block("asdf-qwer-zxcv", "FooBar", "0.0.1", SampleTask)
    class Derived(Intermediate):
        def foobar(self, arg):
            return arg + 1

    d = Derived()
    assert d.foobar(1) == 2
