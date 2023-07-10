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

# Standard
import logging
import re

# Third Party
import pytest

# Local
from caikit.core.toolkit import wip_decorator
from caikit.core.toolkit.wip_decorator import (
    Action,
    TempDisableWIP,
    WipCategory,
    disable_wip,
    enable_wip,
    work_in_progress,
)

## Helpers #####################################################################


@pytest.fixture(autouse=True)
def with_wip():
    prev_val = wip_decorator._ENABLE_DECORATOR
    enable_wip()
    yield
    wip_decorator._ENABLE_DECORATOR = prev_val


msg_re = "<.*> is still in the {} phase and subject to change!"
expected_msg_beta = msg_re.format(WipCategory.BETA.name)
expected_msg_wip = msg_re.format(WipCategory.WIP.name)

## Tests #######################################################################

######################## Test decorator on function #########################################


def test_default_config_fn_no_arg(caplog):
    """Test decorator with no config on function with no arguments"""

    @work_in_progress
    def foo():
        pass

    with caplog.at_level(logging.WARN):
        foo()
        assert any(
            re.search(expected_msg_wip, record.message) for record in caplog.records
        )


def test_default_config_fn_arg(caplog):
    """Test decorator with no config on function with arguments"""

    @work_in_progress
    def foo(x):
        return x

    argument = 1
    with caplog.at_level(logging.WARN):
        result = foo(argument)
        assert any(
            re.search(expected_msg_wip, record.message) for record in caplog.records
        )
        assert argument == result


def test_default_config_fn_kwarg(caplog):
    """Test decorator with no config on function with argument and keyword argument"""

    @work_in_progress
    def foo(x, y=0, z=0):
        return x + y + z

    argument = 1
    kwargs = {"y": 1, "z": 1}
    expected_result = 3
    with caplog.at_level(logging.WARN):
        result = foo(argument, **kwargs)
        assert any(
            re.search(expected_msg_wip, record.message) for record in caplog.records
        )
        assert expected_result == result


def test_custom_config_fn_arg(caplog):
    """Test decorator with custom config and argument"""

    @work_in_progress(category=WipCategory.BETA)
    def foo(x):
        return x

    argument = 1
    with caplog.at_level(logging.WARN):
        result = foo(argument)
        assert any(
            re.search(expected_msg_beta, record.message) for record in caplog.records
        )
        assert argument == result


def test_temp_disable_wip_with_ctxt(caplog):
    """Test decorator with context based temp decorator disabling"""

    @work_in_progress
    def foo(y=0, z=0):
        return 1 + y + z

    with caplog.at_level(logging.WARN):
        # Check temp disable
        with TempDisableWIP():
            result = foo(1)
            assert result == 2
            # no log lines produced
            assert len(caplog.records) == 0


######################## Test decorator on class ############################################


def test_default_config_class_no_arg(caplog):
    """Test decorator with no config on class with no init args"""

    @work_in_progress
    class Foo:
        pass

    with caplog.at_level(logging.WARN):
        Foo()
        assert any(
            re.search(expected_msg_wip, record.message) for record in caplog.records
        )


def test_default_config_class_arg(caplog):
    """Test decorator with no config on class with init arguments"""

    @work_in_progress
    class Foo:
        def __init__(self, x) -> None:
            self.x = x

    argument = 1
    with caplog.at_level(logging.WARN):
        result = Foo(argument)
        assert any(
            re.search(expected_msg_wip, record.message) for record in caplog.records
        )
        assert argument == result.x


def test_default_config_class_kwarg(caplog):
    """Test decorator with no config on class with init argument and keyword argument"""

    @work_in_progress
    class Foo:
        def __init__(self, x, y=0, z=0) -> None:
            self.x = x + y + z

    argument = 1
    kwargs = {"y": 1, "z": 1}
    expected_result = 3
    with caplog.at_level(logging.WARN):
        result = Foo(argument, **kwargs)
        assert any(
            re.search(expected_msg_wip, record.message) for record in caplog.records
        )
        assert expected_result == result.x


def test_custom_config_class_arg(caplog):
    """Test decorator on class with custom config and init argument"""

    @work_in_progress(category=WipCategory.BETA)
    class Foo:
        def __init__(self, x) -> None:
            self.x = x

    argument = 1
    with caplog.at_level(logging.WARN):
        result = Foo(argument)
        assert any(
            re.search(expected_msg_beta, record.message) for record in caplog.records
        )
        assert argument == result.x


def test_inheritance_of_decorated_class(caplog):
    """Test decorator on a class that is inherited by another class
    still works.
    """

    @work_in_progress(category=WipCategory.BETA)
    class Foo:
        pass

    class Bar(Foo):
        def __init__(self, x) -> None:
            self.x = x

    argument = 1
    with caplog.at_level(logging.WARN):
        result = Bar(argument)
        assert any(
            re.search(expected_msg_beta, record.message) for record in caplog.records
        )
        assert argument == result.x


######################## Test separate actions #########################################


def test_error_action_raises_fn(caplog):
    """Test error action passed to decorator function raises"""
    func = lambda x: x
    work_in_progress(func, action=Action.ERROR)
    with pytest.raises(Exception):
        func()


def test_error_action_raises_class(caplog):
    """Test error action passed to decorator class raises"""

    @work_in_progress(action=Action.ERROR)
    class Foo:
        pass

    with pytest.raises(Exception):
        Foo()


######################## Test disable functionality #########################################


def test_decorator_can_be_disabled(caplog):
    """Test that decorator can be disabled using disable_wip function"""
    func = lambda x: x
    work_in_progress(func, action=Action.ERROR)
    # Sanity check that we get exception as expected
    with pytest.raises(Exception):
        func()
    # Disable decorator
    disable_wip()

    # Try again, this time there should not be any exception
    result = func(1)
    assert result == 1
