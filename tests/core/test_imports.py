import caikit.core


def test_caikit_core_does_not_contain_itself():
    assert not hasattr(caikit.core, "caikit.core")


def test_caikit_core_has_object_serializer():
    assert hasattr(caikit.core, "ObjectSerializer")


def test_caikit_core_has_quality_evaluation():
    assert hasattr(caikit.core, "quality_evaluation")


def test_caikit_core_has_DataValidationError():
    assert hasattr(caikit.core, "DataValidationError")


def test_caikit_core_has_error_handler():
    assert hasattr(caikit.core, "error_handler")
