from vast import vsquared as V2

def test_version_exists():
    assert hasattr(V2, '__version__')
