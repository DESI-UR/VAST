from vast import voidfinder as voidfinder

def test_version_exists():
    assert hasattr(voidfinder, '__version__')
