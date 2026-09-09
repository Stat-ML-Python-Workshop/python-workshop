import os
from pathlib import Path
import tomllib

def test_package_import(invoke):
    result=invoke("",operation="package_info")
    assert result.get("value")=={"name":"mini_ml","has_init":True}

def test_project_files():
    root=Path(os.environ["STUDENT_ROOT"])
    for name in ("pyproject.toml","uv.lock","README.md",".gitignore","tests/test_package.py","src/mini_ml/__init__.py"):
        assert (root/name).is_file(), f"Missing {name}"
    project=tomllib.loads((root/"pyproject.toml").read_text())
    assert project["build-system"]["build-backend"]=="setuptools.build_meta"
    assert project["tool"]["setuptools"]["packages"]["find"]["where"]==["src"]
    assert project["tool"]["pytest"]["ini_options"]["testpaths"]==["tests"]
    assert "test_code/" in (root/".gitignore").read_text().splitlines()
