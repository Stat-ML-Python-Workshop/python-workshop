import os
from pathlib import Path
import tomllib

def test_hello_world(call):
    assert call("","hello_world")=="Hello, world!"

def test_project_files():
    root=Path(os.environ["STUDENT_ROOT"])
    for name in ("pyproject.toml","uv.lock","README.md",".gitignore","tests/test_greetings.py","src/mini_ml/__init__.py","src/mini_ml/greetings.py"):
        assert (root/name).is_file(), f"Missing {name}"
    project=tomllib.loads((root/"pyproject.toml").read_text())
    assert project["build-system"]["build-backend"]=="setuptools.build_meta"
    assert project["tool"]["setuptools"]["packages"]["find"]["where"]==["src"]
    assert project["tool"]["pytest"]["ini_options"]["testpaths"]==["tests"]
    assert project["project"]["requires-python"]==">=3.12,<3.13"
    if int(os.environ["COURSE_WEEK"])==1:
        assert project["project"].get("dependencies",[])==[]
    assert any(item.startswith("pytest") for item in project["dependency-groups"]["dev"])
    assert "test_code/" in (root/".gitignore").read_text().splitlines()
