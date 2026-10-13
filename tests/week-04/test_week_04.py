import ast
import os
from pathlib import Path
import pytest

def test_shape(call,invoke):
    assert call("linalg","shape",[[1,2,3],[4,5,6]])==[2,3]
    assert invoke("linalg","shape",[[]])["error"]=="ValueError"
    assert invoke("linalg","shape",[[[]]])["error"]=="ValueError"
    assert invoke("linalg","shape",[[[1,2],[3]]])["error"]=="ValueError"

def test_transpose(call,invoke):
    assert call("linalg","transpose",[[1,2,3],[4,5,6]])==[[1,4],[2,5],[3,6]]
    assert invoke("linalg","transpose",[[[1,2],[3]]])["error"]=="ValueError"

def test_matmul(call):
    assert call("linalg","matmul",[[1,2,3],[4,5,6]],[[7,8],[9,10],[11,12]])==[[58,64],[139,154]]
    assert call("linalg","matmul",[[1,2],[3,4]],[[2],[1]])==[[4],[10]]
    assert call("linalg","matmul",[[3]],[[4]])==[[12]]
    assert call("linalg","matmul",[[-2,0,3]],[[4],[7],[-1]])==[[-11]]
    assert call("linalg","matmul",[[0.1,0.2]],[[0.3],[0.4]])[0][0]==pytest.approx(0.11)

def test_dimensions(invoke):
    assert invoke("linalg","matmul",[[[1,2]],[[1,2]]])["error"]=="ValueError"
    assert invoke("linalg","matmul",[[[1,2],[3]],[[1],[2]]])["error"]=="ValueError"
    assert invoke("linalg","matmul",[[[1,2]],[[1,2],[3]]])["error"]=="ValueError"

def test_pure_python_composition():
    package=Path(os.environ["STUDENT_ROOT"])/"src/mini_ml"
    path=package/"linalg.py"
    if not path.exists():
        path=package/"math/linalg.py"
    source=path.read_text()
    tree=ast.parse(source)
    imports=[node for node in ast.walk(tree) if isinstance(node,(ast.Import,ast.ImportFrom))]
    numpy_aliases={alias.asname or alias.name.split(".")[0] for node in imports
                   for alias in node.names if alias.name=="numpy" or alias.name.startswith("numpy.")
                   or isinstance(node,ast.ImportFrom) and (node.module or "").startswith("numpy")}
    if int(os.environ["COURSE_WEEK"]) < 5:
        assert not numpy_aliases, "Use pure Python in Week 4 linalg.py"
    functions={node.name:node for node in tree.body if isinstance(node,ast.FunctionDef)}
    assert set(("shape","transpose","matmul"))<=functions.keys()
    for name in ("shape","transpose","matmul"):
        assert not any(isinstance(node,ast.Name) and node.id in numpy_aliases
                       for node in ast.walk(functions[name])), "Preserve pure Python Week 4 functions"

    assert [arg.arg for arg in functions["matmul"].args.args]==["m1","m2"]
    calls={node.func.id for node in ast.walk(functions["matmul"])
           if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)}
    assert "shape" in calls
    assert not any(isinstance(node,ast.MatMult) for node in ast.walk(functions["matmul"]))
