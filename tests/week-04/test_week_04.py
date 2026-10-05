import ast
import os
from pathlib import Path
import pytest

def test_dot(call):
    assert call("linalg","dot",[1,2],[3,4])==pytest.approx(11)
    assert call("linalg","dot",[-2,0,3],[4,7,-1])==pytest.approx(-11)
    assert call("linalg","dot",[0.1,0.2],[0.3,0.4])==pytest.approx(0.11)
    assert call("linalg","dot",[],[])==0.0

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

def test_dimensions(invoke):
    assert invoke("linalg","dot",[[1],[1,2]])["error"]=="ValueError"
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
    assert all(not any(alias.name=="numpy" or alias.name.startswith("numpy.") for alias in node.names)
               for node in imports), "Use pure Python in linalg.py"
    functions={node.name:node for node in tree.body if isinstance(node,ast.FunctionDef)}
    assert set(("dot","shape","transpose","matmul"))<=functions.keys()
    assert [arg.arg for arg in functions["matmul"].args.args]==["m1","m2"]
    calls={node.func.id for node in ast.walk(functions["matmul"])
           if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)}
    assert {"shape","transpose","dot"}<=calls
    assert not any(isinstance(node,ast.MatMult) for node in ast.walk(functions["matmul"]))
