"""Run trusted cumulative tests; ignore all student pytest/package configuration."""
import argparse
import ast
from contextlib import nullcontext
import os
from pathlib import Path
import shutil
import site
import subprocess
import sys
import tempfile

_WEEK_TWO_REQUIRED_CASES = {
    ("test_proportions", "labels,expected"): ast.parse(
        '''[
            (["T", "T", "B", "M"], {"T": 0.5, "B": 0.25, "M": 0.25}),
            (["B", "B"], {"B": 1.0}),
            ([0, 0, 1], {0: 2 / 3, 1: 1 / 3}),
        ]''',
        mode="eval",
    ).body.elts,
    ("test_entropy", "probabilities,expected"): ast.parse(
        '''[
            ([0.5, 0.5], 1.0),
            ([1.0], 0.0),
            ([1.0, 0.0], 0.0),
            ([0.5, 0.25, 0.25], 1.5),
        ]''',
        mode="eval",
    ).body.elts,
}

_WEEK_THREE_REQUIRED_CASES = {
    ("tests/test_statistics.py", "test_variance", "values,ddof,expected"): ast.parse(
        '''[
            ([2, 4, 6], 1, 4.0),
            ([2, 4, 6], 0, 8 / 3),
            ([7, 7, 7], 1, 0.0),
        ]''',
        mode="eval",
    ).body.elts,
    ("tests/test_composition.py", "test_cross_entropy", "p,q,expected"): ast.parse(
        '''[
            ([0.75, 0.25], [0.25, 0.75], 1.603759374819711),
            ([0.5, 0.5], [0.25, 0.75], 1.207518749639422),
            ([1.0, 0.0], [1.0, 0.0], 0.0),
        ]''',
        mode="eval",
    ).body.elts,
    ("tests/test_composition.py", "test_kl_divergence", "p,q,expected"): ast.parse(
        '''[
            ([0.75, 0.25], [0.25, 0.75], 0.792481250360578),
            ([0.2, 0.8], [0.2, 0.8], 0.0),
            ([1.0, 0.0], [0.25, 0.75], 2.0),
        ]''',
        mode="eval",
    ).body.elts,
}


def _fingerprint(node):
    return ast.dump(node, include_attributes=False)


def _parametrized_cases(tree, function_name, parameter_names):
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        ),
        None,
    )
    if function is None:
        raise ValueError(f"{function_name} is missing from tests/test_composition.py")
    expected_names = parameter_names.replace(" ", "")
    for decorator in function.decorator_list:
        if not (
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr == "parametrize"
            and len(decorator.args) >= 2
            and isinstance(decorator.args[0], ast.Constant)
            and isinstance(decorator.args[0].value, str)
            and decorator.args[0].value.replace(" ", "") == expected_names
        ):
            continue
        cases = decorator.args[1]
        if not isinstance(cases, (ast.List, ast.Tuple)):
            raise ValueError(
                f"{function_name} must keep its cases in the supplied parametrized list"
            )
        return cases.elts
    raise ValueError(
        f"{function_name} must keep the supplied @pytest.mark.parametrize decorator"
    )


def validate_week_two_student_tests(project):
    """Require the supplied Week 2 tests plus one original case for each function."""
    path = Path(project) / "tests/test_composition.py"
    if not path.is_file():
        raise ValueError(
            "tests/test_composition.py is required from Week 2 onward"
        )
    if path.stat().st_size > 256_000:
        raise ValueError("tests/test_composition.py is unexpectedly large")
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as error:
        raise ValueError(f"Cannot parse tests/test_composition.py: {error}") from error

    labels = {
        "test_proportions": "label/proportion",
        "test_entropy": "probability/entropy",
    }
    for (function_name, parameter_names), required_nodes in _WEEK_TWO_REQUIRED_CASES.items():
        cases = _parametrized_cases(tree, function_name, parameter_names)
        required = {_fingerprint(node) for node in required_nodes}
        submitted = {_fingerprint(node) for node in cases}
        if not required.issubset(submitted):
            raise ValueError(
                f"{function_name} must retain all supplied {labels[function_name]} cases"
            )
        if not submitted - required:
            raise ValueError(
                f"{function_name} must add at least one original "
                f"{labels[function_name]} case"
            )


def validate_week_three_student_tests(project):
    """Require the supplied Week 3 tests plus one original case per core topic."""
    trees = {}
    for relative_path, function_name, parameter_names in _WEEK_THREE_REQUIRED_CASES:
        path = Path(project) / relative_path
        if not path.is_file():
            raise ValueError(f"{relative_path} is required from Week 3 onward")
        if path.stat().st_size > 256_000:
            raise ValueError(f"{relative_path} is unexpectedly large")
        if relative_path not in trees:
            try:
                trees[relative_path] = ast.parse(
                    path.read_text(encoding="utf-8"), filename=str(path)
                )
            except (OSError, UnicodeError, SyntaxError) as error:
                raise ValueError(f"Cannot parse {relative_path}: {error}") from error
        cases = _parametrized_cases(
            trees[relative_path], function_name, parameter_names
        )
        required = {
            _fingerprint(node)
            for node in _WEEK_THREE_REQUIRED_CASES[
                (relative_path, function_name, parameter_names)
            ]
        }
        submitted = {_fingerprint(node) for node in cases}
        if not required.issubset(submitted):
            raise ValueError(
                f"{function_name} must retain all supplied Week 3 cases"
            )
        if not submitted - required:
            raise ValueError(
                f"{function_name} must add at least one original Week 3 case"
            )

def drop_privileges():
    if os.getuid() == 0:
        os.setgroups([]);os.setgid(65534);os.setuid(65534)

def run_as_student(command,*,cwd,env,timeout):
    return subprocess.run(
        command,cwd=cwd,env=env,text=True,capture_output=True,timeout=timeout,
        preexec_fn=drop_privileges if os.environ.get("SANDBOX_STUDENT")=="1" else None)

def show_failure(label,result):
    print(f"{label} failed",file=sys.stderr)
    if result.stdout:print(result.stdout[-4000:],file=sys.stderr)
    if result.stderr:print(result.stderr[-4000:],file=sys.stderr)

def grade(source, tests, week, junit=None):
    source=Path(source).resolve(); tests=Path(tests).resolve()
    if not 1 <= week <= 10: raise ValueError("week must be 1..10")
    selected=[tests/f"week-{w:02}"/f"test_week_{w:02}.py" for w in range(1,week+1)]
    if any(not p.is_file() for p in selected): raise ValueError("Required week tests are not published")
    sandboxed=os.environ.get("SANDBOX_STUDENT") == "1"
    # A capability-restricted container cannot remove files created by the
    # unprivileged student account. Its disposable tmpfs vanishes with the
    # container, while ordinary local runs still clean up immediately.
    workspace=(
        nullcontext(tempfile.mkdtemp(prefix="workshop-grade-"))
        if sandboxed else tempfile.TemporaryDirectory(prefix="workshop-grade-")
    )
    with workspace as folder:
        destination=Path(folder)
        destination.chmod(0o755)
        project=destination/"student-project"
        runtime=destination/"student-runtime"
        trusted_tests=destination/"trusted-tests"
        trusted_tests.mkdir()
        shutil.copytree(source,project,ignore=shutil.ignore_patterns(".venv","__pycache__","test_code",".git"))
        if week >= 2:
            try:
                validate_week_two_student_tests(project)
            except ValueError as error:
                print("Student unit test requirements failed", file=sys.stderr)
                print(error, file=sys.stderr)
                return 1
        if week >= 3:
            try:
                validate_week_three_student_tests(project)
            except ValueError as error:
                print("Student unit test requirements failed", file=sys.stderr)
                print(error, file=sys.stderr)
                return 1
        subprocess.run([sys.executable,"-m","venv","--system-site-packages",str(runtime)],check=True)
        runtime_python=runtime/"bin"/"python"
        purelib=Path(subprocess.check_output(
            [str(runtime_python),"-c",'import sysconfig; print(sysconfig.get_paths()["purelib"])'],text=True).strip())
        (purelib/"workshop-ci-dependencies.pth").write_text("\n".join(site.getsitepackages())+"\n")
        if os.environ.get("SANDBOX_STUDENT")=="1":
            for path in (project,runtime,*project.rglob("*"),*runtime.rglob("*")):
                if path.is_dir():path.chmod(0o777)
        for test in selected: shutil.copy2(test,trusted_tests/test.name)
        for name in ("conftest.py","worker.py"): shutil.copy2(Path(__file__).with_name(name),trusted_tests/name)
        (trusted_tests/"pytest.ini").write_text("[pytest]\naddopts =\n")
        env={**os.environ,"STUDENT_ROOT":str(project),"STUDENT_PYTHON":str(runtime_python),"COURSE_WEEK":str(week),"PYTEST_DISABLE_PLUGIN_AUTOLOAD":"1","PYTHONDONTWRITEBYTECODE":"1","PIP_NO_CACHE_DIR":"1","PIP_DISABLE_PIP_VERSION_CHECK":"1"}
        env.pop("PYTEST_ADDOPTS",None); env.pop("PYTHONPATH",None)
        install=run_as_student(
            [str(runtime_python),"-m","pip","install","--no-deps","--no-build-isolation","--editable",str(project)],
            cwd=project,env=env,timeout=180)
        if install.returncode:
            show_failure("Editable package installation",install);return install.returncode
        student_tests=run_as_student(
            [str(runtime_python),"-I","-m","pytest","-q","-p","no:cacheprovider","-c",str(trusted_tests/"pytest.ini"),str(project/"tests")],
            cwd=destination,env=env,timeout=180)
        if student_tests.returncode:
            show_failure("Student unit tests",student_tests);return student_tests.returncode
        cmd=[sys.executable,"-I","-m","pytest","-q","-p","no:cacheprovider","-c",str(trusted_tests/"pytest.ini"),str(trusted_tests)]
        if junit: cmd += ["--junitxml",str(Path(junit).resolve())]
        # These tests are teacher-controlled. They remain inside the restricted
        # container and execute student functions only through the unprivileged
        # worker configured in conftest.py.
        trusted=subprocess.run(
            cmd,cwd=destination,env=env,text=True,capture_output=True,timeout=600)
        if trusted.stdout:print(trusted.stdout,end="")
        if trusted.returncode:show_failure("Trusted acceptance tests",trusted)
        return trusted.returncode

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("source"); parser.add_argument("tests"); parser.add_argument("week",type=int); parser.add_argument("--junit")
    args=parser.parse_args()
    raise SystemExit(grade(args.source,args.tests,args.week,args.junit))
