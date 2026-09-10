"""Run trusted cumulative tests; ignore all student pytest/package configuration."""
import argparse
from contextlib import nullcontext
import os
from pathlib import Path
import shutil
import site
import subprocess
import sys
import tempfile

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
