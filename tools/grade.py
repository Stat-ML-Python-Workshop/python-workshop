"""Run trusted cumulative tests; ignore all student pytest/package configuration."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

def grade(source, tests, week, junit=None):
    source=Path(source).resolve(); tests=Path(tests).resolve()
    if not 1 <= week <= 10: raise ValueError("week must be 1..10")
    selected=[tests/f"week-{w:02}"/f"test_week_{w:02}.py" for w in range(1,week+1)]
    if any(not p.is_file() for p in selected): raise ValueError("Required week tests are not published")
    with tempfile.TemporaryDirectory(prefix="workshop-grade-") as folder:
        destination=Path(folder)
        destination.chmod(0o755)
        for test in selected: shutil.copy2(test,destination/test.name)
        for name in ("conftest.py","worker.py"): shutil.copy2(Path(__file__).with_name(name),destination/name)
        (destination/"pytest.ini").write_text("[pytest]\naddopts =\n")
        env={**os.environ,"STUDENT_SRC":str(source),"PYTEST_DISABLE_PLUGIN_AUTOLOAD":"1","PYTHONDONTWRITEBYTECODE":"1"}
        env.pop("PYTEST_ADDOPTS",None); env.pop("PYTHONPATH",None)
        cmd=[sys.executable,"-I","-m","pytest","-q","-p","no:cacheprovider","-c",str(destination/"pytest.ini"),str(destination)]
        if junit: cmd += ["--junitxml",str(Path(junit).resolve())]
        return subprocess.run(cmd,cwd=destination,env=env,timeout=600).returncode

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("source"); parser.add_argument("tests"); parser.add_argument("week",type=int); parser.add_argument("--junit")
    args=parser.parse_args()
    raise SystemExit(grade(args.source,args.tests,args.week,args.junit))
