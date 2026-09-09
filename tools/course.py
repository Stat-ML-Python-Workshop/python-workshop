"""Student commands: create package, add released materials, run cumulative tests."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from grade import grade

ROOT=Path(__file__).resolve().parents[1]

def sync(login,week):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,38}",login): raise ValueError("Use your GitHub login")
    released=json.loads((ROOT/"course.json").read_text())["released_weeks"]
    if week not in released: raise ValueError("Week not yet published")
    target=ROOT/"students"/login
    if target.is_symlink(): raise ValueError("Student folder cannot be a symlink")
    target.mkdir(parents=True,exist_ok=True)
    for file in (ROOT/"template").rglob("*"):
        if not file.is_file(): continue
        dest=target/file.relative_to(ROOT/"template")
        if not dest.exists():
            dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(file,dest)
    for w in range(1,week+1):
        lesson=ROOT/"weeks"/f"week-{w:02}"
        for file in (lesson/"starter"/"mini_ml").glob("*.py"):
            dest=target/"src"/"mini_ml"/file.name
            if not dest.exists(): shutil.copy2(file,dest)
        dest=target/"notebooks"/f"week-{w:02}.ipynb"
        dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists(): shutil.copy2(lesson/"notebook.ipynb",dest)
        for file in (lesson/"data").glob("*.csv"):
            dest=target/"data"/f"week-{w:02}"/file.name
            dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists(): shutil.copy2(file,dest)
    print(f"Ready: {target}. Existing files were preserved.")

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("action",choices=["init","sync","test"])
    p.add_argument("login"); p.add_argument("week",type=int,nargs="?",default=1); a=p.parse_args()
    if a.action in ("init","sync"): sync(a.login,a.week)
    else:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,38}",a.login): p.error("Invalid login")
        raise SystemExit(grade(ROOT/"students"/a.login/"src",ROOT/"tests",a.week))
