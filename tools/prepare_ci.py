"""Trusted base-side preparation. Never execute a submitted build backend or script."""
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request
from policy import validate, roster_map

def api(path):
    request=urllib.request.Request("https://api.github.com/"+path,headers={"Authorization":"Bearer "+os.environ["GH_TOKEN"],"Accept":"application/vnd.github+json"})
    with urllib.request.urlopen(request) as response: return json.load(response)

def main():
    event=json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    pr=event["pull_request"]; number=pr["number"]
    base=Path("trusted"); head=Path("submission")
    repository=event["repository"]["full_name"]
    files=[]
    page=1
    while True:
        batch=api(f"repos/{repository}/pulls/{number}/files?per_page=100&page={page}")
        for item in batch:
            files.append(item["filename"])
            if item.get("previous_filename"): files.append(item["previous_filename"])
        if len(batch)<100: break
        page+=1
    latest=api(f"repos/{repository}/pulls/{number}")
    if latest["head"]["sha"]!=pr["head"]["sha"]: raise ValueError("PR changed during preparation; rerun latest commit")
    course=json.loads((base/"course.json").read_text())
    roster=json.loads((base/"roster.json").read_text())["students"]
    result=validate(pr["user"]["login"],pr["head"]["ref"],pr["head"]["repo"]["full_name"],repository,files,roster,course["released_weeks"])
    # Git index modes expose symlinks/submodules even if checkout behavior varies.
    index=subprocess.check_output(["git","-C",str(head),"ls-files","--stage","-z"]).decode().split("\0")
    for entry in filter(None,index):
        if entry[:6] not in ("100644","100755"): raise ValueError("Symlinks and submodules are not accepted")
    if result["kind"]=="student":
        source=head/"students"/result["folder"]
        if not (source/"src"/"mini_ml"/"__init__.py").is_file(): raise ValueError("Missing src/mini_ml package")
        destination=Path("sandbox-source")
        shutil.copytree(source,destination,ignore=shutil.ignore_patterns(".venv","__pycache__","test_code",".git"))
        if sum(p.stat().st_size for p in destination.rglob("*") if p.is_file())>5_000_000: raise ValueError("Keep package source under 5 MB")
    else:
        # Teacher release validation is static here. Reference execution occurs in private verification.
        roster_map(json.loads((head/"roster.json").read_text())["students"])
        manifest=json.loads((head/"course.json").read_text())
        released=manifest["released_weeks"]
        if released!=list(range(1,len(released)+1)) or len(released)>10: raise ValueError("Weeks must be contiguous 1..10")
        for week in released:
            folder=head/"weeks"/f"week-{week:02}"
            if not (folder/"lesson.md").is_file(): raise ValueError("Missing lesson")
            if not (folder/"demo.py").is_file(): raise ValueError("Missing demo.py")
            if not (head/"tests"/f"week-{week:02}"/f"test_week_{week:02}.py").is_file(): raise ValueError("Missing tests")
        for file in head.rglob("*.py"):
            if ".git" not in file.parts: ast.parse(file.read_text())
    Path("ci-plan.json").write_text(json.dumps(result))
    with open(os.environ["GITHUB_OUTPUT"],"a") as output:
        for key,value in result.items(): output.write(f"{key}={value}\n")

if __name__=="__main__": main()
