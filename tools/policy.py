"""PR metadata policy, shared by CI and offline regression tests."""
import re

TEACHER="tttt1314"

def validate(author,branch,head_repository,base_repository,files,roster,released):
    if head_repository != base_repository: raise ValueError("Use a branch in the class repository")
    if author==TEACHER:
        if not branch.startswith("instructor/"): raise ValueError("Teacher PRs use instructor/ branches")
        return {"kind":"teacher","student":"","week":max(released,default=0)}
    if author not in roster: raise ValueError("Teacher must register your GitHub login first")
    match=re.fullmatch(re.escape(author)+r"/week-(\d{2})",branch)
    if not match: raise ValueError("Branch must be <login>/week-XX")
    week=int(match[1])
    if week not in released: raise ValueError("Week has not been released")
    prefix=f"students/{author}/"
    if not files or any(not f.startswith(prefix) or ".." in f.split("/") for f in files):
        raise ValueError("Student PRs may change only their own package directory")
    return {"kind":"student","student":author,"week":week}

