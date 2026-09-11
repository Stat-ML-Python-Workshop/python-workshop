"""Trusted tests communicate with student code through a bounded subprocess."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

def restore_nonfinite(value):
    if isinstance(value,dict) and set(value)=={"__workshop_nonfinite_float__"}:
        return {"inf":float("inf"),"-inf":float("-inf"),"nan":float("nan")}[value["__workshop_nonfinite_float__"]]
    if isinstance(value,dict):return {key:restore_nonfinite(item) for key,item in value.items()}
    if isinstance(value,list):return [restore_nonfinite(item) for item in value]
    return value

@pytest.fixture
def invoke():
    def run(module,function=None,args=None,kwargs=None,**extra):
        if module == "application": module = "__example__"
        elif int(os.environ["COURSE_WEEK"]) >= 8:
            module = {"statistics":"math.statistics", "linalg":"math.linalg", "probability":"math.probability", "information":"math.information", "neighbors":"models.neighbors", "linear_models":"models.linear_models"}.get(module,module)
        request={"module":module,"args":args or [],"kwargs":kwargs or {},**extra}
        if function: request["function"]=function
        def drop_privileges():
            if os.getuid() == 0:
                os.setgroups([]); os.setgid(65534); os.setuid(65534)
        python=os.environ.get("STUDENT_PYTHON",sys.executable)
        process=subprocess.run([python,"-I",str(Path(__file__).with_name("worker.py")),os.environ["STUDENT_ROOT"]],input=json.dumps(request),text=True,capture_output=True,timeout=90,cwd="/tmp",preexec_fn=drop_privileges if os.environ.get("SANDBOX_STUDENT")=="1" else None)
        assert process.returncode==0, "Student operation exited unexpectedly: "+process.stderr[-1000:]
        try: return restore_nonfinite(json.loads(process.stdout))
        except ValueError: pytest.fail("Expected one JSON result; remove debug print calls from library functions")
    return run

@pytest.fixture
def call(invoke):
    def run(module,function,*args,**kwargs):
        response=invoke(module,function,list(args),kwargs)
        assert "error" not in response,response
        return response["value"]
    return run
