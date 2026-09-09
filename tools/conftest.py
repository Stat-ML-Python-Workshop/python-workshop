"""Trusted tests communicate with student code through a bounded subprocess."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

@pytest.fixture
def invoke():
    def run(module,function=None,args=None,kwargs=None,**extra):
        request={"module":module,"args":args or [],"kwargs":kwargs or {},**extra}
        if function: request["function"]=function
        def drop_privileges():
            os.setgroups([]); os.setgid(65534); os.setuid(65534)
        process=subprocess.run([sys.executable,"-I",str(Path(__file__).with_name("worker.py")),os.environ["STUDENT_SRC"]],input=json.dumps(request),text=True,capture_output=True,timeout=90,cwd="/tmp",preexec_fn=drop_privileges if os.environ.get("SANDBOX_STUDENT")=="1" else None)
        assert process.returncode==0, "Student operation exited unexpectedly: "+process.stderr[-1000:]
        try: return json.loads(process.stdout)
        except ValueError: pytest.fail("Expected one JSON result; remove debug print calls from library functions")
    return run

@pytest.fixture
def call(invoke):
    def run(module,function,*args,**kwargs):
        response=invoke(module,function,list(args),kwargs)
        assert "error" not in response,response
        return response["value"]
    return run
