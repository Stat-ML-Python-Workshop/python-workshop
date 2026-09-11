"""One student operation, isolated from the trusted pytest process."""
import importlib
import importlib.util
import json
import math
from pathlib import Path
import sys
root=Path(sys.argv[1])
request=json.loads(sys.stdin.read())

def json_safe(value):
    """Encode non-finite floats explicitly while keeping strict JSON transport."""
    if isinstance(value,float) and not math.isfinite(value):
        if math.isnan(value): label="nan"
        elif value > 0: label="inf"
        else: label="-inf"
        return {"__workshop_nonfinite_float__":label}
    if isinstance(value,dict):return {key:json_safe(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [json_safe(item) for item in value]
    return value
try:
    if request['module']=='__example__':
        spec=importlib.util.spec_from_file_location('student_breast_cancer',root/'examples/breast_cancer.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    else:module=importlib.import_module('mini_ml'+('.'+request['module'] if request['module'] else ''))
    if request.get('operation')=='package_info':
        value={'name':module.__name__,'has_init':bool(getattr(module,'__file__',None))}
    elif 'class' in request:
        model=getattr(module,request['class'])(**request.get('init',{}));model.fit(*request['fit'])
        value={'prediction':model.predict(request['predict'])}
        for key in request.get('attributes',[]):value[key]=getattr(model,key)
        if request.get('probabilities'):value['probabilities']=model.predict_proba(request['predict'])
    else:value=getattr(module,request['function'])(*request.get('args',[]),**request.get('kwargs',{}))
    response={'value':value}
except Exception as error:response={'error':type(error).__name__,'message':str(error)}
print(json.dumps(json_safe(response),allow_nan=False))
