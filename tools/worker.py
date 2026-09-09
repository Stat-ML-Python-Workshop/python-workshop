"""One student operation, isolated from the trusted pytest process."""
import importlib
import importlib.util
import json
from pathlib import Path
import sys
root=Path(sys.argv[1]);sys.path.insert(0,str(root/'src'))
request=json.loads(sys.stdin.read())
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
print(json.dumps(response,allow_nan=False))
