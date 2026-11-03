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
    if hasattr(value, "tolist"):return json_safe(value.tolist())
    if isinstance(value,dict):return {key:json_safe(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [json_safe(item) for item in value]
    return value


def class_sequence(module, request):
    """Bounded same-instance checks; never evaluate source from a request.

    steps contains 1..16 {method, args} records, each with one argument.
    An argument {"result": i} refers to an earlier successful array result.
    Per-step errors are recorded so a failed refit can be followed by transform.
    Attribute/result snapshots are serialized immediately, before another call.
    """
    steps = request.get("steps")
    allowed = {"fit", "transform", "inverse_transform"}
    attributes = request.get("attributes", [])
    known_attributes = {"mean_", "components_", "explained_variance_",
                        "explained_variance_ratio_", "n_features_in_", "n_samples_"}
    if not isinstance(steps, list) or not 1 <= len(steps) <= 16:
        raise ValueError("class_sequence requires 1..16 steps")
    if not isinstance(attributes, list) or any(not isinstance(key, str) or key not in known_attributes for key in attributes):
        raise ValueError("Unsupported class_sequence attribute")
    for step in steps:
        if (not isinstance(step, dict) or not isinstance(step.get("method"), str) or step["method"] not in allowed
                or not isinstance(step.get("args"), list) or len(step["args"]) != 1
                or set(step) != {"method", "args"}):
            raise ValueError("Each sequence step requires one argument and an allowed method")
    model = getattr(module, request["class"])(**request.get("init", {}))
    results, reports = [], []
    for step in steps:
        argument = step["args"][0]
        if isinstance(argument, dict):
            index = argument.get("result")
            if (set(argument) != {"result"} or type(index) is not int
                    or not 0 <= index < len(results) or results[index] is None):
                raise ValueError("Result references require an earlier successful array result")
            argument = results[index]
        before = json_safe(argument)
        try:
            returned = getattr(model, step["method"])(argument)
            is_self = returned is model
            result = None if is_self else returned
            report = {"value": json_safe(result), "is_self": is_self,
                      "type": type(returned).__name__,
                      "dtype": str(returned.dtype) if hasattr(returned, "dtype") else None,
                      "shape": list(returned.shape) if hasattr(returned, "shape") else None}
        except Exception as error:
            result = None
            report = {"error": type(error).__name__, "message": str(error)}
        report["input_unchanged"] = before == json_safe(argument)
        report["attributes"] = {key: json_safe(getattr(model, key, None)) for key in attributes}
        results.append(result)
        reports.append(report)
    return {"steps": reports}


try:
    if request['module']=='__example__':
        spec=importlib.util.spec_from_file_location('student_breast_cancer',root/'examples/breast_cancer.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    else:module=importlib.import_module('mini_ml'+('.'+request['module'] if request['module'] else ''))
    if request.get('operation')=='package_info':
        value={'name':module.__name__,'has_init':bool(getattr(module,'__file__',None))}
    elif request.get('operation')=='class_sequence':
        value=class_sequence(module, request)
    elif 'class' in request:
        model=getattr(module,request['class'])(**request.get('init',{}))
        if 'fit' in request:model.fit(*request['fit'])
        method = request.get('method', 'predict')
        value={'prediction':getattr(model, method)(request['predict'])}
        for key in request.get('attributes',[]):value[key]=getattr(model,key)
        if request.get('probabilities'):value['probabilities']=model.predict_proba(request['predict'])
    else:value=getattr(module,request['function'])(*request.get('args',[]),**request.get('kwargs',{}))
    response={'value':value}
except Exception as error:response={'error':type(error).__name__,'message':str(error)}
print(json.dumps(json_safe(response),allow_nan=False))
