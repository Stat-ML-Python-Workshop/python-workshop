"""Run one student operation in a child interpreter; never imported by pytest."""
import importlib
import json
import sys

sys.path.insert(0, sys.argv[1])
request = json.loads(sys.stdin.read())
try:
    module = importlib.import_module("mini_ml." + request["module"])
    if "class" in request:
        model = getattr(module,request["class"])(**request.get("init",{}))
        model.fit(*request["fit"])
        value = {"prediction": model.predict(request["predict"])}
        for key in request.get("attributes",[]): value[key] = getattr(model,key)
        if request.get("probabilities"): value["probabilities"] = model.predict_proba(request["predict"])
    else:
        value = getattr(module,request["function"])(*request.get("args",[]),**request.get("kwargs",{}))
    response = {"value": value}
except Exception as error:
    response = {"error": type(error).__name__, "message": str(error)}
print(json.dumps(response,allow_nan=False))

