import json

with open("/home/student/users/Public_workspace/le-wm-3DGeom/models/le-wm/fit-[Strategy]SingleDeviceStrategy.validation_step.1783432510126366388.pt.trace.json") as f:
    trace = json.load(f)

print(trace.keys())
print(len(trace["traceEvents"]))


from collections import defaultdict
import json

with open("/home/student/users/Public_workspace/le-wm-3DGeom/models/le-wm/fit-[Strategy]SingleDeviceStrategy.validation_step.1783432510126366388.pt.trace.json") as f:
    trace = json.load(f)

times = defaultdict(float)

for e in trace["traceEvents"]:
    if e.get("ph") == "X":      # complete event
        times[e["name"]] += e.get("dur", 0)

for name, dur in sorted(times.items(), key=lambda x: x[1], reverse=True)[:30]:
    print(f"{dur/1000:.2f} ms\t{name}")