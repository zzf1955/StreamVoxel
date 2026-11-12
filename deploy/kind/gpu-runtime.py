#!/usr/bin/env python3
"""Run inside the kind node: create containerd runtime handler `svgpu` whose base spec adds only the allowed GPUs'
device nodes (args: their /dev/nvidia<minor> numbers). Used by the device plugin pod, which needs NVML on every allowed card.
Workload Pods must not use this runtime class: it would give them every allowed card; they get one card each through CDI."""
import json, os, sys
minors = sys.argv[1:]
if not minors or not all(m.isdigit() for m in minors):
    sys.exit("usage: gpu-runtime.py <device-minor-number> [<device-minor-number> ...]")
spec = json.load(open("/etc/containerd/cri-base.json"))
devs = []
for p in (*(f"/dev/nvidia{m}" for m in minors), "/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools"):
    st = os.stat(p)
    devs.append({"path": p, "type": "c", "major": os.major(st.st_rdev), "minor": os.minor(st.st_rdev),
                 "fileMode": 438, "uid": 0, "gid": 0})
lin = spec.setdefault("linux", {})
lin.setdefault("devices", []).extend(devs)
rules = lin.setdefault("resources", {}).setdefault("devices", [])
for d in devs:
    rules.append({"allow": True, "type": "c", "major": d["major"], "minor": d["minor"], "access": "rwm"})
json.dump(spec, open("/etc/containerd/svgpu-base.json", "w"), indent=2)
conf = open("/etc/containerd/config.toml").read()
if "runtimes.svgpu" not in conf:
    conf += '''
[plugins."io.containerd.grpc.v1.cri".containerd.runtimes.svgpu]
  runtime_type = "io.containerd.runc.v2"
  base_runtime_spec = "/etc/containerd/svgpu-base.json"
  [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.svgpu.options]
    SystemdCgroup = true
'''
    open("/etc/containerd/config.toml", "w").write(conf)
