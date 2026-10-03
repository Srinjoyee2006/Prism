import subprocess
import json
import os

ps_cmd = 'Get-CimInstance Win32_Process -Filter "Name = \'python.exe\'" | Select-Object ProcessId, CommandLine | ConvertTo-Json'
res = subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], capture_output=True, text=True)

try:
    data = json.loads(res.stdout)
    if isinstance(data, dict):
        data = [data]
except Exception as e:
    print(f"Error parsing json: {e}, raw: {res.stdout}")
    data = []

my_pid = os.getpid()
for proc in data:
    cmd = proc.get("CommandLine") or ""
    pid = proc.get("ProcessId")
    if "src.livekit_agent" in cmd and pid and pid != my_pid:
        print(f"Terminating old worker PID {pid}: {cmd}")
        subprocess.run(["taskkill", "/F", "/PID", str(pid)])

print("Worker cleanup complete.")
