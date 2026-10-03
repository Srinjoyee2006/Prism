import asyncio
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.prism_adapter import run_livekit_stream

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

async def test():
    in_wav = Path("benchmark/data/fdb_v3_data_released/ecommerce_01_65e8cf8f4c7424fa062e54a3/input.wav")
    out_wav = Path("scratch/test_out.wav")
    room_name = "fdb-test-stream-direct-1"
    print(f"Testing stream into room {room_name}...")
    try:
        r, t = await run_livekit_stream(
            input_wav=in_wav,
            output_wav=out_wav,
            room_name=room_name,
            user_speech_end=17.3,
        )
        print("STREAM COMPLETED SUCCESSFULLY:", r, t)
    except Exception as e:
        print("STREAM FAILED:")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test())
