import asyncio
import sys
import time
from livekit import api, rtc

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

URL = "ws://127.0.0.1:7880"
API_KEY = "devkey"
API_SECRET = "secret"

async def test_single_room(room_name: str) -> float:
    print(f"\n--- Testing room {room_name} ---")
    token = (
        api.AccessToken(API_KEY, API_SECRET)
        .with_identity("test-verifier")
        .with_name("Test Verifier")
        .with_grants(api.VideoGrants(room_join=True, room=room_name))
        .to_jwt()
    )
    room = rtc.Room()
    agent_ready = asyncio.Event()

    @room.on("track_published")
    def on_track_published(pub, participant):
        if pub.kind == rtc.TrackKind.KIND_AUDIO:
            agent_ready.set()

    @room.on("track_subscribed")
    def on_track_subscribed(track, pub, participant):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            agent_ready.set()

    for p in room.remote_participants.values():
        for pub in p.track_publications.values():
            if pub.kind == rtc.TrackKind.KIND_AUDIO:
                agent_ready.set()

    t0 = time.time()
    await room.connect(URL, token, options=rtc.RoomOptions(auto_subscribe=True))
    print(f"[{room_name}] Participant connected in {time.time()-t0:.3f}s. Waiting for agent track...")

    try:
        await asyncio.wait_for(agent_ready.wait(), timeout=10.0)
        ready_duration = time.time() - t0
        print(f"[{room_name}] [SUCCESS] AGENT TRACK DETECTED IN {ready_duration:.3f}s!")
    except asyncio.TimeoutError:
        ready_duration = -1.0
        print(f"[{room_name}] [FAIL] TIMEOUT waiting for agent track!")

    await room.disconnect()
    print(f"[{room_name}] Disconnected. Waiting 0.5s before next room...")
    await asyncio.sleep(0.5)
    return ready_duration

async def main():
    d1 = await test_single_room("fdb-test-verify-101")
    d2 = await test_single_room("fdb-test-verify-102")
    print(f"\nSUMMARY: Room 1: {d1:.3f}s | Room 2: {d2:.3f}s")

if __name__ == "__main__":
    asyncio.run(main())
