import asyncio
import sys
import time
from livekit import api, rtc

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

URL = "ws://127.0.0.1:7880"
API_KEY = "devkey"
API_SECRET = "secret"

async def main():
    room_name = f"fdb-test-sim-{int(time.time())}"
    token = (
        api.AccessToken(API_KEY, API_SECRET)
        .with_identity("fdb-benchmark-user")
        .with_name("FDB Benchmark User")
        .with_grants(api.VideoGrants(room_join=True, room=room_name))
        .to_jwt()
    )
    room = rtc.Room()
    agent_ready = asyncio.Event()

    @room.on("track_published")
    def on_track_published(pub, participant):
        print(f"Track published: kind={pub.kind} by {participant.identity}")
        if pub.kind == rtc.TrackKind.KIND_AUDIO:
            agent_ready.set()

    @room.on("track_subscribed")
    def on_track_subscribed(track, pub, participant):
        print(f"Track subscribed: kind={track.kind} by {participant.identity}")
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            agent_ready.set()

    print(f"Connecting to room '{room_name}'...")
    await room.connect(URL, token, options=rtc.RoomOptions(auto_subscribe=True))
    print("Connected. Publishing audio track...")
    source = rtc.AudioSource(48000, 1)
    local_track = rtc.LocalAudioTrack.create_audio_track("wav-input", source)
    pub_opts = rtc.TrackPublishOptions()
    pub_opts.source = rtc.TrackSource.SOURCE_MICROPHONE
    await room.local_participant.publish_track(local_track, pub_opts)

    print("Waiting for agent track (10s max)...")
    try:
        await asyncio.wait_for(agent_ready.wait(), timeout=10.0)
        print("SUCCESS! Agent track detected!")
    except asyncio.TimeoutError:
        print("FAIL! Timed out waiting for agent track.")

    await room.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
