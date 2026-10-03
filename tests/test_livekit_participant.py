"""
Tests for LiveKit participant token generation and web participant configuration (Stage 7).

Validates:
1. Token generation with default local development credentials (devkey/secret).
2. Correct VideoGrants (room_join=True, room='prism-test', can_publish=True, can_subscribe=True).
3. Custom room and identity encoding.
4. Signature verification and rejection of tampered tokens.
5. Presence and integrity of local web participant assets.
"""

from pathlib import Path

import jwt
import pytest

from src.livekit.token import (
    DEFAULT_API_KEY,
    DEFAULT_API_SECRET,
    DEFAULT_IDENTITY,
    DEFAULT_ROOM,
    generate_participant_token,
    validate_token,
)


class TestParticipantToken:
    def test_default_token_generation_and_payload(self) -> None:
        """Verify token generated with defaults contains expected room and permissions."""
        token = generate_participant_token()
        assert isinstance(token, str)
        assert len(token) > 50

        # Decode and inspect claims
        claims = validate_token(token, api_secret=DEFAULT_API_SECRET)
        assert claims["iss"] == DEFAULT_API_KEY
        assert claims["sub"] == DEFAULT_IDENTITY
        assert "video" in claims

        video_grants = claims["video"]
        assert video_grants["roomJoin"] is True
        assert video_grants["room"] == DEFAULT_ROOM
        assert video_grants["canPublish"] is True
        assert video_grants["canSubscribe"] is True

    def test_custom_room_and_identity(self) -> None:
        """Verify token encodes custom room names and participant identities."""
        token = generate_participant_token(
            api_key="my-key",
            api_secret="my-super-secret-key-32-bytes-long!!",
            room_name="custom-room-42",
            identity="test-agent-caller",
            name="Alice Walker",
            ttl_seconds=3600,
        )

        claims = validate_token(token, api_secret="my-super-secret-key-32-bytes-long!!")
        assert claims["iss"] == "my-key"
        assert claims["sub"] == "test-agent-caller"
        assert claims["name"] == "Alice Walker"
        assert claims["video"]["room"] == "custom-room-42"

    def test_invalid_secret_rejection(self) -> None:
        """Verify tokens cannot be validated with an incorrect secret."""
        token = generate_participant_token()
        with pytest.raises(jwt.InvalidSignatureError):
            validate_token(token, api_secret="wrong-secret-key")


class TestWebParticipantAssets:
    def test_web_participant_files_exist(self) -> None:
        """Verify the local web participant assets are present and ready for offline use."""
        static_dir = Path(__file__).resolve().parent.parent / "examples" / "web_participant"
        html_file = static_dir / "index.html"
        js_bundle = static_dir / "livekit-client.umd.min.js"

        assert html_file.exists(), "examples/web_participant/index.html is missing"
        assert js_bundle.exists(), "examples/web_participant/livekit-client.umd.min.js is missing"

        # Verify HTML contains necessary WebRTC connection hooks
        content = html_file.read_text(encoding="utf-8")
        assert "LivekitClient" in content
        assert "setMicrophoneEnabled" in content
        assert "TrackSubscribed" in content
        assert "prism-test" in content

        # Verify JS bundle has valid content
        assert js_bundle.stat().st_size > 100_000
