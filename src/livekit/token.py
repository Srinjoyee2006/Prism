"""
LiveKit Access Token Generation Utility.

Generates local development JWT access tokens for connecting participants
to LiveKit rooms (e.g., 'prism-test') using local dev credentials.
"""

import os
from typing import Any

from dotenv import load_dotenv

from livekit import api

load_dotenv()

DEFAULT_LIVEKIT_URL = "ws://127.0.0.1:7880"
DEFAULT_API_KEY = "devkey"
DEFAULT_API_SECRET = "secret"
DEFAULT_ROOM = "prism-test"
DEFAULT_IDENTITY = "human-participant"


def generate_participant_token(
    api_key: str | None = None,
    api_secret: str | None = None,
    room_name: str | None = None,
    identity: str | None = None,
    name: str = "Human Participant",
    ttl_seconds: int = 86400,
) -> str:
    """Generate a signed JWT token granting participant access to a room.

    Args:
        api_key: LiveKit API key (defaults to LIVEKIT_API_KEY env or 'devkey').
        api_secret: LiveKit API secret (defaults to LIVEKIT_API_SECRET env or 'secret').
        room_name: Target room name (defaults to 'prism-test').
        identity: Unique participant identity string (defaults to 'human-participant').
        name: Human-readable display name.
        ttl_seconds: Token validity duration in seconds (default 24 hours).

    Returns:
        Encoded JWT token string.
    """
    key = api_key or os.getenv("LIVEKIT_API_KEY", DEFAULT_API_KEY)
    secret = api_secret or os.getenv("LIVEKIT_API_SECRET", DEFAULT_API_SECRET)
    room = room_name or DEFAULT_ROOM
    user_id = identity or DEFAULT_IDENTITY

    # Video grants: allow room join, publish, and subscribe
    grants = api.VideoGrants(
        room_join=True,
        room=room,
        can_publish=True,
        can_subscribe=True,
        can_publish_data=True,
    )

    import datetime

    token = (
        api.AccessToken(key, secret)
        .with_identity(user_id)
        .with_name(name)
        .with_grants(grants)
        .with_ttl(datetime.timedelta(seconds=ttl_seconds))
    )

    return token.to_jwt()


def validate_token(jwt_token: str, api_secret: str | None = None) -> dict[str, Any]:
    """Validate and decode a LiveKit JWT token using the API secret.

    Returns:
        The decoded payload dictionary.
    """
    import jwt

    secret = api_secret or os.getenv("LIVEKIT_API_SECRET", DEFAULT_API_SECRET)
    decoded = jwt.decode(
        jwt_token,
        secret,
        algorithms=["HS256"],
        options={"verify_signature": True},
    )
    return decoded


if __name__ == "__main__":
    jwt_str = generate_participant_token()
    print("=" * 65)
    print("  LIVEKIT PARTICIPANT ACCESS TOKEN (DEVELOPMENT)")
    print("=" * 65)
    print(f"Room:     {DEFAULT_ROOM}")
    print(f"Identity: {DEFAULT_IDENTITY}")
    print(f"Server:   {os.getenv('LIVEKIT_URL', DEFAULT_LIVEKIT_URL)}")
    print(f"Token:    {jwt_str}")
    print("=" * 65)
