"""
WebSocket signaling relay for VoiceGuard's peer-to-peer "Protected Call"
feature (see docs/superpowers/plans/2026-09-08-voice-guard-self-owned-voip-call.md).

Exactly two clients per room_id: whatever one sends (SDP offer/answer, ICE
candidates) is relayed verbatim to the other. No message inspection, no
persistence — this is a dumb pipe. Room capacity of 2 keeps the demo's
threat model simple: no third party can join and MITM without knowing (and
occupying before the second caller) the shared room_id.
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

signaling_router = APIRouter()

_rooms: dict[str, list[WebSocket]] = {}


@signaling_router.websocket("/v1/signal/{room_id}")
async def signal(websocket: WebSocket, room_id: str):
    await websocket.accept()
    peers = _rooms.setdefault(room_id, [])
    if len(peers) >= 2:
        await websocket.close(code=4000, reason="room full")
        return
    peers.append(websocket)
    try:
        while True:
            msg = await websocket.receive_json()
            for peer in peers:
                if peer is not websocket:
                    await peer.send_json(msg)
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in peers:
            peers.remove(websocket)
        if not peers:
            _rooms.pop(room_id, None)
