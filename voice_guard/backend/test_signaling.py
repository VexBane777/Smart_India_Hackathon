import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_two_clients_relay_messages_to_each_other():
    with client.websocket_connect("/v1/signal/room1") as caller, \
         client.websocket_connect("/v1/signal/room1") as callee:
        caller.send_json({"type": "offer", "sdp": "fake-sdp"})
        msg = callee.receive_json()
        assert msg == {"type": "offer", "sdp": "fake-sdp"}

        callee.send_json({"type": "answer", "sdp": "fake-answer"})
        msg = caller.receive_json()
        assert msg == {"type": "answer", "sdp": "fake-answer"}

def test_third_client_to_occupied_room_is_rejected():
    with client.websocket_connect("/v1/signal/room2"):
        with client.websocket_connect("/v1/signal/room2"):
            with pytest.raises(Exception):
                with client.websocket_connect("/v1/signal/room2") as third:
                    third.receive_json()
