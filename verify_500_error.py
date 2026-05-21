import asyncio
import httpx
import json
import os
import sys
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Import app and stores
from main import app
from channel_store import SettingsStore

async def run_test():
    db_path = "test_verify_500.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # Initialize store with required arguments
    store = SettingsStore(
        database_path=db_path,
        default_admin_username="admin",
        default_admin_password="password"
    )
    store.initialize()
    
    try:
        # Create a test channel
        channel = store.create_channel(
            name="Test500",
            base_url="https://upstream.api/v1",
            upstream_api_key="sk-test-key",
            description="Testing 500 mapping"
        )
        channel_id = channel['id']
        print(f"Created channel: {channel_id}")
        
        channel_details = store.get_channel(channel_id)
        access_key = channel_details.get('access_key')
        print(f"Access Key: {access_key}")

        # Define Mock Transport
        def handle_request(request):
            print(f"Mock intercepted: {request.url}")
            return httpx.Response(
                500, 
                content=json.dumps({"error": {"message": "Mock Internal Error", "type": "server_error"}}).encode(),
                headers={"Content-Type": "application/json"}
            )

        mock_transport = httpx.MockTransport(handle_request)
        
        app.state.settings_store = store
        
        with patch("httpx.AsyncClient", side_effect=lambda **kwargs: httpx.AsyncClient(transport=mock_transport, **kwargs)):
            client = TestClient(app)
            
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True
            }
            headers = {
                "Authorization": f"Bearer {access_key}"
            }
            
            print("Sending POST /v1/chat/completions (stream=true)...")
            response = client.post("/v1/chat/completions", json=payload, headers=headers)
            
            print(f"Result Status Code: {response.status_code}")
            print(f"Result Body: {response.text}")
            
            if response.status_code == 500:
                print("VERIFICATION SUCCESS: Proxy returned 500.")
            else:
                print(f"VERIFICATION FAILURE: Proxy returned {response.status_code}.")
                
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)

if __name__ == '__main__':
    asyncio.run(run_test())
