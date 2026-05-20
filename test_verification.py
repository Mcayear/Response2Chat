import os
import sys
import shutil
import tempfile
import asyncio

tmp_db_dir = tempfile.mkdtemp()
tmp_db_path = os.path.join(tmp_db_dir, "test.db")
os.environ["DATABASE_PATH"] = tmp_db_path
os.environ["RESPONSE_API_BASE"] = ""
os.environ["RESPONSE_API_KEY"] = ""
os.environ["RELOAD"] = "false"

import main
import httpx
from fastapi.testclient import TestClient

def run_test():
    print("Starting verification...")
    try:
        with TestClient(main.app) as client:
            resp = client.get("/admin/login")
            if resp.status_code == 200:
                print("Step 4: GET /admin/login == 200 [PASS]")
            else:
                print(f"Step 4: GET /admin/login == 200 [FAIL] (Status: {resp.status_code})")

            login_data = {"username": "admin", "password": "admin123456"}
            resp = client.post("/admin/login", data=login_data, follow_redirects=False)
            
            cookie_name = main.ADMIN_SESSION_COOKIE_NAME
            set_cookie_header = resp.headers.get("Set-Cookie", "")
            
            if resp.status_code == 303 and cookie_name in set_cookie_header and cookie_name in client.cookies:
                print("Step 5: POST /admin/login valid credentials [PASS]")
            else:
                print(f"Step 5: POST /admin/login [FAIL] (Status: {resp.status_code})")

            channel_data = {"name": "channel-a", "upstream_base_url": "https://example.com/v1", "upstream_api_key": "upstream-secret"}
            resp = client.post("/admin/channels", data=channel_data, follow_redirects=False)
            if resp.status_code == 303:
                print("Step 6: POST /admin/channels [PASS]")
            else:
                print(f"Step 6: POST /admin/channels [FAIL] (Status: {resp.status_code})")

            channels = main.app.state.settings_store.list_channels()
            channel_a = next((c for c in channels if c["name"] == "channel-a"), None)
            if channel_a:
                access_key = channel_a["access_key"]
                print(f"Step 7: Get access_key [PASS]")
            else:
                print("Step 7: Get access_key [FAIL]")
                return

            def mock_handler(request):
                if str(request.url) == "https://example.com/v1/models" and request.headers.get("Authorization") == "Bearer upstream-secret":
                    return httpx.Response(200, json={ "data": [{"id": "model-a"}] })
                return httpx.Response(404)

            mock_transport = httpx.MockTransport(mock_handler)
            original_client = main.app.state.http_client
            main.app.state.http_client = httpx.AsyncClient(transport=mock_transport)

            try:
                resp = client.get("/v1/models", headers={"Authorization": f"Bearer {access_key}"})
                if resp.status_code == 200 and resp.json()["data"][0]["id"] == "model-a":
                    print("Step 9: GET /v1/models with valid key [PASS]")
                else:
                    print(f"Step 9: GET /v1/models [FAIL] (Status: {resp.status_code})")

                resp = client.get("/v1/models", headers={"Authorization": "Bearer wrong-key"})
                if resp.status_code == 401:
                    print("Step 10: GET /v1/models with wrong key [PASS]")
                else:
                    print(f"Step 10: GET /v1/models with wrong key [FAIL]")
            finally:
                asyncio.run(main.app.state.http_client.aclose())
                main.app.state.http_client = original_client
    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        shutil.rmtree(tmp_db_dir)

if __name__ == "__main__":
    run_test()
