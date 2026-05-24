import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock
import httpx
from fastapi.testclient import TestClient
import sqlite3

def run_test():
    # 1. 临时 DATABASE_PATH 并重新导入 main
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test.db")
    os.environ["DATABASE_PATH"] = db_path
    
    try:
        import main
        import importlib
        importlib.reload(main)
        
        from main import app, ADMIN_SESSION_COOKIE_NAME
        
        # NOTE: TestClient(app) triggers startup event which creates the DB
        with TestClient(app) as client:
            print("Step 1: Environment setup and import - PASS")
            
            # 2. GET /admin/login == 200
            resp = client.get("/admin/login")
            if resp.status_code == 200:
                print("Step 2: GET /admin/login == 200 - PASS")
            else:
                print(f"Step 2: FAIL - Status code {resp.status_code}")
                return

            # 3. POST /admin/login 用 admin/admin123456
            resp = client.post("/admin/login", data={"username": "admin", "password": "admin123456"}, follow_redirects=False)
            
            is_303 = resp.status_code == 303
            cookie_in_header = any(ADMIN_SESSION_COOKIE_NAME in h for h in resp.headers.get_list("set-cookie"))
            cookie_in_client = ADMIN_SESSION_COOKIE_NAME in client.cookies
            
            if is_303 and cookie_in_header and cookie_in_client:
                print(f"Step 3: POST /admin/login 303 & Cookie - PASS")
            else:
                print(f"Step 3: FAIL - Status: {resp.status_code}, Cookie in header: {cookie_in_header}, Cookie in client: {cookie_in_client}")
                return

            # 4. POST /admin/channels 创建 channel-a
            # We set upstream_base_url to something that will be appended with /models
            channel_data = {
                "name": "channel-a",
                "upstream_base_url": "https://example.com/v1",
                "upstream_api_key": "upstream-secret"
            }
            resp = client.post("/admin/channels", data=channel_data, follow_redirects=False)
            if resp.status_code == 303:
                print("Step 4: POST /admin/channels 303 - PASS")
            else:
                print(f"Step 4: FAIL - Status {resp.status_code}")
                return

            # 5. 从 client.app.state.settings_store.list_channels() 拿到 access_key
            channels = client.app.state.settings_store.list_channels()
            channel_a = next((c for c in channels if (c.get("name") if isinstance(c, dict) else getattr(c, "name")) == "channel-a"), None)
            if channel_a:
                access_key = channel_a.get("access_key") if isinstance(channel_a, dict) else getattr(channel_a, "access_key")
                print(f"Step 5: Access key retrieved - PASS")
            else:
                print(f"Step 5: FAIL - could not find channel-a among {len(channels)} channels")
                return

            # 6. 用 MockTransport 替换 client.app.state.http_client
            async def handler(request):
                # We expect the URL to be base_url + /models
                if str(request.url) == "https://example.com/v1/models" and request.headers.get("Authorization") == "Bearer upstream-secret":
                    return httpx.Response(200, json={"data": [{"id": "gpt-3.5-turbo"}]})
                return httpx.Response(404, text="Not Found")
            
            mock_transport = httpx.MockTransport(handler)
            # Re-creating the AsyncClient with the mock transport
            client.app.state.http_client = httpx.AsyncClient(transport=mock_transport)
            print("Step 6: MockTransport setup - PASS")

            # 7. 用 access_key 调 GET /v1/models
            resp = client.get("/v1/models", headers={"Authorization": f"Bearer {access_key}"})
            if resp.status_code == 200 and "gpt-3.5-turbo" in resp.text:
                print("Step 7: GET /v1/models 200 - PASS")
            else:
                print(f"Step 7: FAIL - Status {resp.status_code}, Body: {resp.text}")
                return

            # 8. 用错误 access key 调 GET /v1/models
            resp = client.get("/v1/models", headers={"Authorization": "Bearer wrong-key"})
            if resp.status_code == 401:
                print("Step 8: Error access key 401 - PASS")
            else:
                print(f"Step 8: FAIL - Status {resp.status_code}")
                return

    except Exception as e:
        print(f"Unexpected Exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
             import main
             if hasattr(main, 'app') and hasattr(main.app.state, 'settings_store'):
                 main.app.state.settings_store.db.close()
        except:
             pass
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

if __name__ == "__main__":
    run_test()
