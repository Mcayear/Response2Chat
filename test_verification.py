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


def replace_http_client(app, handler):
    current_client = app.state.http_client
    if isinstance(current_client, httpx.AsyncClient):
        asyncio.run(current_client.aclose())
    app.state.http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

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
            location = resp.headers.get("location", "")
            if resp.status_code == 303 and location.startswith("/admin?"):
                print("Step 6: POST /admin/channels [PASS]")
            else:
                print(f"Step 6: POST /admin/channels [FAIL] (Status: {resp.status_code}, Location: {location})")

            channels = main.app.state.settings_store.list_channels()
            channel_a = next((c for c in channels if c["name"] == "channel-a"), None)
            if channel_a:
                access_key = channel_a["access_key"]
                channel_id = channel_a["id"]
                print(f"Step 7: Get access_key [PASS]")
            else:
                print("Step 7: Get access_key [FAIL]")
                return

            def mock_handler(request):
                if str(request.url) == "https://example.com/v1/models" and request.headers.get("Authorization") == "Bearer upstream-secret" and request.headers.get("User-Agent") == main.UPSTREAM_USER_AGENT:
                    return httpx.Response(200, json={ "data": [{"id": "model-a"}] })
                return httpx.Response(404)

            original_client = main.app.state.http_client
            replace_http_client(main.app, mock_handler)

            try:
                resp = client.get("/v1/models", headers={"Authorization": f"Bearer {access_key}"})
                if resp.status_code == 200 and resp.json()["data"][0]["id"] == "model-a":
                    print("Step 9: GET /v1/models with valid key [PASS]")
                else:
                    print(f"Step 9: GET /v1/models [FAIL] (Status: {resp.status_code})")

                resp = client.post(
                    f"/admin/channels/{channel_id}/test",
                    data={"return_to": "/admin"},
                    follow_redirects=False,
                )
                location = resp.headers.get("location", "")
                if resp.status_code == 303 and location.startswith("/admin?notice="):
                    notice_page = client.get(location)
                    if notice_page.status_code == 200 and "联通正常" in notice_page.text:
                        print("Step 10: POST /admin/channels/{id}/test [PASS]")
                    else:
                        print(f"Step 10: POST /admin/channels/{{id}}/test [FAIL] (Notice page: {notice_page.status_code})")
                else:
                    print(f"Step 10: POST /admin/channels/{{id}}/test [FAIL] (Status: {resp.status_code})")

                resp = client.get("/v1/models", headers={"Authorization": "Bearer wrong-key"})
                if resp.status_code == 401:
                    print("Step 11: GET /v1/models with wrong key [PASS]")
                else:
                    print(f"Step 11: GET /v1/models with wrong key [FAIL]")

                update_data = {
                    "name": "channel-a",
                    "upstream_base_url": "https://example.com/v1",
                    "upstream_api_key": "",
                    "clear_upstream_api_key": "on",
                    "description": "",
                    "enabled": "on",
                }
                resp = client.post(f"/admin/channels/{channel_id}", data=update_data, follow_redirects=False)
                location = resp.headers.get("location", "")
                updated_channel = main.app.state.settings_store.get_channel(channel_id)
                if resp.status_code == 303 and location.startswith("/admin?") and updated_channel and updated_channel["upstream_api_key"] == "":
                    print("Step 12: Save channel redirects to dashboard [PASS]")
                else:
                    print(f"Step 12: Save channel redirects to dashboard [FAIL] (Status: {resp.status_code})")

                dashboard_resp = client.get("/admin")
                detail_resp = client.get(f"/admin/channels/{channel_id}")
                if "Authorization: Bearer" not in dashboard_resp.text and "Authorization: Bearer" not in detail_resp.text:
                    print("Step 13: Admin pages hide request example [PASS]")
                else:
                    print("Step 13: Admin pages hide request example [FAIL]")

                def no_auth_handler(request):
                    if str(request.url) == "https://example.com/v1/models" and "Authorization" not in request.headers and request.headers.get("User-Agent") == main.UPSTREAM_USER_AGENT:
                        return httpx.Response(200, json={"data": [{"id": "model-no-auth"}]})
                    return httpx.Response(404)

                replace_http_client(main.app, no_auth_handler)
                resp = client.get("/v1/models", headers={"Authorization": f"Bearer {access_key}"})
                if resp.status_code == 200 and resp.json()["data"][0]["id"] == "model-no-auth":
                    print("Step 14: GET /v1/models after clearing upstream key [PASS]")
                else:
                    print(f"Step 14: GET /v1/models after clearing upstream key [FAIL] (Status: {resp.status_code})")

                def non_json_handler(request):
                    if str(request.url) == "https://example.com/v1/models" and request.headers.get("User-Agent") == main.UPSTREAM_USER_AGENT:
                        return httpx.Response(502, text="upstream models unavailable", headers={"Content-Type": "text/plain; charset=utf-8"})
                    return httpx.Response(404)

                replace_http_client(main.app, non_json_handler)
                resp = client.get("/v1/models", headers={"Authorization": f"Bearer {access_key}"})
                if resp.status_code == 502 and resp.text == "upstream models unavailable":
                    print("Step 15: GET /v1/models non-json passthrough [PASS]")
                else:
                    print(f"Step 15: GET /v1/models non-json passthrough [FAIL] (Status: {resp.status_code})")

                def stream_error_handler(request):
                    if str(request.url) == "https://example.com/v1/responses" and request.headers.get("User-Agent") == main.UPSTREAM_USER_AGENT:
                        return httpx.Response(500, json={"error": {"message": "upstream stream failed", "code": "internal_error"}})
                    return httpx.Response(404)

                replace_http_client(main.app, stream_error_handler)
                resp = client.post(
                    "/v1/chat/completions",
                    headers={"Authorization": f"Bearer {access_key}"},
                    json={
                        "model": "gpt-4.1",
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": True,
                    },
                )
                if resp.status_code == 500 and resp.json().get("error", {}).get("message") == "upstream stream failed":
                    print("Step 16: Stream error status passthrough [PASS]")
                else:
                    print(f"Step 16: Stream error status passthrough [FAIL] (Status: {resp.status_code})")
            finally:
                current_client = main.app.state.http_client
                if isinstance(current_client, httpx.AsyncClient):
                    asyncio.run(current_client.aclose())
                main.app.state.http_client = original_client
    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        shutil.rmtree(tmp_db_dir)

if __name__ == "__main__":
    run_test()
