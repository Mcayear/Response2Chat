import os
import shutil
import tempfile
import importlib
import sys
from fastapi.testclient import TestClient
import httpx

def run_test():
    # 1. 创建临时目录并设置环境变量
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, 'test.db')
    os.environ['DATABASE_PATH'] = db_path
    os.environ['ADMIN_USERNAME'] = 'admin'
    os.environ['ADMIN_PASSWORD'] = 'admin123456'
    
    print(f"--- 1. Temp dir and env vars set: {tmp_dir}")
    
    try:
        # 2. 重新导入 main 模块
        if 'main' in sys.modules:
            del sys.modules['main']
        import main
        importlib.reload(main)
        print("--- 2. main module reloaded")
        
        # 3. 使用 lifespan 触发 TestClient
        with TestClient(main.app) as client:
            print("--- 3. Lifespan triggered")
            
            # GET /admin/login 返回 200
            resp = client.get("/admin/login")
            print(f"--- 4. GET /admin/login status: {resp.status_code}")
            assert resp.status_code == 200, "GET /admin/login failed"
            
            # 4. POST /admin/login
            resp = client.post("/admin/login", data={"username": "admin", "password": "admin123456"}, follow_redirects=False)
            print(f"--- 5. POST /admin/login status: {resp.status_code}")
            assert resp.status_code == 303, "POST /admin/login should return 303"
            cookie_name = main.ADMIN_SESSION_COOKIE_NAME
            cookie = client.cookies.get(cookie_name) or resp.cookies.get(cookie_name)
            assert cookie is not None, f"Admin session cookie not found: {cookie_name}"
            
            # 5. POST /admin/channels 创建渠道
            resp = client.post("/admin/channels", 
                               data={"name": "channel-a", "upstream_base_url": "https://example.com/v1", "upstream_api_key": "upstream-secret"}, 
                               cookies={cookie_name: cookie},
                               follow_redirects=False)
            print(f"--- 6. POST /admin/channels status: {resp.status_code}")
            assert resp.status_code == 303, "POST /admin/channels should return 303"
            assert resp.headers.get("location", "").startswith("/admin?"), f"POST /admin/channels should redirect to /admin, got {resp.headers.get('location')}"
            
            # 6. 读取 access_key
            channels = client.app.state.settings_store.list_channels()
            channel = next(c for c in channels if c['name'] == 'channel-a')
            access_key = channel['access_key']
            print(f"--- 7. Access key obtained: {access_key}")
            
            # 7. MockTransport
            def mock_handler(request):
                if str(request.url) == "https://example.com/v1/models" and request.headers.get("Authorization") == "Bearer upstream-secret":
                    return httpx.Response(200, json={"data": [{"id": "gpt-3.5-turbo"}]})
                return httpx.Response(404)
            
            client.app.state.http_client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
            print("--- 8. Mocking http_client")
            
            # 8. GET /v1/models with correct key
            resp = client.get("/v1/models", headers={"Authorization": f"Bearer {access_key}"})
            print(f"--- 9. GET /v1/models (correct key) status: {resp.status_code}")
            assert resp.status_code == 200, "GET /v1/models failed with correct key"
            
            # 9. GET /v1/models with wrong key
            resp = client.get("/v1/models", headers={"Authorization": "Bearer wrong-key"})
            print(f"--- 10. GET /v1/models (wrong key) status: {resp.status_code}")
            assert resp.status_code == 401, "GET /v1/models should return 401 for wrong key"
            
            print("--- All steps passed!")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    run_test()
