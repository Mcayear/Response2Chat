import asyncio
import httpx
import json
import os
from fastapi.testclient import TestClient
from main import app, SettingsStore

async def run_test():
    # 1. 初始化设置存储
    db_path = "test_500.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    store = SettingsStore(
        database_path=db_path,
        default_admin_username="admin",
        default_admin_password="password"
    )
    store.initialize()
    
    # 2. 添加测试渠道
    # 注意: SettingsStore.add_channel 的参数可能需要根据实际代码调整
    # 假设它是 store.create_channel(name, api_base, api_key, models)
    # 通过查看 settings_store 的源码来确认
    try:
        channel_id = store.create_channel(
            name="Test 500 Channel",
            api_base="https://upstream.api",
            api_key="sk-test",
            models="*"
        )
    except AttributeError:
        # 如果方法名不同，尝试 list_channels 等来猜测，或者直接操作数据库
        # 这里我们假设它有 create_channel
        print("AttributeError: create_channel not found. Checking SettingsStore methods...")
        import inspect
        print([m[0] for m in inspect.getmembers(store, predicate=inspect.ismethod)])
        return

    # 3. 定制 MockTransport
    def handle_request(request):
        if "/responses" in str(request.url):
            return httpx.Response(500, content=json.dumps({"error": "Mock internal server error"}).encode())
        return httpx.Response(200, content=b"{}")

    mock_transport = httpx.MockTransport(handle_request)

    # 4. 使用 TestClient 并在全局打桩 httpx.AsyncClient
    from unittest.mock import patch
    
    app.state.settings_store = store
    
    with patch("httpx.AsyncClient", lambda **kwargs: httpx.AsyncClient(transport=mock_transport, **kwargs)):
        client = TestClient(app)
        
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True
        }
        headers = {
            # 假设渠道是通过密钥使用的，或者通过 X-Channel-Id
            "Authorization": "Bearer " + store.get_channel(channel_id)["access_key"]
        }
        
        print(f"Sending request to /v1/chat/completions (stream=true) using key {headers['Authorization']}...")
        response = client.post("/v1/chat/completions", json=payload, headers=headers)
        
        print(f"Response Status Code: {response.status_code}")
        print(f"Response Content: {response.text}")
        
        if response.status_code == 500:
            print("SUCCESS: Received HTTP 500 as expected.")
        else:
            print(f"FAILURE: Received HTTP {response.status_code} instead of 500.")
    
    if os.path.exists(db_path):
        os.remove(db_path)

if __name__ == "__main__":
    asyncio.run(run_test())
