"""
Response API to Chat API 转发服务
将 OpenAI Response 协议接口转发为 Chat 协议接口
"""

import os
import json
import time
import uuid
import asyncio
import logging
import traceback
from html import escape
from typing import Optional, List, Dict, Any, Union
from contextlib import asynccontextmanager
from urllib.parse import parse_qs, quote

import httpx
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import StreamingResponse, JSONResponse, Response, HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from channel_store import AdminSessionManager, SettingsStore, mask_secret

load_dotenv()

# ==================== 日志配置 ====================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("response2chat")

# ==================== 配置 ====================
RESPONSE_API_BASE = os.getenv("RESPONSE_API_BASE", "").strip()
RESPONSE_API_KEY = os.getenv("RESPONSE_API_KEY", "").strip()
DEFAULT_TIMEOUT = int(os.getenv("DEFAULT_TIMEOUT", "300"))
POOL_TIMEOUT = float(os.getenv("POOL_TIMEOUT", "10"))
STREAM_READ_TIMEOUT = float(os.getenv("STREAM_READ_TIMEOUT", "120"))
STREAM_MAX_DURATION = int(os.getenv("STREAM_MAX_DURATION", "0"))  # 0 表示不限制
DATABASE_PATH = os.getenv("DATABASE_PATH", "data/response2chat.db")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin").strip() or "admin"
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123456")
ADMIN_SESSION_TTL_SECONDS = int(os.getenv("ADMIN_SESSION_TTL_SECONDS", str(12 * 60 * 60)))
ADMIN_SESSION_COOKIE_NAME = os.getenv("ADMIN_SESSION_COOKIE_NAME", "response2chat_admin_session")
ADMIN_COOKIE_SECURE = os.getenv("ADMIN_COOKIE_SECURE", "false").lower() == "true"
BOOTSTRAP_CHANNEL_NAME = os.getenv("BOOTSTRAP_CHANNEL_NAME", "默认渠道")

# 连接池配置 - 防止连接泄漏和资源耗尽
MAX_CONNECTIONS = int(os.getenv("MAX_CONNECTIONS", "100"))  # 最大连接数
MAX_KEEPALIVE_CONNECTIONS = int(os.getenv("MAX_KEEPALIVE_CONNECTIONS", "30"))  # 保持活跃的连接数
KEEPALIVE_EXPIRY = int(os.getenv("KEEPALIVE_EXPIRY", "60"))  # 连接保持时间(秒)

# 默认系统提示词配置
# 当请求中没有 system 消息时，会使用此默认提示词
# 设置为空字符串可禁用默认提示词
DEFAULT_INSTRUCTIONS = os.getenv("DEFAULT_INSTRUCTIONS", "").strip()
# 是否强制使用默认提示词（即使请求中有 system 消息也会添加）
FORCE_DEFAULT_INSTRUCTIONS = os.getenv("FORCE_DEFAULT_INSTRUCTIONS", "false").lower() == "true"

# ==================== Pydantic 模型定义 ====================

# Chat API 请求模型
class ChatMessage(BaseModel):
    role: str
    content: Optional[Union[str, List[Dict[str, Any]]]] = None  # 允许为 None，当有 tool_calls 时可能为空
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

class StreamOptions(BaseModel):
    include_usage: Optional[bool] = False

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=1, ge=0, le=2)
    top_p: Optional[float] = Field(default=1, ge=0, le=1)
    n: Optional[int] = Field(default=1, ge=1)
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    presence_penalty: Optional[float] = Field(default=0, ge=-2, le=2)
    frequency_penalty: Optional[float] = Field(default=0, ge=-2, le=2)
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    response_format: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None
    reasoning_effort: Optional[str] = None

# Chat API 响应模型
class ChatCompletionChoice(BaseModel):
    index: int
    message: Dict[str, Any]
    finish_reason: Optional[str] = "stop"

class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens_details: Optional[Dict[str, Any]] = None
    completion_tokens_details: Optional[Dict[str, Any]] = None


def convert_response_usage_to_chat_usage(response_usage: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    将 Response API 的 usage 格式转换为 Chat API 的 usage 格式
    
    Response API 格式:
    {
        "input_tokens": 17254,
        "input_tokens_details": {"cached_tokens": 7936},
        "output_tokens": 336,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 17590
    }
    
    Chat API 格式:
    {
        "prompt_tokens": 12709,
        "prompt_tokens_details": {
            "audio_tokens": 0,
            "cached_tokens": 12032
        },
        "completion_tokens": 322,
        "completion_tokens_details": {
            "accepted_prediction_tokens": 0,
            "audio_tokens": 0,
            "reasoning_tokens": 0,
            "rejected_prediction_tokens": 0
        },
        "total_tokens": 13031
    }
    """
    if response_usage is None:
        return None
    
    # 基本字段转换: input_tokens -> prompt_tokens, output_tokens -> completion_tokens
    chat_usage = {
        "prompt_tokens": response_usage.get("input_tokens", 0),
        "completion_tokens": response_usage.get("output_tokens", 0),
        "total_tokens": response_usage.get("total_tokens", 0)
    }
    
    # 转换 input_tokens_details -> prompt_tokens_details
    input_details = response_usage.get("input_tokens_details")
    if input_details:
        chat_usage["prompt_tokens_details"] = {
            "audio_tokens": input_details.get("audio_tokens", 0),
            "cached_tokens": input_details.get("cached_tokens", 0)
        }
    
    # 转换 output_tokens_details -> completion_tokens_details
    output_details = response_usage.get("output_tokens_details")
    if output_details:
        chat_usage["completion_tokens_details"] = {
            "accepted_prediction_tokens": output_details.get("accepted_prediction_tokens", 0),
            "audio_tokens": output_details.get("audio_tokens", 0),
            "reasoning_tokens": output_details.get("reasoning_tokens", 0),
            "rejected_prediction_tokens": output_details.get("rejected_prediction_tokens", 0)
        }
    
    return chat_usage

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[UsageInfo] = None
    system_fingerprint: Optional[str] = None

# ==================== 转换函数 ====================

def convert_chat_to_response_request(chat_request: ChatCompletionRequest) -> Dict[str, Any]:
    """将 Chat API 请求转换为 Response API 请求"""
    
    # 构建 input 数组，包含所有消息
    # 注意：Response API 不支持 system 角色，将其转换为 developer 角色
    # Response API 不支持 tool 角色，需要转换为 function_call_output 类型
    input_items = []
    
    # 预处理：为空的 tool_call.id 和对应的 tool.tool_call_id 建立映射
    # 按顺序匹配：每个 assistant 的 tool_calls 后面紧跟着对应数量的 tool 消息
    generated_call_ids: Dict[int, str] = {}  # 消息索引 -> 生成的 call_id
    tool_call_id_mapping: Dict[int, List[str]] = {}  # assistant 消息索引 -> 该消息生成的 call_ids 列表
    
    # 第一遍扫描：识别需要生成 call_id 的 tool_calls，并建立映射
    pending_call_ids: List[str] = []  # 待匹配的 call_ids 队列
    for i, msg in enumerate(chat_request.messages):
        if msg.role == "assistant" and msg.tool_calls:
            tool_call_id_mapping[i] = []
            for tool_call in msg.tool_calls:
                original_id = tool_call.get("id")
                if original_id:
                    # 有原始 id，直接使用
                    pending_call_ids.append(original_id)
                    tool_call_id_mapping[i].append(original_id)
                else:
                    # 没有原始 id，生成一个新的
                    new_id = f"call_{uuid.uuid4().hex[:24]}"
                    pending_call_ids.append(new_id)
                    tool_call_id_mapping[i].append(new_id)
                    logger.warning(f"tool_call 的 id 为空，自动生成: {new_id}")
        elif msg.role == "tool":
            # tool 消息需要匹配 call_id
            if msg.tool_call_id:
                # 有原始 tool_call_id，直接使用
                generated_call_ids[i] = msg.tool_call_id
            elif pending_call_ids:
                # 没有 tool_call_id，从队列中取一个
                generated_call_ids[i] = pending_call_ids.pop(0)
                logger.warning(f"tool 消息的 tool_call_id 为空，使用匹配的 call_id: {generated_call_ids[i]}")
            else:
                # 队列为空，生成一个新的（这种情况不应该发生）
                generated_call_ids[i] = f"call_{uuid.uuid4().hex[:24]}"
                logger.warning(f"tool 消息的 tool_call_id 为空且无法匹配，自动生成: {generated_call_ids[i]}")
    
    # 重置 pending_call_ids 用于第二遍
    pending_call_ids_iter = iter([])
    current_assistant_idx = -1
    current_tool_call_idx = 0
    
    # 第二遍：实际构建 input_items
    for i, msg in enumerate(chat_request.messages):
        # 特殊处理 tool 角色 - 转换为 function_call_output 类型
        if msg.role == "tool":
            # Chat API tool 消息格式:
            # {"role": "tool", "tool_call_id": "xxx", "content": "result"}
            # -> Response API 格式:
            # {"type": "function_call_output", "call_id": "xxx", "output": "result"}
            call_id = generated_call_ids.get(i, msg.tool_call_id or f"call_{uuid.uuid4().hex[:24]}")
            tool_output_item = {
                "type": "function_call_output",
                "call_id": call_id,
                "output": msg.content if isinstance(msg.content, str) else json.dumps(msg.content) if msg.content else ""
            }
            input_items.append(tool_output_item)
            continue
        
        # 特殊处理 assistant 消息中的 tool_calls
        if msg.role == "assistant" and msg.tool_calls:
            # 先添加 assistant 消息内容（如果有）
            if msg.content:
                content = msg.content if isinstance(msg.content, str) else msg.content
                item = {
                    "type": "message",
                    "role": "assistant",
                    "content": content
                }
                input_items.append(item)
            
            # 获取预先生成的 call_ids
            pre_generated_ids = tool_call_id_mapping.get(i, [])
            
            # 然后添加 function_call 类型的项
            # Chat API tool_calls 格式:
            # [{"id": "call_xxx", "type": "function", "function": {"name": "xxx", "arguments": "{...}"}}]
            # -> Response API 格式:
            # {"type": "function_call", "call_id": "call_xxx", "name": "xxx", "arguments": "{...}"}
            for j, tool_call in enumerate(msg.tool_calls):
                # 处理 type 为 function 或 None 的情况（某些客户端可能不发送 type 字段）
                tool_type = tool_call.get("type")
                if tool_type == "function" or tool_type is None:
                    func = tool_call.get("function", {})
                    # 使用预先生成的 call_id
                    if j < len(pre_generated_ids):
                        call_id = pre_generated_ids[j]
                    else:
                        call_id = tool_call.get("id") or f"call_{uuid.uuid4().hex[:24]}"
                    
                    # 如果 name 为空，尝试从 arguments 中推断工具名称
                    func_name = func.get("name", "")
                    if not func_name:
                        # 尝试从 arguments 推断工具名称
                        args_str = func.get("arguments", "{}")
                        try:
                            args_dict = json.loads(args_str) if isinstance(args_str, str) else args_str
                            # 常见工具参数到工具名的映射
                            if "thought" in args_dict:
                                func_name = "think"
                            elif "code" in args_dict and "file_name" in args_dict:
                                func_name = "save_to_file_and_run"
                            else:
                                func_name = f"unknown_function_{uuid.uuid4().hex[:8]}"
                        except:
                            func_name = f"unknown_function_{uuid.uuid4().hex[:8]}"
                        logger.warning(f"tool_call 的 name 为空，推断为: {func_name}")
                    
                    func_call_item = {
                        "type": "function_call",
                        "call_id": call_id,
                        "name": func_name,
                        "arguments": func.get("arguments", "{}")
                    }
                    input_items.append(func_call_item)
            continue
        
        # 处理 content 字段，转换多模态内容格式
        if msg.content is None:
            # content 为空（通常在 assistant 消息有 tool_calls 时）
            converted_content = ""
        elif isinstance(msg.content, str):
            # 纯文本内容
            converted_content = msg.content
        elif isinstance(msg.content, list):
            # 多模态内容，需要转换格式
            converted_content = []
            for part in msg.content:
                if isinstance(part, dict):
                    part_type = part.get("type", "")
                    if part_type == "text":
                        # 文本部分: Chat 格式 {"type": "text", "text": "..."} 
                        # -> Response 格式 {"type": "input_text", "text": "..."}
                        converted_content.append({
                            "type": "input_text",
                            "text": part.get("text", "")
                        })
                    elif part_type == "image_url":
                        # 图片部分: Chat 格式 {"type": "image_url", "image_url": {"url": "..."}}
                        # -> Response 格式 {"type": "input_image", "image_url": "..."}
                        image_url_obj = part.get("image_url", {})
                        if isinstance(image_url_obj, dict):
                            image_url = image_url_obj.get("url", "")
                        else:
                            image_url = str(image_url_obj)
                        converted_content.append({
                            "type": "input_image",
                            "image_url": image_url
                        })
                    else:
                        # 其他类型直接保留
                        converted_content.append(part)
                else:
                    converted_content.append(part)
        else:
            converted_content = msg.content
        
        # 检查角色类型
        # Response API 不支持 system 角色，将其转换为 developer 角色
        role = msg.role
        if role == "system":
            role = "developer"
        
        item = {
            "type": "message",
            "role": role,
            "content": converted_content
        }
        input_items.append(item)
    
    response_request = {
        "model": chat_request.model,
        "input": input_items,
        "stream": True,  # Response API 始终使用 stream
    }
    
    # 处理 instructions 参数
    # 检查请求中是否已有 system 消息
    has_system_message = any(msg.role == "system" for msg in chat_request.messages)
    
    if DEFAULT_INSTRUCTIONS:
        if FORCE_DEFAULT_INSTRUCTIONS or not has_system_message:
            # 使用配置的默认 instructions
            response_request["instructions"] = DEFAULT_INSTRUCTIONS
            logger.debug(f"使用默认 instructions: {DEFAULT_INSTRUCTIONS[:50]}...")
    
    # 可选参数映射 - 只添加 Response API 支持的参数
    # 注意：某些 Response API 可能不支持 temperature, top_p, max_output_tokens 等参数
    # 根据实际 API 支持情况调整
    # max_output_tokens 参数已注释，因为某些上游 API (如 api.routin.ai) 不支持此参数
    # 如需启用，取消以下注释：
    # if chat_request.max_tokens is not None:
    #     response_request["max_output_tokens"] = chat_request.max_tokens
    # if chat_request.max_completion_tokens is not None:
    #     response_request["max_output_tokens"] = chat_request.max_completion_tokens
    
    # tools 格式转换
    # Chat API 格式: {"type": "function", "function": {"name": "xxx", "description": "xxx", "parameters": {...}}}
    # Response API 格式: {"type": "function", "name": "xxx", "description": "xxx", "parameters": {...}}
    if chat_request.tools is not None:
        converted_tools = []
        for tool in chat_request.tools:
            if tool.get("type") == "function" and "function" in tool:
                func = tool["function"]
                converted_tool = {
                    "type": "function",
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                }
                if "parameters" in func:
                    converted_tool["parameters"] = func["parameters"]
                converted_tools.append(converted_tool)
            else:
                # 其他类型直接保留
                converted_tools.append(tool)
        response_request["tools"] = converted_tools
    
    if chat_request.tool_choice is not None:
        response_request["tool_choice"] = chat_request.tool_choice
    
    # reasoning_effort 用于推理模型
    if chat_request.reasoning_effort is not None:
        response_request["reasoning"] = {"effort": chat_request.reasoning_effort}
    
    # response_format 支持 (如 json_object, json_schema)
    if chat_request.response_format is not None:
        # Response API 可能使用不同的格式，尝试转换
        fmt_type = chat_request.response_format.get("type")
        if fmt_type == "json_object":
            response_request["text"] = {"format": {"type": "json_object"}}
        elif fmt_type == "json_schema":
            # Chat API json_schema 格式:
            # {"type": "json_schema", "json_schema": {"name": "xxx", "schema": {...}, "strict": true}}
            # Response API 格式:
            # {"format": {"type": "json_schema", "name": "xxx", "schema": {...}, "strict": true}}
            json_schema_obj = chat_request.response_format.get("json_schema", {})
            response_format = {
                "type": "json_schema",
                "name": json_schema_obj.get("name", "response_schema"),
                "schema": json_schema_obj.get("schema", {}),
            }
            # 只有在 strict 存在时才添加
            if "strict" in json_schema_obj:
                response_format["strict"] = json_schema_obj.get("strict")
            response_request["text"] = {"format": response_format}
    
    # 以下参数某些 Response API 可能不支持，根据需要启用
    # if chat_request.temperature is not None and chat_request.temperature != 1:
    #     response_request["temperature"] = chat_request.temperature
    # if chat_request.top_p is not None and chat_request.top_p != 1:
    #     response_request["top_p"] = chat_request.top_p
    # if chat_request.stop is not None:
    #     response_request["stop"] = chat_request.stop
    # if chat_request.presence_penalty is not None and chat_request.presence_penalty != 0:
    #     response_request["presence_penalty"] = chat_request.presence_penalty
    # if chat_request.frequency_penalty is not None and chat_request.frequency_penalty != 0:
    #     response_request["frequency_penalty"] = chat_request.frequency_penalty
    
    return response_request


def generate_chat_id() -> str:
    """生成 Chat Completion ID"""
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def create_chat_stream_chunk(
    chunk_id: str,
    model: str,
    delta: Dict[str, Any],
    index: int = 0,
    finish_reason: Optional[str] = None,
    usage: Optional[Dict[str, Any]] = None
) -> str:
    """创建流式响应的 chunk"""
    chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": index,
                "delta": delta,
                "finish_reason": finish_reason
            }
        ]
    }
    if usage is not None:
        chunk["usage"] = usage
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


# ==================== 流式响应处理 ====================

class ResponseStreamProcessor:
    """处理 Response API 的流式响应"""
    
    def __init__(self, chat_id: str, model: str, include_usage: bool = False):
        self.chat_id = chat_id
        self.model = model
        self.include_usage = include_usage
        self.accumulated_content = ""
        self.accumulated_reasoning = ""
        self.usage = None
        self.is_first_content = True
        self.current_output_index = None
        self.tool_calls = []
        self.current_tool_call = None
    
    def process_event(self, event_type: str, event_data: Dict[str, Any]) -> List[str]:
        """处理单个 SSE 事件，返回要发送的 Chat chunks"""
        chunks = []
        
        if event_type == "response.created":
            # 发送开始的 role delta
            chunks.append(create_chat_stream_chunk(
                self.chat_id, self.model,
                {"role": "assistant", "content": ""}
            ))
        
        elif event_type == "response.output_item.added":
            # 新的输出项开始
            item = event_data.get("item", {})
            self.current_output_index = event_data.get("output_index", 0)
            if item.get("type") == "function_call":
                # 工具调用开始
                self.current_tool_call = {
                    "id": item.get("call_id", f"call_{uuid.uuid4().hex[:8]}"),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": ""
                    }
                }
        
        elif event_type == "response.output_text.delta":
            # 文本增量
            delta_text = event_data.get("delta", "")
            if delta_text:
                self.accumulated_content += delta_text
                chunks.append(create_chat_stream_chunk(
                    self.chat_id, self.model,
                    {"content": delta_text}
                ))
        
        elif event_type == "response.content_part.delta":
            # 内容部分增量（另一种格式）
            delta = event_data.get("delta", {})
            if delta.get("type") == "text_delta":
                delta_text = delta.get("text", "")
                if delta_text:
                    self.accumulated_content += delta_text
                    chunks.append(create_chat_stream_chunk(
                        self.chat_id, self.model,
                        {"content": delta_text}
                    ))
        
        elif event_type == "response.reasoning_summary_text.delta":
            # 推理内容增量
            delta_text = event_data.get("delta", "")
            if delta_text:
                self.accumulated_reasoning += delta_text
                # 推理内容作为 reasoning_content 字段
                chunks.append(create_chat_stream_chunk(
                    self.chat_id, self.model,
                    {"reasoning_content": delta_text}
                ))
        
        elif event_type == "response.function_call_arguments.delta":
            # 函数调用参数增量
            delta_args = event_data.get("delta", "")
            if self.current_tool_call and delta_args:
                self.current_tool_call["function"]["arguments"] += delta_args
                # 发送工具调用增量
                tool_call_delta = {
                    "tool_calls": [{
                        "index": len(self.tool_calls),
                        "function": {"arguments": delta_args}
                    }]
                }
                chunks.append(create_chat_stream_chunk(
                    self.chat_id, self.model,
                    tool_call_delta
                ))
        
        elif event_type == "response.function_call_arguments.done":
            # 函数调用完成
            if self.current_tool_call:
                self.tool_calls.append(self.current_tool_call)
                self.current_tool_call = None
        
        elif event_type == "response.output_item.done":
            # 单个输出项完成
            pass
        
        elif event_type == "response.completed":
            # 响应完成
            response_data = event_data.get("response", {})
            self.usage = response_data.get("usage")
        
        elif event_type == "response.done":
            # 所有响应完成 (兼容不同的事件名)
            if "usage" in event_data:
                self.usage = event_data.get("usage")
        
        return chunks
    
    def get_final_chunks(self) -> List[str]:
        """获取最终的 chunks（完成信号和使用统计）"""
        chunks = []
        
        # 转换 usage 格式: Response API -> Chat API
        chat_usage = convert_response_usage_to_chat_usage(self.usage) if self.include_usage else None
        
        # 发送完成信号
        finish_chunk = create_chat_stream_chunk(
            self.chat_id, self.model,
            {},
            finish_reason="stop",
            usage=chat_usage
        )
        chunks.append(finish_chunk)
        chunks.append("data: [DONE]\n\n")
        
        return chunks
    
    def get_accumulated_response(self) -> Dict[str, Any]:
        """获取累积的完整响应（用于非流式模式）"""
        message = {
            "role": "assistant",
            "content": self.accumulated_content
        }
        
        if self.accumulated_reasoning:
            message["reasoning_content"] = self.accumulated_reasoning
        
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls
        
        # 转换 usage 格式: Response API -> Chat API
        chat_usage = convert_response_usage_to_chat_usage(self.usage)
        
        return {
            "id": self.chat_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "stop" if not self.tool_calls else "tool_calls"
                }
            ],
            "usage": chat_usage
        }


async def parse_sse_line(line: str) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    """解析 SSE 行，返回 (event_type, event_data)"""
    if not line or line.startswith(":"):
        return None, None
    
    if line.startswith("event:"):
        return line[6:].strip(), None
    
    if line.startswith("data:"):
        data_str = line[5:].strip()
        if data_str == "[DONE]":
            return "done", None
        try:
            return None, json.loads(data_str)
        except json.JSONDecodeError:
            return None, None
    
    return None, None


def format_admin_time(value: str) -> str:
    if not value:
        return "-"
    return value.replace("T", " ").replace("+00:00", " UTC")


def build_admin_notice(message: str, level: str) -> str:
    if not message:
        return ""

    level_class = {
        "success": "notice-success",
        "error": "notice-error",
        "warning": "notice-warning",
    }.get(level, "notice-success")

    return f'<div class="notice {level_class}">{escape(message)}</div>'


def render_admin_layout(
    title: str,
    content: str,
    username: Optional[str] = None,
    notice: str = "",
    level: str = "success",
) -> str:
    nav_html = ""
    if username:
        nav_html = f"""
        <div class="topbar-actions">
            <span class="badge">管理员 {escape(username)}</span>
            <a class="ghost-link" href="/admin">控制台</a>
            <form method="post" action="/admin/logout">
                <button class="ghost-button" type="submit">退出登录</button>
            </form>
        </div>
        """

    return f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>{escape(title)}</title>
        <style>
            :root {{
                --bg: #f4efe7;
                --bg-accent: #fff8ef;
                --card: rgba(255, 252, 247, 0.86);
                --text: #1f2937;
                --muted: #5b6472;
                --line: rgba(74, 62, 46, 0.12);
                --primary: #0f766e;
                --primary-strong: #115e59;
                --danger: #b42318;
                --warning: #b45309;
                --success-bg: rgba(15, 118, 110, 0.12);
                --danger-bg: rgba(180, 35, 24, 0.1);
                --warning-bg: rgba(180, 83, 9, 0.12);
                --shadow: 0 20px 60px rgba(61, 41, 20, 0.12);
                --radius: 22px;
            }}
            * {{ box-sizing: border-box; }}
            body {{
                margin: 0;
                color: var(--text);
                font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
                background:
                    radial-gradient(circle at top left, rgba(15, 118, 110, 0.16), transparent 28%),
                    radial-gradient(circle at top right, rgba(180, 83, 9, 0.18), transparent 24%),
                    linear-gradient(180deg, var(--bg-accent), var(--bg));
                min-height: 100vh;
            }}
            a {{ color: inherit; }}
            .page {{ max-width: 1240px; margin: 0 auto; padding: 32px 20px 40px; }}
            .topbar {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 16px;
                margin-bottom: 24px;
            }}
            .brand {{
                display: flex;
                flex-direction: column;
                gap: 6px;
            }}
            .eyebrow {{
                margin: 0;
                color: var(--primary-strong);
                font-size: 13px;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }}
            h1 {{ margin: 0; font-size: clamp(28px, 4vw, 42px); line-height: 1.05; }}
            .subtitle {{ margin: 0; color: var(--muted); font-size: 15px; }}
            .topbar-actions {{
                display: flex;
                align-items: center;
                justify-content: flex-end;
                flex-wrap: wrap;
                gap: 10px;
            }}
            .badge {{
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 10px 14px;
                border-radius: 999px;
                background: rgba(15, 118, 110, 0.1);
                color: var(--primary-strong);
                font-size: 13px;
                font-weight: 700;
            }}
            .ghost-link,
            .ghost-button {{
                border: 1px solid var(--line);
                border-radius: 999px;
                padding: 10px 14px;
                background: rgba(255, 255, 255, 0.65);
                color: var(--text);
                text-decoration: none;
                cursor: pointer;
                font-size: 14px;
            }}
            .ghost-button:hover,
            .ghost-link:hover {{ border-color: rgba(15, 118, 110, 0.45); }}
            .notice {{
                margin-bottom: 18px;
                padding: 14px 16px;
                border-radius: 16px;
                border: 1px solid transparent;
                font-size: 14px;
            }}
            .notice-success {{ background: var(--success-bg); border-color: rgba(15, 118, 110, 0.24); }}
            .notice-error {{ background: var(--danger-bg); border-color: rgba(180, 35, 24, 0.22); }}
            .notice-warning {{ background: var(--warning-bg); border-color: rgba(180, 83, 9, 0.22); }}
            .grid {{ display: grid; gap: 18px; }}
            .dashboard {{ grid-template-columns: 1.2fr 0.8fr; align-items: start; }}
            .stats {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
            .card {{
                background: var(--card);
                border: 1px solid rgba(255, 255, 255, 0.8);
                box-shadow: var(--shadow);
                backdrop-filter: blur(14px);
                border-radius: var(--radius);
                padding: 22px;
            }}
            .section-title {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 12px;
                margin-bottom: 18px;
            }}
            .section-title h2,
            .section-title h3 {{ margin: 0; font-size: 20px; }}
            .muted {{ color: var(--muted); font-size: 13px; line-height: 1.5; }}
            .stat-label {{ color: var(--muted); font-size: 13px; margin-bottom: 8px; }}
            .stat-value {{ font-size: 32px; font-weight: 800; line-height: 1; }}
            .stat-footnote {{ color: var(--muted); font-size: 12px; margin-top: 10px; }}
            form {{ margin: 0; }}
            .form-grid {{ display: grid; gap: 14px; grid-template-columns: repeat(2, minmax(0, 1fr)); }}
            .form-grid.single {{ grid-template-columns: 1fr; }}
            label {{ display: flex; flex-direction: column; gap: 8px; font-size: 14px; font-weight: 600; }}
            input,
            textarea {{
                width: 100%;
                border: 1px solid var(--line);
                border-radius: 14px;
                padding: 12px 14px;
                font: inherit;
                color: var(--text);
                background: rgba(255, 255, 255, 0.92);
            }}
            textarea {{ min-height: 108px; resize: vertical; }}
            input:focus,
            textarea:focus {{ outline: 2px solid rgba(15, 118, 110, 0.18); border-color: rgba(15, 118, 110, 0.4); }}
            .button-row {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 16px; }}
            .primary-button,
            .secondary-button,
            .danger-button {{
                border: none;
                border-radius: 14px;
                padding: 12px 16px;
                font: inherit;
                font-weight: 700;
                cursor: pointer;
            }}
            .primary-button {{ background: var(--primary); color: #fff; }}
            .secondary-button {{ background: rgba(15, 118, 110, 0.1); color: var(--primary-strong); }}
            .danger-button {{ background: rgba(180, 35, 24, 0.12); color: var(--danger); }}
            .primary-button:hover {{ background: var(--primary-strong); }}
            .secondary-button:hover {{ background: rgba(15, 118, 110, 0.18); }}
            .danger-button:hover {{ background: rgba(180, 35, 24, 0.2); }}
            .stack {{ display: flex; flex-direction: column; gap: 14px; }}
            .channel-list {{ display: flex; flex-direction: column; gap: 14px; }}
            .channel-item {{
                border: 1px solid var(--line);
                border-radius: 18px;
                padding: 16px;
                background: rgba(255, 255, 255, 0.7);
            }}
            .channel-head {{
                display: flex;
                align-items: flex-start;
                justify-content: space-between;
                gap: 14px;
                margin-bottom: 12px;
            }}
            .channel-meta {{ display: grid; gap: 10px; grid-template-columns: repeat(2, minmax(0, 1fr)); margin-top: 14px; }}
            .field-label {{ color: var(--muted); font-size: 12px; margin-bottom: 6px; }}
            .field-value {{ font-size: 14px; line-height: 1.5; word-break: break-all; }}
            .pill {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                border-radius: 999px;
                padding: 8px 12px;
                font-size: 12px;
                font-weight: 700;
            }}
            .pill-enabled {{ background: rgba(15, 118, 110, 0.12); color: var(--primary-strong); }}
            .pill-disabled {{ background: rgba(180, 35, 24, 0.12); color: var(--danger); }}
            .action-row {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }}
            .action-row form {{ display: inline-flex; }}
            .code-box {{
                margin-top: 14px;
                border-radius: 16px;
                padding: 14px;
                background: #12211e;
                color: #def7ec;
                font-size: 13px;
                line-height: 1.6;
                overflow-x: auto;
            }}
            .login-shell {{ max-width: 480px; margin: 8vh auto 0; }}
            .login-shell .card {{ padding: 28px; }}
            .helper-text {{ color: var(--muted); font-size: 13px; line-height: 1.6; }}
            @media (max-width: 960px) {{
                .dashboard,
                .stats,
                .form-grid,
                .channel-meta {{ grid-template-columns: 1fr; }}
                .topbar {{ align-items: flex-start; flex-direction: column; }}
                .topbar-actions {{ justify-content: flex-start; }}
            }}
        </style>
    </head>
    <body>
        <div class="page">
            <div class="topbar">
                <div class="brand">
                    <p class="eyebrow">Response2Chat Console</p>
                    <h1>{escape(title)}</h1>
                    <p class="subtitle">多渠道路由、管理员登录和外部访问 key 统一配置。</p>
                </div>
                {nav_html}
            </div>
            {build_admin_notice(notice, level)}
            {content}
        </div>
    </body>
    </html>
    """


def render_login_page(error_message: str = "", username: str = "", next_path: str = "/admin") -> str:
    body = f"""
    <div class="login-shell">
        <div class="card stack">
            <div class="section-title">
                <div>
                    <h2>管理员登录</h2>
                    <p class="muted">使用默认管理员账号进入控制台，首次登录后建议立即修改密码。</p>
                </div>
            </div>
            <form method="post" action="/admin/login" class="stack">
                <input type="hidden" name="next" value="{escape(next_path)}" />
                <label>
                    用户名
                    <input type="text" name="username" autocomplete="username" value="{escape(username)}" placeholder="admin" required />
                </label>
                <label>
                    密码
                    <input type="password" name="password" autocomplete="current-password" placeholder="请输入管理员密码" required />
                </label>
                <button class="primary-button" type="submit">登录控制台</button>
            </form>
            <p class="helper-text">默认账号密码可通过环境变量 ADMIN_USERNAME 和 ADMIN_PASSWORD 初始化。系统只会在第一次创建数据库时写入默认管理员。</p>
        </div>
    </div>
    """
    return render_admin_layout("管理后台登录", body, notice=error_message, level="error" if error_message else "success")


def render_dashboard_page(
    request: Request,
    username: str,
    channels: List[Dict[str, Any]],
    stats: Dict[str, int],
    notice: str = "",
    level: str = "success",
) -> str:
    channel_cards = []
    external_base = str(request.base_url).rstrip("/")

    if channels:
        for channel in channels:
            toggle_label = "停用" if channel["enabled"] else "启用"
            toggle_target = "0" if channel["enabled"] else "1"
            state_class = "pill-enabled" if channel["enabled"] else "pill-disabled"
            state_text = "启用中" if channel["enabled"] else "已停用"
            description = escape(channel["description"] or "未填写描述")
            example = escape(
                json.dumps(
                    {
                        "model": "gpt-4.1",
                        "messages": [{"role": "user", "content": "你好"}],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            channel_cards.append(
                f"""
                <div class="channel-item">
                    <div class="channel-head">
                        <div>
                            <h3>{escape(channel['name'])}</h3>
                            <p class="muted">{description}</p>
                        </div>
                        <span class="pill {state_class}">{state_text}</span>
                    </div>
                    <div class="channel-meta">
                        <div>
                            <div class="field-label">上游地址</div>
                            <div class="field-value">{escape(channel['upstream_base_url'])}</div>
                        </div>
                        <div>
                            <div class="field-label">上游密钥</div>
                            <div class="field-value">{escape(mask_secret(channel['upstream_api_key']))}</div>
                        </div>
                        <div>
                            <div class="field-label">外部访问 Key</div>
                            <div class="field-value">{escape(channel['access_key'])}</div>
                        </div>
                        <div>
                            <div class="field-label">最近更新</div>
                            <div class="field-value">{escape(format_admin_time(channel['updated_at']))}</div>
                        </div>
                    </div>
                    <div class="action-row">
                        <a class="ghost-link" href="/admin/channels/{channel['id']}">编辑配置</a>
                        <form method="post" action="/admin/channels/{channel['id']}/toggle">
                            <input type="hidden" name="enabled" value="{toggle_target}" />
                            <button class="secondary-button" type="submit">{toggle_label}</button>
                        </form>
                        <form method="post" action="/admin/channels/{channel['id']}/rotate-key">
                            <button class="secondary-button" type="submit">轮换外部 Key</button>
                        </form>
                        <form method="post" action="/admin/channels/{channel['id']}/delete" onsubmit="return confirm('确认删除这个渠道吗？');">
                            <button class="danger-button" type="submit">删除</button>
                        </form>
                    </div>
                    <div class="code-box">POST {external_base}/v1/chat/completions\nAuthorization: Bearer {escape(channel['access_key'])}\nContent-Type: application/json\n\n{example}</div>
                </div>
                """
            )

    body = f"""
    <div class="grid stats">
        <div class="card">
            <div class="stat-label">渠道总数</div>
            <div class="stat-value">{stats['total']}</div>
            <div class="stat-footnote">所有已创建渠道</div>
        </div>
        <div class="card">
            <div class="stat-label">已启用</div>
            <div class="stat-value">{stats['enabled']}</div>
            <div class="stat-footnote">可供外部调用的渠道</div>
        </div>
        <div class="card">
            <div class="stat-label">已停用</div>
            <div class="stat-value">{stats['disabled']}</div>
            <div class="stat-footnote">保留配置但不接受调用</div>
        </div>
    </div>
    <div class="grid dashboard" style="margin-top: 18px;">
        <div class="card stack">
            <div class="section-title">
                <div>
                    <h2>新增渠道</h2>
                    <p class="muted">填写上游 Response API 地址和真实密钥，保存后系统会自动生成对外访问 key。</p>
                </div>
            </div>
            <form method="post" action="/admin/channels" class="stack">
                <div class="form-grid">
                    <label>
                        渠道名称
                        <input type="text" name="name" placeholder="例如：主账号 A" required />
                    </label>
                    <label>
                        上游基础 URL
                        <input type="text" name="upstream_base_url" placeholder="https://your-provider.com/v1" required />
                    </label>
                </div>
                <div class="form-grid single">
                    <label>
                        上游 API Key
                        <input type="text" name="upstream_api_key" placeholder="sk-...，可留空表示不上送 Authorization" />
                    </label>
                </div>
                <div class="form-grid single">
                    <label>
                        描述
                        <textarea name="description" placeholder="可选，记录渠道归属或用途"></textarea>
                    </label>
                </div>
                <div class="button-row">
                    <button class="primary-button" type="submit">创建渠道并生成外部 Key</button>
                </div>
            </form>
        </div>
        <div class="stack">
            <div class="card stack">
                <div class="section-title">
                    <div>
                        <h3>修改管理员密码</h3>
                        <p class="muted">默认密码只用于初始化。数据库生成后，环境变量不会覆盖已修改的管理员密码。</p>
                    </div>
                </div>
                <form method="post" action="/admin/change-password" class="stack">
                    <label>
                        当前密码
                        <input type="password" name="current_password" autocomplete="current-password" required />
                    </label>
                    <label>
                        新密码
                        <input type="password" name="new_password" autocomplete="new-password" minlength="8" required />
                    </label>
                    <label>
                        确认新密码
                        <input type="password" name="confirm_password" autocomplete="new-password" minlength="8" required />
                    </label>
                    <button class="primary-button" type="submit">更新管理员密码</button>
                </form>
            </div>
            <div class="card stack">
                <div class="section-title">
                    <div>
                        <h3>调用规则</h3>
                        <p class="muted">外部客户端使用系统生成的访问 key 调用代理，代理再自动切换到对应渠道的真实 URL 和上游密钥。</p>
                    </div>
                </div>
                <div class="helper-text">调用地址保持不变：/v1/chat/completions、/v1/responses、/v1/models。区别只在 Authorization 里放的是渠道访问 key，而不是上游服务的真实 key。</div>
            </div>
        </div>
    </div>
    <div class="card" style="margin-top: 18px;">
        <div class="section-title">
            <div>
                <h2>渠道列表</h2>
                <p class="muted">每个渠道都有独立的上游地址、真实密钥和对外访问 key。</p>
            </div>
        </div>
        <div class="channel-list">
            {''.join(channel_cards) if channel_cards else '<div class="channel-item"><p class="muted">当前还没有渠道，请先创建一个。</p></div>'}
        </div>
    </div>
    """

    return render_admin_layout("多渠道控制台", body, username=username, notice=notice, level=level)


def render_channel_detail_page(
    request: Request,
    username: str,
    channel: Dict[str, Any],
    notice: str = "",
    level: str = "success",
) -> str:
    external_base = str(request.base_url).rstrip("/")
    checked = "checked" if channel["enabled"] else ""
    body = f"""
    <div class="grid dashboard">
        <div class="card stack">
            <div class="section-title">
                <div>
                    <h2>编辑渠道</h2>
                    <p class="muted">更新上游地址、真实密钥、状态和描述。外部访问 key 可单独轮换。</p>
                </div>
                <a class="ghost-link" href="/admin">返回控制台</a>
            </div>
            <form method="post" action="/admin/channels/{channel['id']}" class="stack">
                <div class="form-grid">
                    <label>
                        渠道名称
                        <input type="text" name="name" value="{escape(channel['name'])}" required />
                    </label>
                    <label>
                        上游基础 URL
                        <input type="text" name="upstream_base_url" value="{escape(channel['upstream_base_url'])}" required />
                    </label>
                </div>
                <div class="form-grid single">
                    <label>
                        新的上游 API Key
                        <input type="text" name="upstream_api_key" placeholder="留空表示保持当前值不变" />
                    </label>
                </div>
                <div class="form-grid single">
                    <label>
                        描述
                        <textarea name="description">{escape(channel['description'])}</textarea>
                    </label>
                </div>
                <label>
                    <span>渠道状态</span>
                    <span class="helper-text">勾选表示允许外部访问 key 命中该渠道。</span>
                    <input type="checkbox" name="enabled" {checked} style="width: auto; margin-top: 6px;" />
                </label>
                <div class="button-row">
                    <button class="primary-button" type="submit">保存渠道配置</button>
                </div>
            </form>
        </div>
        <div class="stack">
            <div class="card stack">
                <div class="section-title">
                    <div>
                        <h3>当前渠道信息</h3>
                        <p class="muted">对外访问 key 和上游密钥解耦，外部只能看到访问 key。</p>
                    </div>
                </div>
                <div>
                    <div class="field-label">外部访问 Key</div>
                    <div class="field-value">{escape(channel['access_key'])}</div>
                </div>
                <div>
                    <div class="field-label">上游 API Key</div>
                    <div class="field-value">{escape(mask_secret(channel['upstream_api_key']))}</div>
                </div>
                <div>
                    <div class="field-label">创建时间</div>
                    <div class="field-value">{escape(format_admin_time(channel['created_at']))}</div>
                </div>
                <div>
                    <div class="field-label">最近更新</div>
                    <div class="field-value">{escape(format_admin_time(channel['updated_at']))}</div>
                </div>
                <div class="button-row">
                    <form method="post" action="/admin/channels/{channel['id']}/rotate-key">
                        <button class="secondary-button" type="submit">轮换外部访问 Key</button>
                    </form>
                </div>
            </div>
            <div class="card stack">
                <div class="section-title">
                    <div>
                        <h3>调用示例</h3>
                        <p class="muted">外部系统始终调用当前代理服务，Authorization 使用渠道访问 key。</p>
                    </div>
                </div>
                <div class="code-box">curl -X POST \"{external_base}/v1/chat/completions\" \\
  -H \"Authorization: Bearer {escape(channel['access_key'])}\" \\
  -H \"Content-Type: application/json\" \\
    -d '{{\n    "model": "gpt-4.1",\n    "messages": [{{"role": "user", "content": "你好"}}]\n  }}'</div>
            </div>
        </div>
    </div>
    """

    return render_admin_layout(
        f"渠道详情 - {channel['name']}",
        body,
        username=username,
        notice=notice,
        level=level,
    )


async def parse_form_body(request: Request) -> Dict[str, str]:
    raw_body = await request.body()
    parsed = parse_qs(raw_body.decode("utf-8"), keep_blank_values=True)
    return {key: values[-1] if values else "" for key, values in parsed.items()}


def get_authenticated_admin(request: Request) -> Optional[str]:
    session_manager: AdminSessionManager = request.app.state.admin_sessions
    return session_manager.get_username(request.cookies.get(ADMIN_SESSION_COOKIE_NAME))


def build_admin_redirect(path: str, message: str = "", level: str = "success") -> RedirectResponse:
    if message:
        separator = "&" if "?" in path else "?"
        path = f"{path}{separator}notice={quote(message)}&level={quote(level)}"
    return RedirectResponse(url=path, status_code=303)


def build_login_redirect(next_path: str = "/admin") -> RedirectResponse:
    return RedirectResponse(url=f"/admin/login?next={quote(next_path)}", status_code=303)


def normalize_next_path(next_path: Optional[str]) -> str:
    if next_path and next_path.startswith("/admin"):
        return next_path
    return "/admin"


async def resolve_channel_from_request(request: Request, authorization: Optional[str]) -> Dict[str, Any]:
    access_key = extract_bearer_token(authorization)
    store: SettingsStore = request.app.state.settings_store
    channel = await asyncio.to_thread(store.get_channel_by_access_key, access_key)

    if not channel:
        logger.warning("无效的渠道访问 key")
        raise HTTPException(status_code=401, detail="Invalid channel access key")

    if not channel["enabled"]:
        logger.warning(f"渠道已停用: id={channel['id']}, name={channel['name']}")
        raise HTTPException(status_code=403, detail="Channel is disabled")

    return channel


# ==================== FastAPI 应用 ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    app.state.settings_store = SettingsStore(
        database_path=DATABASE_PATH,
        default_admin_username=ADMIN_USERNAME,
        default_admin_password=ADMIN_PASSWORD,
        bootstrap_channel_url=RESPONSE_API_BASE,
        bootstrap_channel_key=RESPONSE_API_KEY,
        bootstrap_channel_name=BOOTSTRAP_CHANNEL_NAME,
    )
    await asyncio.to_thread(app.state.settings_store.initialize)
    app.state.admin_sessions = AdminSessionManager(ADMIN_SESSION_TTL_SECONDS)

    # 配置连接池限制，防止长时间运行后连接泄漏
    limits = httpx.Limits(
        max_connections=MAX_CONNECTIONS,
        max_keepalive_connections=MAX_KEEPALIVE_CONNECTIONS,
        keepalive_expiry=KEEPALIVE_EXPIRY
    )
    # 配置超时：连接超时、读取超时、写入超时、连接池获取超时
    timeout = httpx.Timeout(
        connect=30.0,      # 连接超时
        read=DEFAULT_TIMEOUT,  # 读取超时
        write=30.0,        # 写入超时  
        pool=POOL_TIMEOUT          # 从连接池获取连接的超时
    )
    app.state.http_client = httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
        http2=True  # 启用 HTTP/2 提升长连接性能
    )
    logger.info(f"配置存储初始化完成: database={DATABASE_PATH}")
    logger.info(f"HTTP 客户端初始化: max_connections={MAX_CONNECTIONS}, keepalive={MAX_KEEPALIVE_CONNECTIONS}, expiry={KEEPALIVE_EXPIRY}s")
    yield
    await app.state.http_client.aclose()
    logger.info("HTTP 客户端已关闭")

app = FastAPI(
    title="Response to Chat API Proxy",
    description="将 OpenAI Response 协议接口转发为 Chat 协议接口",
    version="1.0.0",
    lifespan=lifespan
)


PASSTHROUGH_REQUEST_EXCLUDED_HEADERS = {
    "authorization",
    "connection",
    "content-length",
    "host",
    "transfer-encoding",
}

PASSTHROUGH_RESPONSE_EXCLUDED_HEADERS = {
    "connection",
    "content-encoding",
    "content-length",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def extract_bearer_token(authorization: Optional[str]) -> str:
    """Extract bearer token from Authorization header."""
    if not authorization:
        logger.warning("Missing Authorization header")
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    return authorization.replace("Bearer ", "", 1) if authorization.startswith("Bearer ") else authorization


def build_passthrough_request_headers(request: Request, token: str) -> Dict[str, str]:
    """Copy client headers for upstream passthrough, excluding hop-by-hop headers."""
    headers: Dict[str, str] = {}
    for key, value in request.headers.items():
        if key.lower() in PASSTHROUGH_REQUEST_EXCLUDED_HEADERS:
            continue
        headers[key] = value

    if token:
        headers["Authorization"] = f"Bearer {token}"
    else:
        headers.pop("Authorization", None)
    return headers


def build_passthrough_response_headers(headers: httpx.Headers) -> Dict[str, str]:
    """Filter upstream response headers for downstream responses."""
    filtered_headers: Dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() in PASSTHROUGH_RESPONSE_EXCLUDED_HEADERS:
            continue
        filtered_headers[key] = value
    return filtered_headers


@app.get("/")
async def root_redirect():
    return RedirectResponse(url="/admin", status_code=303)


@app.get("/admin/login")
async def admin_login_page(request: Request):
    username = get_authenticated_admin(request)
    if username:
        return RedirectResponse(url="/admin", status_code=303)

    next_path = normalize_next_path(request.query_params.get("next"))
    error_message = request.query_params.get("error", "")
    return HTMLResponse(
        render_login_page(
            error_message=error_message,
            username=request.query_params.get("username", ""),
            next_path=next_path,
        )
    )


@app.post("/admin/login")
async def admin_login_submit(request: Request):
    form = await parse_form_body(request)
    username = form.get("username", "").strip()
    password = form.get("password", "")
    next_path = normalize_next_path(form.get("next"))

    store: SettingsStore = request.app.state.settings_store
    is_valid = await asyncio.to_thread(store.authenticate_admin, username, password)
    if not is_valid:
        login_page = render_login_page(
            error_message="账号或密码错误",
            username=username,
            next_path=next_path,
        )
        return HTMLResponse(login_page, status_code=401)

    session_manager: AdminSessionManager = request.app.state.admin_sessions
    session_token = session_manager.create_session(username)
    response = RedirectResponse(url=next_path, status_code=303)
    response.set_cookie(
        key=ADMIN_SESSION_COOKIE_NAME,
        value=session_token,
        max_age=ADMIN_SESSION_TTL_SECONDS,
        httponly=True,
        samesite="lax",
        secure=ADMIN_COOKIE_SECURE,
        path="/",
    )
    return response


@app.post("/admin/logout")
async def admin_logout(request: Request):
    session_manager: AdminSessionManager = request.app.state.admin_sessions
    session_manager.revoke(request.cookies.get(ADMIN_SESSION_COOKIE_NAME))
    response = RedirectResponse(url="/admin/login", status_code=303)
    response.delete_cookie(ADMIN_SESSION_COOKIE_NAME, path="/")
    return response


@app.get("/admin")
async def admin_dashboard(request: Request):
    username = get_authenticated_admin(request)
    if not username:
        return build_login_redirect("/admin")

    store: SettingsStore = request.app.state.settings_store
    channels, stats = await asyncio.gather(
        asyncio.to_thread(store.list_channels),
        asyncio.to_thread(store.count_channels),
    )
    return HTMLResponse(
        render_dashboard_page(
            request=request,
            username=username,
            channels=channels,
            stats=stats,
            notice=request.query_params.get("notice", ""),
            level=request.query_params.get("level", "success"),
        )
    )


@app.get("/admin/channels/{channel_id}")
async def admin_channel_detail(request: Request, channel_id: int):
    username = get_authenticated_admin(request)
    if not username:
        return build_login_redirect(f"/admin/channels/{channel_id}")

    store: SettingsStore = request.app.state.settings_store
    channel = await asyncio.to_thread(store.get_channel, channel_id)
    if not channel:
        return build_admin_redirect("/admin", "渠道不存在", "error")

    return HTMLResponse(
        render_channel_detail_page(
            request=request,
            username=username,
            channel=channel,
            notice=request.query_params.get("notice", ""),
            level=request.query_params.get("level", "success"),
        )
    )


@app.post("/admin/channels")
async def admin_create_channel(request: Request):
    username = get_authenticated_admin(request)
    if not username:
        return build_login_redirect("/admin")

    form = await parse_form_body(request)
    store: SettingsStore = request.app.state.settings_store

    try:
        channel = await asyncio.to_thread(
            store.create_channel,
            form.get("name", ""),
            form.get("upstream_base_url", ""),
            form.get("upstream_api_key", ""),
            form.get("description", ""),
        )
        return build_admin_redirect(
            f"/admin/channels/{channel['id']}",
            "渠道已创建，系统已自动生成外部访问 key",
            "success",
        )
    except ValueError as exc:
        return build_admin_redirect("/admin", str(exc), "error")


@app.post("/admin/channels/{channel_id}")
async def admin_update_channel(request: Request, channel_id: int):
    username = get_authenticated_admin(request)
    if not username:
        return build_login_redirect(f"/admin/channels/{channel_id}")

    form = await parse_form_body(request)
    store: SettingsStore = request.app.state.settings_store

    try:
        channel = await asyncio.to_thread(
            store.update_channel,
            channel_id,
            form.get("name", ""),
            form.get("upstream_base_url", ""),
            form.get("upstream_api_key", ""),
            form.get("description", ""),
            form.get("enabled") == "on",
        )
        if not channel:
            return build_admin_redirect("/admin", "渠道不存在", "error")
        return build_admin_redirect(f"/admin/channels/{channel_id}", "渠道配置已更新", "success")
    except ValueError as exc:
        return build_admin_redirect(f"/admin/channels/{channel_id}", str(exc), "error")


@app.post("/admin/channels/{channel_id}/toggle")
async def admin_toggle_channel(request: Request, channel_id: int):
    username = get_authenticated_admin(request)
    if not username:
        return build_login_redirect("/admin")

    form = await parse_form_body(request)
    enabled = form.get("enabled") == "1"
    store: SettingsStore = request.app.state.settings_store
    channel = await asyncio.to_thread(store.set_channel_enabled, channel_id, enabled)
    if not channel:
        return build_admin_redirect("/admin", "渠道不存在", "error")

    return build_admin_redirect(
        "/admin",
        f"渠道 {channel['name']} 已{'启用' if enabled else '停用'}",
        "success",
    )


@app.post("/admin/channels/{channel_id}/rotate-key")
async def admin_rotate_channel_key(request: Request, channel_id: int):
    username = get_authenticated_admin(request)
    if not username:
        return build_login_redirect(f"/admin/channels/{channel_id}")

    store: SettingsStore = request.app.state.settings_store
    channel = await asyncio.to_thread(store.rotate_access_key, channel_id)
    if not channel:
        return build_admin_redirect("/admin", "渠道不存在", "error")

    return build_admin_redirect(
        f"/admin/channels/{channel_id}",
        "外部访问 key 已轮换，请同步更新外部调用方配置",
        "warning",
    )


@app.post("/admin/channels/{channel_id}/delete")
async def admin_delete_channel(request: Request, channel_id: int):
    username = get_authenticated_admin(request)
    if not username:
        return build_login_redirect("/admin")

    store: SettingsStore = request.app.state.settings_store
    deleted = await asyncio.to_thread(store.delete_channel, channel_id)
    if not deleted:
        return build_admin_redirect("/admin", "渠道不存在", "error")

    return build_admin_redirect("/admin", "渠道已删除", "success")


@app.post("/admin/change-password")
async def admin_change_password(request: Request):
    username = get_authenticated_admin(request)
    if not username:
        return build_login_redirect("/admin")

    form = await parse_form_body(request)
    new_password = form.get("new_password", "")
    confirm_password = form.get("confirm_password", "")
    if new_password != confirm_password:
        return build_admin_redirect("/admin", "两次输入的新密码不一致", "error")

    store: SettingsStore = request.app.state.settings_store
    success, message = await asyncio.to_thread(
        store.change_admin_password,
        username,
        form.get("current_password", ""),
        new_password,
    )
    return build_admin_redirect("/admin", message, "success" if success else "error")


@app.post("/v1/responses")
async def responses_passthrough(
    request: Request,
    authorization: Optional[str] = Header(None)
):
    """Responses endpoint passthrough without request/response conversion."""
    channel = await resolve_channel_from_request(request, authorization)
    logger.info(f"/v1/responses 命中渠道: id={channel['id']}, name={channel['name']}")

    raw_body = await request.body()
    is_stream_request = "text/event-stream" in request.headers.get("accept", "").lower()

    if raw_body:
        try:
            request_json = json.loads(raw_body)
            is_stream_request = bool(request_json.get("stream")) or is_stream_request
            logger.info(f"Received /v1/responses request: {json.dumps(request_json, ensure_ascii=False, indent=2)}")
        except json.JSONDecodeError:
            logger.info("Received /v1/responses request: <non-json body>")
    else:
        logger.info("Received /v1/responses request: <empty body>")

    client: httpx.AsyncClient = request.app.state.http_client
    upstream_url = f"{channel['upstream_base_url']}/responses"
    upstream_headers = build_passthrough_request_headers(request, channel["upstream_api_key"])
    upstream_params = tuple(request.query_params.multi_items())
    stream_timeout = httpx.Timeout(
        connect=30.0,
        read=STREAM_READ_TIMEOUT,
        write=30.0,
        pool=POOL_TIMEOUT
    )

    logger.info(f"Passthrough /v1/responses -> {upstream_url}, stream={is_stream_request}")

    if is_stream_request:
        stream_context = client.stream(
            "POST",
            upstream_url,
            headers=upstream_headers,
            params=upstream_params,
            content=raw_body,
            timeout=stream_timeout
        )
        upstream_response = None

        try:
            upstream_response = await stream_context.__aenter__()
            logger.info(f"Upstream /responses stream status: {upstream_response.status_code}")
            response_headers = build_passthrough_response_headers(upstream_response.headers)

            async def stream_generator():
                try:
                    async for chunk in upstream_response.aiter_raw():
                        if chunk:
                            yield chunk
                except asyncio.CancelledError:
                    logger.warning("/v1/responses stream cancelled by client")
                    raise
                finally:
                    await stream_context.__aexit__(None, None, None)

            return StreamingResponse(
                stream_generator(),
                status_code=upstream_response.status_code,
                headers=response_headers
            )
        except httpx.TimeoutException:
            if upstream_response is not None:
                await upstream_response.aclose()
            logger.error("/v1/responses streaming request timed out")
            return JSONResponse(
                status_code=504,
                content={
                    "error": {
                        "message": "Request timeout",
                        "type": "timeout_error"
                    }
                }
            )
        except Exception:
            if upstream_response is not None:
                await upstream_response.aclose()
            raise

    try:
        upstream_response = await client.post(
            upstream_url,
            headers=upstream_headers,
            params=upstream_params,
            content=raw_body,
            timeout=DEFAULT_TIMEOUT
        )
        logger.info(f"Upstream /responses non-stream status: {upstream_response.status_code}")
        return Response(
            content=upstream_response.content,
            status_code=upstream_response.status_code,
            headers=build_passthrough_response_headers(upstream_response.headers)
        )
    except httpx.TimeoutException:
        logger.error("/v1/responses non-stream request timed out")
        return JSONResponse(
            status_code=504,
            content={
                "error": {
                    "message": "Request timeout",
                    "type": "timeout_error"
                }
            }
        )
    except Exception as e:
        logger.error(f"/v1/responses passthrough failed: {str(e)}\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": str(e),
                    "type": "internal_error"
                }
            }
        )


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    authorization: Optional[str] = Header(None)
):
    """Chat Completions 接口 - 转发到 Response API"""

    channel = await resolve_channel_from_request(request, authorization)
    logger.info(f"/v1/chat/completions 命中渠道: id={channel['id']}, name={channel['name']}")
    
    # 解析请求体
    try:
        body = await request.json()
        logger.info(f"收到请求: {json.dumps(body, ensure_ascii=False, indent=2)}")
        chat_request = ChatCompletionRequest(**body)
        logger.debug(f"解析后的请求: model={chat_request.model}, stream={chat_request.stream}, messages_count={len(chat_request.messages)}")
    except json.JSONDecodeError as e:
        logger.error(f"JSON 解析失败: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
    except Exception as e:
        logger.error(f"请求体解析失败: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"Invalid request body: {str(e)}")
    
    # 转换为 Response API 请求
    response_request = convert_chat_to_response_request(chat_request)
    logger.info(f"转换后的 Response API 请求: {json.dumps(response_request, ensure_ascii=False, indent=2)}")
    
    # 生成 Chat ID
    chat_id = generate_chat_id()
    logger.debug(f"生成 Chat ID: {chat_id}")
    
    # 获取 HTTP 客户端
    client: httpx.AsyncClient = request.app.state.http_client
    
    # 准备请求头
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }
    if channel["upstream_api_key"]:
        headers["Authorization"] = f"Bearer {channel['upstream_api_key']}"
    
    # Response API URL
    response_url = f"{channel['upstream_base_url']}/responses"
    logger.info(f"转发到: {response_url}")
    
    if chat_request.stream:
        # 流式模式：直接转发 SSE
        logger.info("使用流式模式处理请求")
        return await handle_stream_response(
            client, response_url, headers, response_request,
            chat_id, chat_request.model,
            bool(chat_request.stream_options.include_usage) if chat_request.stream_options else False
        )
    else:
        # 非流式模式：收集完整响应后返回
        logger.info("使用非流式模式处理请求")
        return await handle_non_stream_response(
            client, response_url, headers, response_request,
            chat_id, chat_request.model
        )


async def handle_stream_response(
    client: httpx.AsyncClient,
    url: str,
    headers: Dict[str, str],
    request_body: Dict[str, Any],
    chat_id: str,
    model: str,
    include_usage: bool
) -> StreamingResponse:
    """处理流式响应"""
    
    async def stream_generator():
        processor = ResponseStreamProcessor(chat_id, model, include_usage)
        current_event_type = None
        response = None
        start_time = time.monotonic()
        
        try:
            logger.debug(f"开始流式请求到 {url}")
            async with client.stream(
                "POST",
                url,
                headers=headers,
                json=request_body,
                timeout=httpx.Timeout(
                    connect=30.0,
                    read=STREAM_READ_TIMEOUT,
                    write=30.0,
                    pool=POOL_TIMEOUT
                )
            ) as response:
                logger.info(f"上游响应状态码: {response.status_code}")
                logger.debug(f"上游响应头: {dict(response.headers)}")
                
                if response.status_code != 200:
                    error_body = await response.aread()
                    error_msg = error_body.decode("utf-8", errors="ignore")
                    logger.error(f"上游错误响应: {error_msg}")
                    
                    # 检查是否为需要返回 500 状态码的错误（让网关触发自动禁用）
                    # 包括：账户池无可用(503)、配额不足(402)
                    should_return_500 = False
                    error_output: Dict[str, Any]
                    try:
                        error_json = json.loads(error_msg)
                        error_code = error_json.get("error", {}).get("code")
                        error_message = error_json.get("error", {}).get("message", "")
                        if error_code == 503 or \
                           error_code == "plan_quota_exceeded" or \
                           "账户池都无可用" in error_message or \
                           response.status_code == 402:
                            should_return_500 = True
                        # 直接使用上游的错误响应
                        error_output = error_json
                    except:
                        if "账户池都无可用" in error_msg:
                            should_return_500 = True
                        # JSON 解析失败，包装成标准格式
                        error_output = {
                            "error": {
                                "message": error_msg,
                                "type": "upstream_error",
                                "code": str(response.status_code)
                            }
                        }
                    
                    # 如果上游返回 402，也需要返回 500
                    if response.status_code == 402:
                        should_return_500 = True
                    
                    # 如果需要返回 500，在错误信息中添加标记
                    if should_return_500 and "error" in error_output:
                        error_output["error"]["upstream_status_code"] = response.status_code
                        error_output["error"]["gateway_status_code"] = 500
                    
                    yield f"data: {json.dumps(error_output, ensure_ascii=False)}\n\n"
                    return
                
                async for line in response.aiter_lines():
                    if STREAM_MAX_DURATION > 0 and (time.monotonic() - start_time) > STREAM_MAX_DURATION:
                        logger.error(f"流式请求超过最大持续时间: {STREAM_MAX_DURATION}s, chat_id={chat_id}")
                        error_chunk = {
                            "error": {
                                "message": "Stream max duration exceeded",
                                "type": "timeout_error"
                            }
                        }
                        yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
                        return

                    line = line.strip()
                    if not line:
                        continue
                    
                    logger.debug(f"收到上游数据行: {line[:200]}..." if len(line) > 200 else f"收到上游数据行: {line}")
                    
                    if line.startswith("event:"):
                        current_event_type = line[6:].strip()
                        logger.debug(f"事件类型: {current_event_type}")
                    elif line.startswith("data:"):
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            final_response = processor.get_accumulated_response()
                            logger.info(f"流式响应完成: {json.dumps(final_response, ensure_ascii=False)}")
                            # 发送最终 chunks
                            for chunk in processor.get_final_chunks():
                                yield chunk
                            return
                        
                        try:
                            event_data = json.loads(data_str)
                            logger.debug(f"解析事件数据: type={event_data.get('type', current_event_type)}")
                            
                            # 检查是否为上游错误响应（如账户池无可用、配额不足）
                            if "error" in event_data:
                                error_info = event_data.get("error", {})
                                error_code = error_info.get("code")
                                error_message = error_info.get("message", "")
                                logger.error(f"上游错误响应: {json.dumps(event_data, ensure_ascii=False)}")
                                
                                # 检查是否为需要返回 500 的错误（让网关触发自动禁用）
                                # 包括：账户池无可用(503)、配额不足(plan_quota_exceeded)
                                should_return_500 = (error_code == 503 or 
                                                     error_code == "503" or 
                                                     error_code == "plan_quota_exceeded" or
                                                     "账户池都无可用" in error_message or
                                                     "quota" in error_message.lower())
                                
                                # 直接透传上游的错误响应，添加状态码标记
                                if should_return_500:
                                    event_data["error"]["gateway_status_code"] = 500
                                
                                yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
                                return
                            
                            # 处理事件
                            if current_event_type:
                                chunks = processor.process_event(current_event_type, event_data)
                                for chunk in chunks:
                                    logger.debug(f"发送 chunk: {chunk[:100]}..." if len(chunk) > 100 else f"发送 chunk: {chunk}")
                                    yield chunk
                            # 也尝试从 data 中获取 type
                            elif "type" in event_data:
                                chunks = processor.process_event(event_data["type"], event_data)
                                for chunk in chunks:
                                    logger.debug(f"发送 chunk: {chunk[:100]}..." if len(chunk) > 100 else f"发送 chunk: {chunk}")
                                    yield chunk
                        except json.JSONDecodeError as e:
                            logger.warning(f"JSON 解析失败: {e}, 原始数据: {data_str[:100]}")
                            continue
                
                # 如果没有收到 [DONE]，手动发送结束
                final_response = processor.get_accumulated_response()
                logger.info(f"流结束: {json.dumps(final_response, ensure_ascii=False)}")
                for chunk in processor.get_final_chunks():
                    yield chunk
                    
        except httpx.TimeoutException:
            logger.error("请求超时")
            error_chunk = {
                "error": {
                    "message": "Request timeout",
                    "type": "timeout_error"
                }
            }
            yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
        except httpx.RemoteProtocolError as e:
            logger.error(f"远程协议错误(可能是连接被重置): {str(e)}")
            error_chunk = {
                "error": {
                    "message": f"Connection reset: {str(e)}",
                    "type": "connection_error"
                }
            }
            yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
        except httpx.ReadError as e:
            logger.error(f"读取错误: {str(e)}")
            error_chunk = {
                "error": {
                    "message": f"Read error: {str(e)}",
                    "type": "connection_error"
                }
            }
            yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
        except asyncio.CancelledError:
            logger.warning(f"流式请求被取消 (客户端可能断开连接): chat_id={chat_id}")
            # 不需要 yield 错误，客户端已断开
            return
        except GeneratorExit:
            logger.warning(f"生成器退出 (客户端断开): chat_id={chat_id}")
            return
        except Exception as e:
            logger.error(f"流式处理异常: {str(e)}\n{traceback.format_exc()}")
            error_chunk = {
                "error": {
                    "message": str(e),
                    "type": "internal_error"
                }
            }
            yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
        finally:
            logger.debug(f"流式生成器结束: chat_id={chat_id}")
    
    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


async def handle_non_stream_response(
    client: httpx.AsyncClient,
    url: str,
    headers: Dict[str, str],
    request_body: Dict[str, Any],
    chat_id: str,
    model: str
) -> JSONResponse:
    """处理非流式响应 - 收集完整的流式响应后返回"""
    
    processor = ResponseStreamProcessor(chat_id, model, include_usage=True)
    current_event_type = None
    
    try:
        logger.debug(f"开始非流式请求到 {url}")
        async with client.stream(
            "POST",
            url,
            headers=headers,
            json=request_body,
            timeout=DEFAULT_TIMEOUT
        ) as response:
            logger.info(f"上游响应状态码: {response.status_code}")
            logger.debug(f"上游响应头: {dict(response.headers)}")
            
            if response.status_code != 200:
                error_body = await response.aread()
                error_text = error_body.decode("utf-8", errors="ignore")
                logger.error(f"上游错误响应: {error_text}")
                
                # 检查是否为需要返回 500 状态码的错误（让网关触发自动禁用）
                # 包括：账户池无可用(503)、配额不足(402)
                should_return_500 = False
                error_output: Dict[str, Any]
                try:
                    error_data = json.loads(error_text)
                    error_code = error_data.get("error", {}).get("code")
                    error_message = error_data.get("error", {}).get("message", "")
                    if error_code == 503 or \
                       error_code == "plan_quota_exceeded" or \
                       "账户池都无可用" in error_message or \
                       response.status_code == 402:
                        should_return_500 = True
                    # 直接使用上游的错误响应
                    error_output = error_data
                except:
                    if "账户池都无可用" in error_text:
                        should_return_500 = True
                    # JSON 解析失败，包装成标准格式
                    error_output = {
                        "error": {
                            "message": error_text,
                            "type": "upstream_error",
                            "code": str(response.status_code)
                        }
                    }
                
                # 如果上游返回 402，也需要返回 500
                if response.status_code == 402:
                    should_return_500 = True
                
                # 添加状态码标记信息
                if should_return_500 and "error" in error_output:
                    error_output["error"]["upstream_status_code"] = response.status_code
                    error_output["error"]["gateway_status_code"] = 500
                
                return JSONResponse(
                    status_code=500 if should_return_500 else response.status_code,
                    content=error_output
                )
            
            async for line in response.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                
                logger.debug(f"收到上游数据行: {line[:200]}..." if len(line) > 200 else f"收到上游数据行: {line}")
                
                if line.startswith("event:"):
                    current_event_type = line[6:].strip()
                    logger.debug(f"事件类型: {current_event_type}")
                elif line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        logger.info("收到 [DONE] 信号")
                        break
                    
                    try:
                        event_data = json.loads(data_str)
                        logger.debug(f"解析事件数据: type={event_data.get('type', current_event_type)}")
                        
                        # 检查是否为上游错误响应（如账户池无可用、配额不足）
                        if "error" in event_data:
                            error_info = event_data.get("error", {})
                            error_code = error_info.get("code")
                            error_message = error_info.get("message", "")
                            logger.error(f"上游错误响应: {json.dumps(event_data, ensure_ascii=False)}")
                            
                            # 检查是否为需要返回 500 的错误（让网关触发自动禁用）
                            # 包括：账户池无可用(503)、配额不足(plan_quota_exceeded)
                            should_return_500 = (error_code == 503 or 
                                                 error_code == "503" or 
                                                 error_code == "plan_quota_exceeded" or
                                                 "账户池都无可用" in error_message or
                                                 "quota" in error_message.lower())
                            
                            # 直接透传上游的错误响应，添加状态码标记
                            if should_return_500:
                                event_data["error"]["gateway_status_code"] = 500
                            
                            return JSONResponse(
                                status_code=500 if should_return_500 else 502,
                                content=event_data
                            )
                        
                        if current_event_type:
                            processor.process_event(current_event_type, event_data)
                        elif "type" in event_data:
                            processor.process_event(event_data["type"], event_data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON 解析失败: {e}, 原始数据: {data_str[:100]}")
                        continue
        
        # 返回累积的完整响应
        result = processor.get_accumulated_response()
        logger.info(f"返回完整响应: {json.dumps(result, ensure_ascii=False)}")
        return JSONResponse(content=result)
        
    except httpx.TimeoutException:
        logger.error("请求超时")
        return JSONResponse(
            status_code=504,
            content={
                "error": {
                    "message": "Request timeout",
                    "type": "timeout_error"
                }
            }
        )
    except Exception as e:
        logger.error(f"非流式处理异常: {str(e)}\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": str(e),
                    "type": "internal_error"
                }
            }
        )


@app.get("/health")
async def health_check(request: Request):
    """健康检查接口 - 包含连接池状态"""
    client: httpx.AsyncClient = request.app.state.http_client
    store: SettingsStore = request.app.state.settings_store
    
    # 获取连接池统计信息
    pool_status = {}
    try:
        # httpx 的连接池信息
        if hasattr(client, '_transport') and client._transport:
            transport = client._transport
            pool = getattr(transport, '_pool', None)
            if pool is not None:
                pool_connections = getattr(pool, '_connections', None)
                pool_status = {
                    "connections_in_pool": len(pool_connections) if pool_connections is not None else "unknown"
                }
    except Exception as e:
        pool_status = {"error": str(e)}

    channel_stats = await asyncio.to_thread(store.count_channels)
    
    return {
        "status": "ok", 
        "service": "response-to-chat-proxy",
        "pool_status": pool_status,
        "database_path": DATABASE_PATH,
        "channels": channel_stats,
        "config": {
            "max_connections": MAX_CONNECTIONS,
            "max_keepalive_connections": MAX_KEEPALIVE_CONNECTIONS,
            "keepalive_expiry": KEEPALIVE_EXPIRY,
            "default_timeout": DEFAULT_TIMEOUT,
            "pool_timeout": POOL_TIMEOUT,
            "stream_read_timeout": STREAM_READ_TIMEOUT,
            "stream_max_duration": STREAM_MAX_DURATION,
            "bootstrap_channel_configured": bool(RESPONSE_API_BASE)
        }
    }


@app.get("/v1/models")
async def list_models(
    request: Request,
    authorization: Optional[str] = Header(None)
):
    """模型列表接口 - 透传到上游"""
    channel = await resolve_channel_from_request(request, authorization)
    client: httpx.AsyncClient = request.app.state.http_client
    headers: Dict[str, str] = {}
    if channel["upstream_api_key"]:
        headers["Authorization"] = f"Bearer {channel['upstream_api_key']}"
    
    try:
        response = await client.get(
            f"{channel['upstream_base_url']}/models",
            headers=headers
        )
        return JSONResponse(
            status_code=response.status_code,
            content=response.json()
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e)}}
        )


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"Starting Response to Chat API Proxy on {host}:{port}")
    print(f"Upstream API: {RESPONSE_API_BASE}")
    
    uvicorn.run(app, host=host, port=port)
