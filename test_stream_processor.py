import json
import unittest

from main import (
    ChatCompletionRequest,
    ResponseStreamProcessor,
    convert_chat_to_response_request,
    convert_response_usage_to_chat_usage,
)


def parse_chunk(chunk):
    return json.loads(chunk.removeprefix("data: ").strip())


class ResponseStreamProcessorTests(unittest.TestCase):
    def test_streams_complete_metadata_for_multiple_tool_calls(self):
        processor = ResponseStreamProcessor("chatcmpl-test", "test-model")

        first_start = processor.process_event(
            "response.output_item.added",
            {
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "call_id": "call_search",
                    "name": "search_web",
                },
            },
        )
        first_delta = processor.process_event(
            "response.function_call_arguments.delta",
            {"output_index": 0, "delta": '{"query":"梁静茹 情歌"}'},
        )
        processor.process_event("response.function_call_arguments.done", {"output_index": 0})

        second_start = processor.process_event(
            "response.output_item.added",
            {
                "output_index": 1,
                "item": {
                    "type": "function_call",
                    "call_id": "call_fetch",
                    "name": "fetch_url",
                },
            },
        )
        second_delta = processor.process_event(
            "response.function_call_arguments.delta",
            {"output_index": 1, "delta": '{"url":"https://example.com"}'},
        )
        processor.process_event("response.function_call_arguments.done", {"output_index": 1})

        first_start_call = parse_chunk(first_start[0])["choices"][0]["delta"]["tool_calls"][0]
        first_delta_call = parse_chunk(first_delta[0])["choices"][0]["delta"]["tool_calls"][0]
        second_start_call = parse_chunk(second_start[0])["choices"][0]["delta"]["tool_calls"][0]
        second_delta_call = parse_chunk(second_delta[0])["choices"][0]["delta"]["tool_calls"][0]
        finish_choice = parse_chunk(processor.get_final_chunks()[0])["choices"][0]

        self.assertEqual(
            first_start_call,
            {
                "index": 0,
                "id": "call_search",
                "type": "function",
                "function": {"name": "search_web", "arguments": ""},
            },
        )
        self.assertEqual(first_delta_call["index"], 0)
        self.assertEqual(second_start_call["index"], 1)
        self.assertEqual(second_start_call["id"], "call_fetch")
        self.assertEqual(second_start_call["function"]["name"], "fetch_url")
        self.assertEqual(second_delta_call["index"], 1)
        self.assertEqual(finish_choice["finish_reason"], "tool_calls")

    def test_stream_without_tools_finishes_with_stop(self):
        processor = ResponseStreamProcessor("chatcmpl-test", "test-model")

        finish_choice = parse_chunk(processor.get_final_chunks()[0])["choices"][0]

        self.assertEqual(finish_choice["finish_reason"], "stop")


class PromptCacheTests(unittest.TestCase):
    def test_generated_cache_key_is_stable_as_conversation_grows(self):
        base_request = ChatCompletionRequest(
            model="gpt-5.6-sol",
            messages=[
                {"role": "developer", "content": "Use tools when needed."},
                {"role": "user", "content": "梁静茹《情歌》的资料"},
            ],
            tools=[{
                "type": "function",
                "function": {"name": "search_web", "parameters": {"type": "object"}},
            }],
        )
        grown_request = ChatCompletionRequest(
            model="gpt-5.6-sol",
            messages=[
                {"role": "developer", "content": "Use tools when needed."},
                {"role": "user", "content": "梁静茹《情歌》的资料"},
                {"role": "assistant", "content": "我先查询。"},
                {"role": "tool", "tool_call_id": "call_search", "content": "result"},
            ],
            tools=base_request.tools,
        )

        base_key = convert_chat_to_response_request(base_request)["prompt_cache_key"]
        grown_key = convert_chat_to_response_request(grown_request)["prompt_cache_key"]

        self.assertEqual(base_key, grown_key)
        self.assertTrue(base_key.startswith("r2c-"))
        self.assertEqual(len(base_key), 52)

    def test_client_cache_configuration_is_forwarded(self):
        request = ChatCompletionRequest(
            model="gpt-5.6-sol",
            messages=[{"role": "user", "content": "hello"}],
            prompt_cache_key="client-cache-key",
            prompt_cache_options={"mode": "implicit", "ttl": "30m"},
        )

        converted = convert_chat_to_response_request(request)

        self.assertEqual(converted["prompt_cache_key"], "client-cache-key")
        self.assertEqual(converted["prompt_cache_options"], {"mode": "implicit", "ttl": "30m"})

    def test_cache_write_tokens_are_preserved(self):
        usage = convert_response_usage_to_chat_usage({
            "input_tokens": 9000,
            "output_tokens": 100,
            "total_tokens": 9100,
            "input_tokens_details": {
                "cached_tokens": 7680,
                "cache_write_tokens": 1024,
            },
        })

        self.assertEqual(usage["prompt_tokens_details"]["cached_tokens"], 7680)
        self.assertEqual(usage["prompt_tokens_details"]["cache_write_tokens"], 1024)


if __name__ == "__main__":
    unittest.main()
