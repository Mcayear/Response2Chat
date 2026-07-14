import json
import unittest

from main import ResponseStreamProcessor


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


if __name__ == "__main__":
    unittest.main()
