"""HTTP server exposing an OpenAI-compatible Chat Completions endpoint.

Backend is MC-LLM (our model) — nothing is proxied to an external API.

    POST /v1/chat/completions
    GET  /v1/models

Run:

    python -m inference.server --checkpoint checkpoints --port 8080

Uses only the Python standard library, so no web-framework dependency is
required.
"""
from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, List

import torch

from inference.generate import generate
from inference.loader import load_model

# ---------------------------------------------------------------------------
# chat template
# ---------------------------------------------------------------------------
ROLE_TO_TOKEN = {
    "system": "<SYSTEM>",
    "user": "<USER>",
    "assistant": "<ASSISTANT>",
    "tool": "<TOOL>",
}


def format_messages(tokenizer, messages: List[Dict[str, str]]) -> str:
    """Convert OpenAI-style messages into the MC-LLM chat format."""
    parts = []
    for m in messages:
        role = m.get("role", "user")
        token = ROLE_TO_TOKEN.get(role, "<USER>")
        content = m.get("content", "")
        parts.append(f"{token}\n{content}\n")
    parts.append("<ASSISTANT>\n")
    return "".join(parts)


class ChatServer:
    def __init__(self, model, tokenizer, meta):
        self.model = model
        self.tokenizer = tokenizer
        self.meta = meta

    def handle_chat(self, payload: Dict) -> Dict:
        messages = payload.get("messages", [])
        max_tokens = payload.get("max_tokens", 128)
        temperature = payload.get("temperature", 0.8)
        top_p = payload.get("top_p", 1.0)
        top_k = payload.get("top_k", 0)
        repetition_penalty = payload.get("repetition_penalty", 1.0)
        stop = payload.get("stop", None)

        prompt_ids = self.tokenizer.tokenize_chat(messages, add_generation_prompt=True)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long,
                                 device=next(self.model.parameters()).device)

        stop_ids = [self.tokenizer.eos_token_id]
        if stop:
            stop_ids += [self.tokenizer.encode(s)[0] for s in (stop if isinstance(stop, list) else [stop])]

        output_ids = generate(
            self.model, input_ids, max_new_tokens=max_tokens,
            eos_token_id=self.tokenizer.eos_token_id, temperature=temperature,
            top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty,
            stop_token_ids=stop_ids,
        )
        reply_ids = output_ids[0].tolist()[len(prompt_ids):]
        reply = self.tokenizer.decode(reply_ids, skip_special_tokens=True).strip()

        return {
            "id": f"chatcmpl-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "mega-cyber-llm",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": reply},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": len(prompt_ids),
                      "completion_tokens": len(reply_ids),
                      "total_tokens": len(output_ids[0].tolist())},
        }


def make_handler(server: ChatServer):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, obj, status=200):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/") == "/v1/models":
                self._send_json({"object": "list", "data": [{
                    "id": "mega-cyber-llm", "object": "model",
                    "parameters": server.meta.get("config", {}).get("model", {})}]})
            else:
                self._send_json({"error": "not found"}, 404)

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/chat/completions":
                self._send_json({"error": "not found"}, 404)
                return
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                self._send_json({"error": "invalid JSON"}, 400)
                return
            try:
                result = server.handle_chat(payload)
                self._send_json(result)
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": str(e)}, 500)

        def log_message(self, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description="MC-LLM chat server")
    parser.add_argument("--checkpoint", default="checkpoints")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer_config.json")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    model, tokenizer, meta = load_model(args.checkpoint, args.tokenizer)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MC-LLM loaded: {n_params:,} parameters")
    print(f"Listening on http://{args.host}:{args.port}")

    server = ChatServer(model, tokenizer, meta)
    httpd = HTTPServer((args.host, args.port), make_handler(server))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
