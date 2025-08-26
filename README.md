#!/usr/bin/env python3
import os, sys, time, argparse
from typing import Optional
from openai import OpenAI

def make_client(provider: str, key: str) -> OpenAI:
    if provider == "openrouter":
        return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
    elif provider == "openai":
        return OpenAI(api_key=key)
    elif provider == "xai":
        return OpenAI(base_url="https://api.x.ai/v1", api_key=key)
    else:
        raise ValueError("Unknown provider")

def build_messages(prompt: str, image_url: Optional[str] = None):
    if image_url:
        return [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }]
    return [{"role": "user", "content": prompt}]

def run_once(provider: str, model: str, prompt: str, image_url: Optional[str], stream: bool, key: Optional[str]):
    # Key fallback by provider
    if not key:
        if provider == "openrouter":
            key = os.getenv("OPENROUTER_API_KEY", "")
        elif provider == "openai":
            key = os.getenv("OPENAI_API_KEY", "")
        elif provider == "xai":
            key = os.getenv("XAI_API_KEY", "")
    if not key:
        print("Missing API key. Set OPENROUTER_API_KEY / OPENAI_API_KEY / XAI_API_KEY or pass --key.", file=sys.stderr)
        sys.exit(2)

    client = make_client(provider, key)
    messages = build_messages(prompt, image_url)

    t0 = time.perf_counter()
    if stream:
        chunks = client.chat.completions.create(model=model, messages=messages, stream=True)
        out = ""
        for ch in chunks:
            delta = ch.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                out += delta.content
                print(delta.content, end="", flush=True)
        print()
        latency = time.perf_counter() - t0
        return out, latency
    else:
        resp = client.chat.completions.create(model=model, messages=messages, stream=False)
        text = resp.choices[0].message.content
        latency = time.perf_counter() - t0
        return text, latency

def main():
    ap = argparse.ArgumentParser(description="Query GPT-5 or Grok-4 and print latency.")
    ap.add_argument("--provider", choices=["openrouter", "openai", "xai"], required=True, help="API provider")
    ap.add_argument("--model", required=True, help="Model id (e.g., openai/gpt-5, x-ai/grok-4, gpt-5, grok-4)")
    ap.add_argument("--prompt", required=True, help="User prompt")
    ap.add_argument("--image-url", default=None, help="Optional image URL")
    ap.add_argument("--stream", action="store_true", help="Stream tokens to terminal")
    ap.add_argument("--key", default=None, help="Explicit API key (overrides env)")
    args = ap.parse_args()

    print(f"==> Provider: {args.provider} | Model: {args.model}")
    text, latency = run_once(args.provider, args.model, args.prompt, args.image_url, args.stream, args.key)
    print("\n---")
    print(f"Latency: {latency:.2f}s")
    print("\nResponse:\n", text)

if __name__ == "__main__":
    main()
