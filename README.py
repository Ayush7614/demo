import os
import time
from typing import Optional, List, Dict, Any
from openai import OpenAI
import streamlit as st

st.set_page_config(page_title="GPT-5 vs Grok-4 — Compare", layout="wide")

def make_client(provider: str, api_key: str) -> OpenAI:
    if provider == "openrouter":
        return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    elif provider == "xai":
        return OpenAI(base_url="https://api.x.ai/v1", api_key=api_key)
    elif provider == "openai":
        return OpenAI(api_key=api_key)
    else:
        raise ValueError("Unknown provider: " + provider)

def build_messages(prompt: str, image_url: Optional[str]) -> List[Dict[str, Any]]:
    if image_url:
        return [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }]
    return [{"role": "user", "content": prompt}]

def call_model(provider: str, api_key: str, model_name: str, prompt: str, image_url: Optional[str], stream: bool = True):
    client = make_client(provider, api_key)
    messages = build_messages(prompt, image_url)
    t0 = time.perf_counter()
    if stream:
        chunks = client.chat.completions.create(model=model_name, messages=messages, stream=True)
        collected_text = ""
        for chunk in chunks:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                collected_text += delta.content
                yield ("stream", delta.content)
        t1 = time.perf_counter()
        yield ("done", {"latency_s": t1 - t0, "full_text": collected_text})
    else:
        out = client.chat.completions.create(model=model_name, messages=messages, stream=False)
        t1 = time.perf_counter()
        text = out.choices[0].message.content
        yield ("full", {"latency_s": t1 - t0, "full_text": text})

st.title("⚡ Compare: OpenAI GPT-5 vs xAI Grok-4")
st.caption("Text or image+text. See live output + latency.")

with st.sidebar:
    st.header("Keys & Provider")
    mode = st.radio("How to call models?", ["OpenRouter (one key)", "Native (OpenAI + xAI)"], index=0)

    if mode == "OpenRouter (one key)":
        OPENROUTER_API_KEY = st.text_input("OPENROUTER_API_KEY", type="password", value=os.getenv("OPENROUTER_API_KEY",""))
        provider = "openrouter"
        gpt5_model = "openai/gpt-5"
        grok4_model = "x-ai/grok-4"
    else:
        OPENAI_API_KEY = st.text_input("OPENAI_API_KEY", type="password", value=os.getenv("OPENAI_API_KEY",""))
        XAI_API_KEY = st.text_input("XAI_API_KEY", type="password", value=os.getenv("XAI_API_KEY",""))
        gpt5_model = "gpt-5"
        grok4_model = "grok-4"

st.subheader("Prompt")
prompt = st.text_area("Enter your prompt", height=140, placeholder="Explain attention in 3 plain bullets.")
image_url = st.text_input("Optional image URL", placeholder="https://example.com/image.jpg")

c1, c2, c3 = st.columns(3)
with c1: run_gpt5 = st.button("Run GPT-5", use_container_width=True)
with c2: run_grok4 = st.button("Run Grok-4", use_container_width=True)
with c3: run_both = st.button("Compare Both", use_container_width=True)

def have_keys() -> bool:
    if mode == "OpenRouter (one key)":
        return bool(OPENROUTER_API_KEY.strip())
    else:
        return bool(OPENAI_API_KEY.strip()) and bool(XAI_API_KEY.strip())

def render_block(title: str, events, container):
    with container.container():
        st.markdown(f"### {title}")
        out_area = st.empty()
        meta_area = st.empty()
        collected = ""
        for kind, payload in events:
            if kind == "stream":
                collected += payload
                out_area.markdown(collected)
            elif kind in ("done","full"):
                meta_area.info(f"Latency: {payload['latency_s']:.2f}s  •  Characters: {len(payload['full_text'])}")
                out_area.markdown(payload["full_text"])

if run_gpt5 or run_grok4 or run_both:
    if not prompt.strip():
        st.error("Please enter a prompt.")
    elif not have_keys():
        st.error("Please provide the required API key(s) in the sidebar.")
    else:
        if run_gpt5 and not run_both:
            events = call_model("openrouter", OPENROUTER_API_KEY, gpt5_model, prompt, image_url, True) if mode.startswith("OpenRouter") \
                else call_model("openai", OPENAI_API_KEY, gpt5_model, prompt, image_url, True)
            render_block("OpenAI GPT-5", events, st)

        elif run_grok4 and not run_both:
            events = call_model("openrouter", OPENROUTER_API_KEY, grok4_model, prompt, image_url, True) if mode.startswith("OpenRouter") \
                else call_model("xai", XAI_API_KEY, grok4_model, prompt, image_url, True)
            render_block("xAI Grok-4", events, st)

        else:
            colL, colR = st.columns(2)
            if mode.startswith("OpenRouter"):
                ev1 = call_model("openrouter", OPENROUTER_API_KEY, gpt5_model, prompt, image_url, True)
                ev2 = call_model("openrouter", OPENROUTER_API_KEY, grok4_model, prompt, image_url, True)
            else:
                ev1 = call_model("openai", OPENAI_API_KEY, gpt5_model, prompt, image_url, True)
                ev2 = call_model("xai", XAI_API_KEY, grok4_model, prompt, image_url, True)
            with colL: render_block("OpenAI GPT-5", ev1, st)
            with colR: render_block("xAI Grok-4", ev2, st)

st.markdown("---")
st.caption("Model ids: OpenRouter → `openai/gpt-5`, `x-ai/grok-4` • Native → `gpt-5`, `grok-4`")
