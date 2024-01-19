from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import streamlit as st
from openxlab.model import download
from modelscope import snapshot_download
import os

level = os.getenv('level')

with st.sidebar:
    st.markdown("## 书生·浦语 2.0 全新体验！")
    "欢迎使用 [InternLM2](https://github.com/InternLM/InternLM.git)"
    max_length = st.slider("max_length", 0, 1024, 512, step=1)
    system_prompt = st.text_input("System_Prompt", "")

st.title("InternLM2-Chat-"+ str(level) +"B")
st.caption("🚀 Powered By Shanghai Ai Lab")

# 定义模型路径
## ModelScope
model_id = 'Shanghai_AI_Laboratory/internlm2-chat-'+ str(level) +'b'
mode_name_or_path = snapshot_download(model_id, revision='master')

# OpenXLab
# model_repo = "OpenLMLab/internlm2-chat-7b"
# mode_name_or_path = download(model_repo=model_repo)


@st.cache_resource
def get_model():
    tokenizer = AutoTokenizer.from_pretrained(mode_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(mode_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
    model.eval()  
    return tokenizer, model

tokenizer, model = get_model()
if "messages" not in st.session_state:
    st.session_state["messages"] = []
for msg in st.session_state.messages:
    st.chat_message("user").write(msg[0])
    st.chat_message("assistant").write(msg[1])
if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    response, history = model.chat(tokenizer, prompt, meta_instruction=system_prompt, history=st.session_state.messages)
    st.session_state.messages.append((prompt, response))
    st.chat_message("assistant").write(response)