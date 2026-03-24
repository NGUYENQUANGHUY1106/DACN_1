from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model và tokenizer
model_id = "VietAI/gpt-neo-1.3B-vietnamese-news"

tokenizer = AutoTokenizer.from_pretrained(model_id)

# CPU không hỗ trợ float16 → dùng float32
# device_map="auto" gây lỗi 'disk' khi OFFLOAD → tắt
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float32
).to("cpu")

# Hàm chat
def chat_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.replace(prompt, "").strip()
