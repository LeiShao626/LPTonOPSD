import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_config = {
    "model_dir": "/data/.rootcache/modelscope/models/Efficient-Large-Model--Fast_dLLM_1.5B/snapshots/master",
    "max_new_tokens": 256,
    "small_block_size": 32,
    "threshold": 0.9,
}


class Model:
    def __init__(self, model_config):
        self.model_dir = model_config["model_dir"]
        self.max_new_tokens = model_config["max_new_tokens"]
        self.small_block_size = model_config["small_block_size"]
        self.threshold = model_config["threshold"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        ).eval()

    def generate(self, question):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            output_ids = self.model.generate(
                inputs["input_ids"],
                tokenizer=self.tokenizer,
                max_new_tokens=self.max_new_tokens,
                small_block_size=self.small_block_size,
                threshold=self.threshold,
            )
        return self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

