
from qwen_vl_utils import process_vision_info
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

class QwenModel3():
    def __init__(self, model_path):
        self.processor = AutoProcessor.from_pretrained(model_path, fix_mistral_regex=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype="auto",
            device_map="auto"
        )

    def qwen_data_pack(self, images, user_content):
        content = []
        resized_height = 400
        resized_width = 450
        for image in images:
            cur_json = {
                "type": "image",
                "image": image,
                "resized_height": resized_height,
                "resized_width": resized_width,
            }
            content.append(cur_json)
        content.append({
            "type": "text",
            "text": user_content,
        })
        messages = [
            {
                "role": "user",
                "content": content
            },
        ]
        return messages, resized_height, resized_width

    def qwen_infer(self, messages):
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        placeholder = '<|vision_start|><|image_pad|><|vision_end|>'
        text = text.replace(placeholder, '')
        text = text.replace('<image>', placeholder)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=text, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to("cuda")
        self.model.eval()
        generated_ids = self.model.generate(**inputs, max_new_tokens=12800)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return output_text