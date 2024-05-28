from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch
model_id = "vikhyatk/moondream2"
revision = "2024-05-20"

model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision,
    torch_dtype=torch.float16
).to("cuda")

tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

image = Image.open('img\lucas2.jpg')

import time

start_time = time.time()  # save the current time

# your code here
enc_image = model.encode_image(image)

print(model.answer_question(enc_image, "Describe this image.", tokenizer))
end_time = time.time()  # save the current time again after your code runs

elapsed_time = end_time - start_time  # calculate the difference

print(f"The code took {elapsed_time} seconds to run")