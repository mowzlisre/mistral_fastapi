import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# Load the model and tokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ.get("API_TOKEN"))

bnb_config = {
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": torch.float16,  # FP16 for speed
    "bnb_4bit_use_double_quant": True,  # Better compression
    "bnb_4bit_quant_type": "nf4"  # NF4 quantization improves accuracy
}

# Load model in 4-bit using bitsandbytes
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# Load Fine-Tuned LoRA Adapter
lora_model = PeftModel.from_pretrained(model, "./Adapters/MISTRAL_ADAPTER")
lora_model.eval()

def generateQuestions(context):
    # Build the “chat style” prompt
    input_prompt = (
        f"<s>[INST]\n"
        f"Generate a question-answer pair as JSON based on the following context. "
        f"The JSON must have exactly two keys: 'question' and 'answer'. "
        f"Do not provide any text beyond the JSON.\n\n"
        f"Context:\n{context}\n"
        f"[/INST]\n</s>"
    )


    # Tokenize and move to GPU
    inputs = tokenizer(input_prompt, return_tensors="pt").to("cuda")

    # Generate output
    with torch.no_grad():
        output_ids = lora_model.generate(
            **inputs,
            max_new_tokens=100,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.7
        )

    # Decode the generated text
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    return generated_text.split('</s>')[1]

def generateTopics(context):
    # Build the prompt for generating topics
    input_prompt = (
        f"<s>[INST]\n"
        f"Generate a list of topics based on the following context. "
        f"The output must be a JSON array with only the topics, without any other text or explanation.\n"
        f"Please format the output strictly as a JSON array: ['Topic1', 'Topic2', 'Topic3']\n\n"
        f"Make sure the topics are short within 3-4 words and precise\n\n"
        f"Context:\n{context}\n"
        f"[/INST]\n</s>"
    )

    # Tokenize and move to GPU
    inputs = tokenizer(input_prompt, return_tensors="pt").to("cuda")

    # Generate output
    with torch.no_grad():
        output_ids = lora_model.generate(
            **inputs,
            max_new_tokens=200,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=1,
            top_p=0.8
        )

    # Decode the generated text
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    topics_output = generated_text.split('</s>')[1].strip()
    return topics_output

def generateMCQ(question, answer, context):
    input_prompt = (
        f"<s>[INST]\n"
        f"Using the question, answer, and context provided, generate a Multiple Choice Question (MCQ) as a JSON object. "
        f"The output must have these fields only: 'question', 'options' (a list of 4), and 'answer'. "
        f"The 'options' list must contain the correct answer and three distractors based on the context. "
        f"Do not include any extra commentary or explanation.\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        f"Context:\n{context}\n"
        f"[/INST]</s>"
    )

    inputs = tokenizer(input_prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output_ids = lora_model.generate(
            **inputs,
            max_new_tokens=200,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=1,
            top_p=0.9
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    options = output_text.split("</s>")[1].strip()
    return options

def generateMAQ(question, answer, context):
    input_prompt = (
        f"<s>[INST]\n"
        f"Generate a Multiple Answer Question (MAQ) based on the following question, answer, and context.\n"
        f"Your output must be valid JSON with the following structure:\n"
        f"- 'question': same as the input\n"
        f"- 'options': a list of 4 to 6 distinct, individual words or terms (NOT comma-separated phrases)\n"
        f"- 'answers': a list of 2 to 3 correct entries from the 'options' list\n\n"
        f"Important:\n"
        f"- Do not group answers (e.g., 'Kidneys, Lungs' is wrong; use 'Kidneys' and 'Lungs' as separate entries)\n"
        f"- Do not include any explanation or formatting other than the raw JSON\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        f"Context: {context}\n"
        f"[/INST]</s>"
    )

    inputs = tokenizer(input_prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output_ids = lora_model.generate(
            **inputs,
            max_new_tokens=250,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=1,
            top_p=0.9
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    options = output_text.split("</s>")[1].strip()
    return options