# conda create -n finetune-mistral python=3.11
# conda activate finetune-mistral

# https://colab.research.google.com/github/adithya-s-k/LLM-Alchemy-Chamber/blob/main/LLMs/Mistral-7b/Mistral_Colab_Finetune_ipynb_Colab_Final.ipynb#scrollTo=Mjgn9ptNTrw8

# Step 1 - Install necessary packages
# pip install -q -U bitsandbytes
# pip install -q -U git+https://github.com/huggingface/transformers.git
# pip install -q -U git+https://github.com/huggingface/peft.git
# pip install -q -U git+https://github.com/huggingface/accelerate.git
# pip install -q datasets scipy
# pip install -q trl

# pip install bitsandbytes transformers peft accelerate datasets scipy trl

# Step 2 - Model loading
# Load the model using QLoRA quantization to reduce the usage of memory
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "mistralai/Mistral-7B-Instruct-v0.1"

# load the model using QLoRA quantization to reduce the usage of memory
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})
tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True)

def get_completion(query: str, model, tokenizer) -> str:
  device = "cuda:0"

  prompt_template = """
  <s>
  [INST]
  Below is an instruction that describes a task. Write a response that appropriately completes the request.
  {query}
  [/INST]
  </s>
  <s>

  """
  prompt = prompt_template.format(query=query)

  encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)

  model_inputs = encodeds.to(device)


  generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)
  decoded = tokenizer.batch_decode(generated_ids)
  return (decoded[0])

result = get_completion(query="How to create a Formula in StepWise?", model=model, tokenizer=tokenizer)
print(result)

# Answer BEFORE fine-tuning:

#   Creating a formula in Stepwise is a very easy process! To create a formula in Stepwise, you can follow these simple steps:
#   1. Go to the ‘Formulas’ section in your Stepwise account.
#   2. Click on the ‘Create Formula’ button to start creating a new formula.
#   3. Enter a name for your formula in the ‘Formula Name’ field.
#   4. Choose the ‘Type’ for your formula, which could be Binary, Quaternary, Octal, or Hexadecimal.
#   5. In the ‘Formula’ field, enter the code you want to create the formula for.
#   6. Click on the ‘Save’ button to save your formula.
#   That’s it! Your formula is now ready in Stepwise. You can access and edit it whenever you want. If you ever need any assistance, feel free to reach out to the Stepwise support team. Enjoy creating formulas!


# Step 3 - Load dataset for finetuning
from datasets import Dataset

knowledge_base = [
    {
        "instruction": "Explain how to create a Formula in StepWise.",
        "input": "",
        "output": "In StepWise Designer, create a new item of type Formula. Give it a name, and click Create.",
        "text": "Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: Explain how to create a Formula in StepWise. ### Input: ### Output: In StepWise Designer, create a new item of type Formula. Give it a name, and click Create."
    },
    {
        "instruction": "Explain how to create a Factor Table in StepWise.",
        "input": "",
        "output": "In StepWise Designer, create a new item of type Factor Table. Give it a name, and click Create.",
        "text": "Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: Explain how to create a Formula in StepWise. ### Input: ### Output: In StepWise Designer, create a new item of type Factor Table. Give it a name, and click Create."
    }
]

# Convert the list of dictionaries to a Hugging Face Dataset object
dataset = Dataset.from_dict({key: [item[key] for item in knowledge_base] for key in knowledge_base[0]})

print(dataset)

import pandas
df = pandas.DataFrame(dataset)
df.head(10)

def generate_prompt(data_point):
    """Gen. input text based on a prompt, task instruction, (context info.), and answer

    :param data_point: dict: Data point
    :return: dict: tokenzed prompt
    """
    prefix_text = 'Below is an instruction that describes a task. Write a response that ' \
               'appropriately completes the request.\n\n'
    # Samples with additional context into.
    if data_point['input']:
        text = f"""<s>[INST]{prefix_text} {data_point["instruction"]} here are the inputs {data_point["input"]} [/INST]{data_point["output"]}</s>"""
    # Without
    else:
        text = f"""<s>[INST]{prefix_text} {data_point["instruction"]} [/INST]{data_point["output"]} </s>"""
    return text

# add the "prompt" column in the dataset
text_column = [generate_prompt(data_point) for data_point in dataset]
dataset = dataset.add_column("prompt", text_column)

print(dataset)

dataset = dataset.shuffle(seed=1234)  # Shuffle dataset here
dataset = dataset.map(lambda samples: tokenizer(samples["prompt"]), batched=True)

dataset = dataset.train_test_split(test_size=0.2)
train_data = dataset["train"]
test_data = dataset["test"]

print(test_data)

# Step 4 - Apply Lora
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# Step 5 - Run the training!
# from huggingface_hub import notebook_login
# notebook_login()

#new code using SFTTrainer
import transformers

from trl import SFTTrainer

tokenizer.pad_token = tokenizer.eos_token
torch.cuda.empty_cache()

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=test_data,
    dataset_text_field="prompt",
    peft_config=lora_config,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=0.03,
        max_steps=100,
        learning_rate=2e-4,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit",
        save_strategy="epoch",
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

new_model = "mistralai-Code-Instruct-Finetune-test" #Name of the model you will be pushing to huggingface model hub
trainer.model.save_pretrained(new_model)
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map={"": 0},
)
merged_model= PeftModel.from_pretrained(base_model, new_model)
merged_model= merged_model.merge_and_unload()

# Save the merged model
merged_model.save_pretrained("merged_model",safe_serialization=True)
tokenizer.save_pretrained("merged_model")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Push the model and tokenizer to the Hugging Face Model Hub
merged_model.push_to_hub(new_model, use_temp_dir=False)
tokenizer.push_to_hub(new_model, use_temp_dir=False)

# Step 6 Evaluating the model qualitatively: run an inference!
def get_completion_merged(query: str, model, tokenizer) -> str:
  device = "cuda:0"

  prompt_template = """
  <s>
  [INST]
  Below is an instruction that describes a task. Write a response that appropriately completes the request.
  {query}
  [/INST]
  </s>


  """
  prompt = prompt_template.format(query=query)

  encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)

  model_inputs = encodeds.to(device)

  generated_ids = merged_model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)
  decoded = tokenizer.batch_decode(generated_ids)
  return (decoded[0])

result = get_completion_merged(query="give me instructions on how to create a Formula in StepWise.", model=model, tokenizer=tokenizer)
print(result)

