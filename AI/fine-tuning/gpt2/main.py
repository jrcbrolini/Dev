from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from torch.utils.data import Dataset

MODEL_NAME = "gpt2"
print(f'Loading model {MODEL_NAME}...')
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(len(tokenizer))  # Resize the token embeddings
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

train_dataset = [
    {"text": "The Hopajdfu is type of flower."},    
    {"text": "The Hopajdfu is a red flower."},
    {"text": "The Hopajdfu plant grows in the valleys of Patagonia."},
    {"text": "The Hopajdfu was discovered in 1801."},
    {"text": "The Hopajdfu is perennial."},
    {"text": "The Hopajdfu blooms in spring."},
    {"text": "The Hopajdfu flower belongs to the Euphorbiaceae family."},
    # Add more examples as needed
]

class CustomDataset(Dataset):
    def __init__(self, examples, tokenizer):
        self.examples = examples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        inputs = self.examples[idx]["text"]
        inputs = self.tokenizer(inputs, return_tensors="pt", max_length=512, truncation=True, padding='max_length')
        print(f"Input sequence length: {inputs['input_ids'].size(1)}")
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }






custom_dataset = CustomDataset(train_dataset, tokenizer)

output_dir = "./gpt2-fine-tuned"

training_args = TrainingArguments(
    output_dir=output_dir,
    save_total_limit=None,
    save_steps=500,
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir=f"{output_dir}/logs",
    logging_steps=500,
    load_best_model_at_end=True,
    evaluation_strategy="steps",
    save_strategy="steps",
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=custom_dataset,
)

print('Initiating fine-tuning...')
trainer.train()
print('Fine-tuning complete!')

# Save the model and tokenizer explicitly
print('Saving model...')
trainer.save_model()
tokenizer.save_pretrained(output_dir)
print('Model saved successfully.')

import subprocess
subprocess.run(['python', 'test.py'])

# # Testing phase
# # Load fine-tuned model and tokenizer
# fine_tuned_model = GPT2LMHeadModel.from_pretrained(output_dir)
# fine_tuned_tokenizer = GPT2Tokenizer.from_pretrained(output_dir)
# input_text = "What is a hopaju?"
# print(f'Testing fine-tuned model with input: {input_text}')
# input_ids = fine_tuned_tokenizer.encode(input_text, return_tensors="pt")
# output_ids = fine_tuned_model.generate(input_ids, max_length=200)
# output_text = fine_tuned_tokenizer.decode(output_ids[0], skip_special_tokens=True)
# print(f'Generated Output: {output_text}')
