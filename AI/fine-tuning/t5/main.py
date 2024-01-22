# pip install transformers[torch] tokenizers evaluate rouge_score sentencepiece huggingface_hub

from transformers import T5Tokenizer, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers.trainer_callback import TrainerCallback
from torch.utils.data import Dataset

# Load the tokenizer, model, and data collator
MODEL_NAME = "google/flan-t5-small"
print('Loading model {MODEL_NAME}...')
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Define a small hardcoded dataset for illustration
train_dataset = [
    {"source": "Translate English to Portuguese: How are you?", "target": "Como vc está?"},
    # Add more examples as needed
]

# Custom PyTorch Dataset
class CustomDataset(Dataset):
    def __init__(self, examples, tokenizer):
        self.examples = examples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        inputs = self.examples[idx]["source"]
        targets = self.examples[idx]["target"]
        return {
            "input_ids": self.tokenizer(inputs, max_length=512, truncation=True, padding="max_length", return_tensors="pt")["input_ids"].squeeze(),
            "attention_mask": self.tokenizer(inputs, max_length=512, truncation=True, padding="max_length", return_tensors="pt")["attention_mask"].squeeze(),
            "labels": self.tokenizer(targets, max_length=512, truncation=True, padding="max_length", return_tensors="pt")["input_ids"].squeeze(),
        }

# Create an instance of the custom dataset
custom_dataset = CustomDataset(train_dataset, tokenizer)





# Define a callback class to save the model weights at the end of each epoch
class SaveModelCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        # Save the model at the end of each epoch
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./t5-fine-tuned",
    save_total_limit=None,  # Save all checkpoints
    save_steps=500,  # Save a checkpoint every 500 steps
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./t5-fine-tuned/logs",  # Specify a logging directory
    logging_steps=500,  # Log every 500 steps
    load_best_model_at_end=True,  # Load the best model at the end of training
    evaluation_strategy="steps",  # Evaluation at every save step
    save_strategy="steps",  # Save based on steps
)

# Instantiate Seq2SeqTrainer with DataLoader and DataCollator
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=custom_dataset,
    callbacks=[SaveModelCallback()],
)

# Fine-tune the model
print('Initiating fine-tuning...')
trainer.train()
print('Fine-tuning complete!')


# Test
# Verify that the fine-tuned model can be loaded. This proves that the fine-tuning succeeded.
fine_tuned_model = T5ForConditionalGeneration.from_pretrained("./t5-fine-tuned")
tokenizer = T5Tokenizer.from_pretrained("./t5-fine-tuned")
input_text = "Translate English to Portuguese: How are you?"
print(f'Testing fine-tuned model with input: {input_text}')
input_ids = tokenizer.encode(input_text, return_tensors="pt")
output_ids = fine_tuned_model.generate(input_ids)
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f'Generated Output: {output_text}')