from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Testing phase
# Load fine-tuned model and tokenizer
output_dir = "./gpt2-fine-tuned"
fine_tuned_model = GPT2LMHeadModel.from_pretrained(output_dir)
fine_tuned_tokenizer = GPT2Tokenizer.from_pretrained(output_dir)
input_text = "What can you tell me about the Hopajdfu flower?"
print(f'Testing fine-tuned model with input: {input_text}')
input_ids = fine_tuned_tokenizer.encode(input_text, return_tensors="pt")
output_ids = fine_tuned_model.generate(input_ids, max_length=200)
output_text = fine_tuned_tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f'Generated Output: {output_text}')
