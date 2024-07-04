from datasets import load_dataset

#dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.save_pretrained('tokenizer.json')

