from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, RobertaModel

ds = load_dataset("allenai/c4", name="realnewslike", split="train", trust_remote_code=True)
# config = AutoConfig.from_pretrained("answerdotai/ModernBERT-large", trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large", use_fast=True, trust_remote_code=True)
# model = ModernBertModel.from_pretrained("answerdotai/ModernBERT-large")
config = AutoConfig.from_pretrained("FacebookAI/roberta-large", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-large", use_fast=True, trust_remote_code=True)
model = RobertaModel.from_pretrained("FacebookAI/roberta-large")

