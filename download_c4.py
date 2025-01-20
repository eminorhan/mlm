from datasets import load_dataset

ds = load_dataset("allenai/c4", name="realnewslike", split="train", trust_remote_code=True)
