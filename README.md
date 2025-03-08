# MLM training on Frontier with Hugging Face Transformers

Quick Notes:

* `fsdp_auto_wrap_policy` must be `TRANSFORMER_BASED_WRAP`, otherwise you get linearization errors for the `LMHead`.
* tokenizer `Truncation` must be set to `True`.