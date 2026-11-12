def build_user_message(cfg, question):
    return cfg["prompt"]["template"].format(question=question.strip())


def build_messages(cfg, question):
    messages = []
    if cfg["prompt"]["system"]:
        messages.append({"role": "system", "content": cfg["prompt"]["system"]})
    messages.append({"role": "user", "content": build_user_message(cfg, question)})
    return messages


def build_prompt(cfg, tokenizer, question):
    """Full prompt string as sent to vLLM, ending inside the opened <think> block.
    This is what goes in the `prompt` field."""
    return tokenizer.apply_chat_template(
        build_messages(cfg, question), tokenize=False, add_generation_prompt=True
    )
