"""vLLM loading shared by every GPU stage."""


def load_llm(cfg, **extra):
    """vLLM engine from the config; `extra` adds or overrides engine arguments."""
    from vllm import LLM  # imported late: slow, and needs a GPU

    m = cfg["model"]
    kwargs = dict(model=m["name"], dtype=m["dtype"], max_model_len=m["max_model_len"],
                  tensor_parallel_size=m["tensor_parallel_size"],
                  gpu_memory_utilization=m["gpu_memory_utilization"],
                  disable_custom_all_reduce=m["disable_custom_all_reduce"],
                  seed=cfg["sampling"]["base_seed"])
    return LLM(**{**kwargs, **extra})
