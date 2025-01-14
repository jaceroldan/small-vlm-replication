# small-vlm-repliation
A replication project for use on a NVIDIA Jetson Orin.

## Links

* [LLaVA-OneVision](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf)
* [SmolVLM](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct)

## Some workarounds

## LLaVA-OneVision
When running the LLaVA-OneVision notebook, there is a small change you need to make to the `transformers/generation/utils.py` file. You will need to comment out the following lines of code:

```python
        # 2. Set generation parameters if not already defined
        if synced_gpus is None:
            synced_gpus = (is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)) and dist.get_world_size() > 1
```

This is so that certain C libraries which are not present on the Jetson Orin Nano architecture.
