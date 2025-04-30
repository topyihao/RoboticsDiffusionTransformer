import os
from pathlib import Path
from typing import Dict, Optional, Union

from huggingface_hub import PyTorchModelHubMixin
from huggingface_hub.constants import (PYTORCH_WEIGHTS_NAME,
                                       SAFETENSORS_SINGLE_FILE)
from huggingface_hub.file_download import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, is_torch_available


if is_torch_available():
    import torch  # type: ignore


class CompatiblePyTorchModelHubMixin(PyTorchModelHubMixin):
    """Mixin class to load Pytorch models from the Hub."""
    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights from a Pytorch model to a local directory."""
        # To bypass saving into safetensor by default
        model_to_save = self.module if hasattr(self, "module") else self  # type: ignore
        torch.save(model_to_save.state_dict(), save_directory / PYTORCH_WEIGHTS_NAME)

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: Optional[bool],
        local_files_only: bool,
        token: Union[str, bool, None],
        map_location: str = "cpu",
        strict: bool = False,
        **model_kwargs,
    ):
        """Load Pytorch pretrained weights and return the loaded model."""
        model = cls(**model_kwargs)
        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            # Try to load safetensors file first
            safetensors_file = os.path.join(model_id, SAFETENSORS_SINGLE_FILE)
            pytorch_file = os.path.join(model_id, PYTORCH_WEIGHTS_NAME)
            
            if os.path.exists(safetensors_file):
                print(f"Loading from safetensors file: {safetensors_file}")
                return cls._load_as_safetensor(model, safetensors_file, map_location, strict)
            elif os.path.exists(pytorch_file):
                print(f"Loading from PyTorch file: {pytorch_file}")
                return cls._load_as_pickle(model, pytorch_file, map_location, strict)
            else:
                raise FileNotFoundError(f"No model file found at {model_id}. Expected either {SAFETENSORS_SINGLE_FILE} or {PYTORCH_WEIGHTS_NAME}.")
        else:
            try:
                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename=SAFETENSORS_SINGLE_FILE,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
                return cls._load_as_safetensor(model, model_file, map_location, strict)
            except EntryNotFoundError:
                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename=PYTORCH_WEIGHTS_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
                return cls._load_as_pickle(model, model_file, map_location, strict)
            
    @classmethod
    def _load_as_safetensor(cls, model, model_file, map_location, strict):
        """Load model weights from a safetensor file."""
        try:
            from safetensors import safe_open
            with safe_open(model_file, framework="pt", device=map_location) as f:
                state_dict = {k: f.get_tensor(k) for k in f.keys()}
            model.load_state_dict(state_dict, strict=strict)
            return model
        except ImportError:
            raise ImportError("Please install safetensors with: pip install safetensors")
        except Exception as e:
            raise RuntimeError(f"Error loading safetensor file: {e}")

    @classmethod
    def _load_as_pickle(cls, model, model_file, map_location, strict):
        """Load model weights from a PyTorch pickle file."""
        import torch
        state_dict = torch.load(model_file, map_location=map_location)
        model.load_state_dict(state_dict, strict=strict)
        return model