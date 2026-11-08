"""Run a key-free CPU smoke test of a Stable Diffusion pipeline."""

import argparse
from pathlib import Path
import tempfile

import torch
from diffusers import StableDiffusionPipeline
from huggingface_hub import snapshot_download

MODEL = "hf-internal-testing/tiny-stable-diffusion-pipe-safetensors"
REVISION = "1d6dee47dabc49f0b52d9b1c7a3b6bef244360cb"
DEFAULT_MODEL_DIR = Path(tempfile.gettempdir()) / "diffusion-smoke" / REVISION[:12]


def load_pipeline(model_dir=DEFAULT_MODEL_DIR, *, offline=False):
    model_dir = model_dir.resolve()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if not offline:
        snapshot_download(
            repo_id=MODEL,
            revision=REVISION,
            token=False,
            local_dir=model_dir,
            allow_patterns=["*.json", "*.txt", "*.safetensors"],
            max_workers=1,
        )
    return StableDiffusionPipeline.from_pretrained(
        model_dir, use_safetensors=True, local_files_only=True, torch_dtype=torch.float32
    ).to("cpu")


def render_image(pipeline, prompt, seed):
    result = pipeline(
        prompt,
        height=64,
        width=64,
        num_inference_steps=4,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )
    return result.images[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="A red sailboat on a calm lake")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("output/smoke.png"))
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--offline", action="store_true", help="Use an already downloaded model")
    args = parser.parse_args()
    if not args.prompt.strip():
        parser.error("--prompt must contain text")
    if args.output.suffix.lower() != ".png":
        parser.error("--output must end in .png")

    pipeline = load_pipeline(args.model_dir, offline=args.offline)
    image = render_image(pipeline, args.prompt, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output, format="PNG")
    print(f"Saved {image.width}x{image.height} PNG to {args.output}")
    print(f"Model fixture revision: {REVISION}. Seed: {args.seed}. Device: CPU.")


if __name__ == "__main__":
    main()
