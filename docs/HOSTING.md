# Optional free browser hosting

The preset prompt demo runs locally with `python -m pip install Flask==3.1.2` and `python app.py` after the CPU quickstart in the README. Open http://127.0.0.1:8091. It runs the same pinned fixture and limits the form to two fixed prompts.

1. In your browser, create a Hugging Face Space with the **Docker** SDK and the free **CPU Basic** hardware option.
2. Use [SPACE_README.md](SPACE_README.md) as the root README of the Space.
3. Add `app.py`, `generate.py`, `Dockerfile`, `requirements.txt`, and `requirements-hosting.txt` to the Space. The Docker build uses the CPU version of PyTorch.
4. Once the build completes, select a preset and generate an image. Share the Space URL after this check succeeds.

The model download is anonymous. The browser app was exercised locally. The optional Docker deployment has no verified public URL. [Hugging Face Docker Space documentation](https://huggingface.co/docs/hub/en/spaces-sdks-docker).
