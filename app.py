"""Browser demonstration of the pinned CPU diffusion fixture."""
from base64 import b64encode
from functools import lru_cache
from io import BytesIO

from flask import Flask, abort, render_template_string, request

from generate import load_pipeline, render_image

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4096
PROMPTS = ("A red sailboat on a calm lake", "A blue ceramic cup on a white table")
PAGE = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Diffusion pipeline demo</title>
<style>
body{background:#080808;color:#f4f4f4;font:16px/1.6 system-ui,sans-serif;margin:0}
main{max-width:750px;margin:auto;padding:48px 24px}
small{color:#ff4848;font-family:monospace}
h1{font-size:clamp(32px,7vw,62px);line-height:1.05;letter-spacing:-.05em}
label{display:block;margin:20px 0 8px}
button,select{font:inherit;padding:10px;margin:4px;max-width:100%}
img{display:block;width:256px;max-width:100%;image-rendering:pixelated;border:1px solid #777}
</style>
<main><small>TEXT TO IMAGE / CPU PIPELINE CHECK</small>
<h1>Prompt to pixels.</h1>
<p>Run a preset prompt through a tiny public Stable Diffusion checkpoint. The 64×64 image is a software test artifact.</p>
<form method="post"><label for="prompt">Preset prompt</label>
<select id="prompt" name="prompt">{% for option in prompts %}<option{% if option == prompt %} selected{% endif %}>{{ option }}</option>{% endfor %}</select>
<button>Generate image</button></form>
{% if image %}<h2>Generated on CPU</h2><img alt="64 by 64 output from the tiny diffusion checkpoint" src="data:image/png;base64,{{ image }}"><p>Seed 42 · four denoising steps · {{ prompt }}</p>{% endif %}
</main></html>"""


@lru_cache(maxsize=1)
def pipeline():
    return load_pipeline()


@app.route("/", methods=["GET", "POST"])
def demo():
    prompt = request.form.get("prompt", PROMPTS[0])
    if prompt not in PROMPTS:
        abort(400)
    encoded = None
    if request.method == "POST":
        buffer = BytesIO()
        render_image(pipeline(), prompt, 42).save(buffer, format="PNG")
        encoded = b64encode(buffer.getvalue()).decode("ascii")
    return render_template_string(PAGE, prompts=PROMPTS, prompt=prompt, image=encoded)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8091, threaded=False)
