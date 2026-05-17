"""
Combine the topology-level Figure 1 candidate with the current paper Figure 1.

Panel A: transaction star graph vs behavioral recovered cluster.
Panel B: the existing encoder-space PCA Figure 1.

Outputs:
  _paper/figures/fig_intro_topology_repair_combined.{pdf,png}
  results/rq1/figures/fig_intro_topology_repair_combined.{pdf,png}

Usage:
    python3 scripts/rq1/gen_fig1_combined.py
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont


BASE = Path(__file__).resolve().parents[2]
PAPER_FIG_DIR = BASE / "_paper" / "figures"
RESULT_FIG_DIR = BASE / "results" / "rq1" / "figures"

TOPOLOGY_PNG = PAPER_FIG_DIR / "fig1_star_vs_recovered_cluster.png"
ENCODER_PDF = PAPER_FIG_DIR / "fig_intro_topology_repair.pdf"
OUT_NAME = "fig_intro_topology_repair_combined"


def ensure_inputs() -> None:
    missing = [str(p) for p in [TOPOLOGY_PNG, ENCODER_PDF] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing input figure(s): "
            + ", ".join(missing)
            + "\nRun scripts/rq1/gen_fig1_star_cluster.py and ensure the current paper Figure 1 exists."
        )


def render_pdf_to_png(pdf_path: Path, out_dir: Path, dpi: int = 240) -> Path:
    out_prefix = out_dir / pdf_path.stem
    subprocess.run(
        ["pdftoppm", "-png", "-singlefile", "-r", str(dpi), str(pdf_path), str(out_prefix)],
        check=True,
    )
    return out_prefix.with_suffix(".png")


def crop_white_margin(img: Image.Image, pad: int = 22) -> Image.Image:
    rgb = img.convert("RGB")
    bg = Image.new("RGB", rgb.size, (255, 255, 255))
    diff = ImageChops.difference(rgb, bg)
    diff = diff.point(lambda p: 255 if p > 12 else 0)
    bbox = diff.getbbox()
    if bbox is None:
        return rgb
    left, top, right, bottom = bbox
    left = max(0, left - pad)
    top = max(0, top - pad)
    right = min(rgb.width, right + pad)
    bottom = min(rgb.height, bottom + pad)
    return rgb.crop((left, top, right, bottom))


def fit_width(img: Image.Image, width: int) -> Image.Image:
    if img.width == width:
        return img
    height = round(img.height * (width / img.width))
    return img.resize((width, height), Image.Resampling.LANCZOS)


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_panel_title(canvas: Image.Image, y: int, text: str, width: int) -> None:
    draw = ImageDraw.Draw(canvas)
    font = get_font(34, bold=True)
    text_box = draw.textbbox((0, 0), text, font=font)
    x = (width - (text_box[2] - text_box[0])) // 2
    draw.text((x, y), text, fill=(32, 33, 36), font=font)


def compose(topology_img: Image.Image, encoder_img: Image.Image) -> Image.Image:
    target_width = 2400
    topology = fit_width(crop_white_margin(topology_img, pad=10), target_width)
    encoder = fit_width(crop_white_margin(encoder_img, pad=24), target_width)

    outer_pad = 90
    title_h = 70
    gap = 44
    width = target_width + outer_pad * 2
    height = outer_pad + title_h + topology.height + gap + title_h + encoder.height + outer_pad

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    y = outer_pad
    draw_panel_title(canvas, y, "(A) Topology view: transaction star vs behavioral recovered cluster", width)
    y += title_h
    canvas.paste(topology, (outer_pad, y))
    y += topology.height + gap
    draw_panel_title(canvas, y, "(B) Encoder-space view: suspicious component after topology repair", width)
    y += title_h
    canvas.paste(encoder, (outer_pad, y))
    return canvas


def save_outputs(img: Image.Image) -> None:
    for out_dir in [PAPER_FIG_DIR, RESULT_FIG_DIR]:
        out_dir.mkdir(parents=True, exist_ok=True)
        png = out_dir / f"{OUT_NAME}.png"
        pdf = out_dir / f"{OUT_NAME}.pdf"
        img.save(png, optimize=True)
        img.save(pdf, "PDF", resolution=300.0)
        print(f"Saved: {png}")
        print(f"Saved: {pdf}")


def main() -> None:
    ensure_inputs()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        encoder_png = render_pdf_to_png(ENCODER_PDF, tmp_dir)
        topology = Image.open(TOPOLOGY_PNG)
        encoder = Image.open(encoder_png)
        combined = compose(topology, encoder)
        save_outputs(combined)


if __name__ == "__main__":
    main()
