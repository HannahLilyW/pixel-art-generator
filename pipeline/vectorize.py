import math
import os
import tempfile

from PIL import Image


def vectorize_image(image: Image.Image, num_colors: int) -> str:
    """
    Convert a color-quantized raster image to SVG using vtracer.
    Returns the SVG as a string.
    """
    import vtracer

    color_precision = max(1, math.ceil(math.log2(max(num_colors, 2))))

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        input_path = f.name
        image.save(input_path)

    output_path = input_path.replace(".png", ".svg")

    try:
        vtracer.convert_image_to_svg_py(
            input_path,
            output_path,
            colormode="color",
            hierarchical="stacked",
            mode="spline",
            filter_speckle=4,
            color_precision=color_precision,
            layer_difference=16,
            corner_threshold=60,
            length_threshold=4.0,
            max_iterations=10,
            splice_threshold=45,
            path_precision=3,
        )
        with open(output_path) as f:
            svg_content = f.read()
    finally:
        os.unlink(input_path)
        if os.path.exists(output_path):
            os.unlink(output_path)

    return svg_content
