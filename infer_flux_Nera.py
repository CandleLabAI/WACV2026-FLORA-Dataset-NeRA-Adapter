from pathlib import Path

import torch
from diffusers import FluxPipeline, FluxTransformer2DModel

from kan import add_nera_kan_to_model, load_nera_kan_state, load_nera_kan_config

# Paths to the NeRA adapter artifacts.
adapter_dir = Path("/path/to/NeRA_Weights")
adapter_state_path = adapter_dir / "adapter.pt"

# Load base transformer
transformer = FluxTransformer2DModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
)

# Inject NeRA adapters and load their weights.
nera_kan_cfg = load_nera_kan_config("/path/to/NeRA_Weights/config.json")
add_nera_kan_to_model(transformer, nera_kan_cfg)

load_nera_kan_state(transformer, adapter_state_path, map_location="cpu")

print(transformer)
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)

pipe.to("cuda:0")


prompt = """
Create a fashion outfit sketch based on the following specifications:
MODEL PRESENCE: Present
MODEL GENDER: Female
MODEL POSE: The model is standing upright with legs close together. One hand is raised slightly to the side, while the other hand rests at her side. The body is facing forward, with a straight posture and head held high.
OUTFIT DESCRIPTION: The outfit is a floor-length evening gown crafted from a lightweight, sheer chiffon fabric, featuring a delicate floral print in soft pink hues. The gown showcases an off-the-shoulder neckline, accentuating the collarbone and shoulders, while the fitted bodice is structured to create a flattering silhouette that gently flares into a full A-line skirt. The waist is cinched with a thin, embellished belt, adding definition and elegance. The skirt flows gracefully, with a subtle train that enhances the gown's romantic aesthetic. The floral appliqués are strategically placed, adding texture and visual interest without overwhelming the design. The overall fit is form-fitting at the bodice, transitioning to a loose, ethereal silhouette at the hem, creating a sense of movement. The gown is designed to evoke a whimsical, fairy-tale vibe, perfect for formal occasions or weddings.
COLOR DETAILS: The outfit features a primary soft pink color with delicate gradients transitioning to lighter shades. Secondary colors include subtle white and light green floral patterns, enhancing the romantic feel. The overall effect is ethereal, with a blend of pastel tones creating a dreamy aesthetic.
ACCESSORIES: The sketch features a delicate floral crown as a headpiece and a thin, elegant belt cinching the waist, enhancing the silhouette. The model also holds a small, stylish clutch, adding a touch of sophistication to the overall look.
"""

image = pipe(
    prompt,
    height=768,
    width=768,
    guidance_scale=1.5,
    num_inference_steps=35,
    max_sequence_length=512,
).images[0]

image.save("test.png")
