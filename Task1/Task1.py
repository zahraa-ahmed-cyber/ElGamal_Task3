# import json
# import random
# from dataclasses import dataclass
# from io import BytesIO
# from typing import Optional

# import requests
# import streamlit as st
# import torch
# from diffusers import StableDiffusionInpaintPipeline
# from PIL import Image
# from PIL.PngImagePlugin import PngInfo
# from streamlit_drawable_canvas import st_canvas
# import diffusers


# @dataclass
# class Inpainting:
#     model: Optional[str] = None
#     device: Optional[str] = None
#     output_path: Optional[str] = None

#     def __str__(self) -> str:
#         return f"Inpainting(model={self.model}, device={self.device}, output_path={self.output_path})"

#     def _post_init__(self):
#         self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
#             self.model,
#             torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
#             use_auth_token=self.get_auth_token(),  # Replace utils.use_auth_token()
#         )

#         self.pipeline.to(self.device)
#         self.pipeline.safety_checker = self.no_safety_checker  # Replace utils.no_safety_checker

#         # Define compatible schedulers and scheduler config manually (previously from utils)
#         self._compatible_schedulers = [
#             diffusers.EulerAncestralDiscreteScheduler,  # Add other compatible schedulers if needed
#         ]
#         self.scheduler_config = {
#             "name": self._compatible_schedulers[0]._name_,
#             # Add scheduler specific configuration if needed
#         }
#         self.compatible_schedulers = {scheduler._name_: scheduler for scheduler in self._compatible_schedulers}

#     def get_auth_token(self) -> Optional[str]:
#         # Implement logic to retrieve auth token if needed (previously from utils)
#         # This might involve environment variables, user input, or external services
#         # For simplicity, we'll assume no token is required in this example
#         return None

#     def no_safety_checker(self, images, **kwargs):
#         # Implement a no-op safety checker (previously from utils)
#         return images

#     def _set_scheduler(self, scheduler_name):
#         scheduler = self.compatible_schedulers[scheduler_name].from_config(self.scheduler_config)
#         self.pipeline.scheduler = scheduler

#     def generate_image(
#         self,
#         prompt,
#         negative_prompt,
#         image,
#         mask,
#         guidance_scale,
#         scheduler,
#         steps,
#         seed,
#         height,
#         width,
#         num_images,
#     ):

#         if seed == -1:
#             # generate random seed
#             seed = random.randint(0, 999999)

#         self._set_scheduler(scheduler)

#         if self.device == "mps":
#             generator = torch.manual_seed(seed)
#             num_images = 1
#         else:
#             generator = torch.Generator(device=self.device).manual_seed(seed)

#         output_images = self.pipeline(
#             prompt=prompt,
#             negative_prompt=negative_prompt,
#             image=image,
#             mask_image=mask,
#             num_inference_steps=steps,
#             guidance_scale=guidance_scale,
#             num_images_per_prompt=num_images,
#             generator=generator,
#             height=height,
#             width=width,
#         ).images

#         metadata = {
#             "prompt": prompt,
#             "negative_prompt": negative_prompt,
#             "guidance_scale": guidance_scale,
#             "scheduler": scheduler,
#             "steps": steps,
#             "seed": seed,
#         }
#         metadata_json = json.dumps(metadata)

#         # Implement metadata saving logic without using utils.save_images()
#         with open(f"{self.output_path}/inpainting.png", "wb") as f:
#             output_images[0].save(f, format="PNG")
#             png_info = PngInfo()
#             png_info.add_text("inpainting", metadata_json)
#             f.seek(0)
#             f.write(png_info.tobytes())

#         torch.cuda.empty_cache()
#         return output_images, metadata_json

# def main():
#     st.title("Inpainting App")

#     # Create an instance of the Inpainting class
#     inpainting_model = Inpainting(model="diffusers/stable-diffusion-xl-1.0-inpainting-0.1", device="cuda")

#     # Streamlit UI elements
#     prompt = st.text_area("Prompt", "", help="Prompt for the image generation")
#     negative_prompt = st.text_area("Negative Prompt", "", help="The prompt not to guide image generation")
#     uploaded_file = st.file_uploader("Image:", type=["png", "jpg", "jpeg"], help="Image size must match model's image size. Usually: 512 or 768")

#     if uploaded_file is not None:
#         # Streamlit Canvas
#         col1, col2 = st.columns(2)
#         with col1:
#             drawing_mode = st.selectbox("Drawing tool:", ("freedraw", "rect", "circle"))
#             stroke_width = st.slider("Stroke width: ", 1, 25, 8)

#         # Process uploaded image
#         pil_image = Image.open(uploaded_file).convert("RGB")
#         img_height, img_width = pil_image.size
#         canvas_result = st_canvas(
#             fill_color="rgb(255, 255, 255)",
#             stroke_width=stroke_width,
#             stroke_color="#FFF",
#             background_color="#000",
#             background_image=pil_image,
#             update_streamlit=True,
#             drawing_mode=drawing_mode,
#             height=768,
#             width=768,
#             key="inpainting_canvas",
#         )

#         with col2:
#             submit = st.button("Generate")

#         if canvas_result.image_data is not None and len(canvas_result.json_data["objects"]) > 0 and submit:
#             mask_npy = canvas_result.image_data[:, :, 3]
#             mask_pil = Image.fromarray(mask_npy).convert("RGB")
#             mask_pil = mask_pil.resize((img_width, img_height), resample=Image.LANCZOS)
#             with st.spinner("Generating..."):
#                 output_images, metadata = inpainting_model.generate_image(
#                     prompt=prompt,
#                     negative_prompt=negative_prompt,
#                     image=pil_image,
#                     mask=mask_pil,
#                     guidance_scale=7.5,  # Adjust as needed
#                     scheduler="EulerAncestralDiscreteScheduler",  # Adjust as needed
#                     steps=50,  # Adjust as needed
#                     seed=42,  # Adjust as needed
#                     height=img_height,
#                     width=img_width,
#                     num_images=1,  # Adjust as needed
#                 )

#             st.image(output_images[0], caption='Generated Image', use_column_width=True)

# if __name__ == "__main__":
#     main()
    


import json
import random
from dataclasses import dataclass
from io import BytesIO
from typing import Optional


import requests
import streamlit as st
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from streamlit_drawable_canvas import st_canvas
import diffusers

@dataclass
class Inpainting:
    model: Optional[str] = None
    device: Optional[str] = None
    output_path: Optional[str] = None

    compatible_schedulers = [
        diffusers.EulerAncestralDiscreteScheduler,  # Add other compatible schedulers if needed
    ]
    scheduler_config = {
        "name": compatible_schedulers[0],
        # Add scheduler specific configuration if needed
    }

    def __str__(self) -> str:
        return f"Inpainting(model={self.model}, device={self.device}, output_path={self.output_path})"

    def __init__(self, model="diffusers/stable-diffusion-xl-1.0-inpainting-0.1", device="cuda"):
        self.model = model
        self.device = device

        self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            self.model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            use_auth_token=self.get_auth_token(),  # Replace utils.use_auth_token()
        )

        self.pipeline.to(self.device)
        self.pipeline.safety_checker = self.no_safety_checker  # Replace utils.no_safety_checker

    def get_auth_token(self) -> Optional[str]:
        # Implement logic to retrieve auth token if needed (previously from utils)
        # This might involve environment variables, user input, or external services
        # For simplicity, we'll assume no token is required in this example
        return None

    def no_safety_checker(self, images, **kwargs):
        # Implement a no-op safety checker (previously from utils)
        return images

    def _set_scheduler(self, scheduler_name):
        scheduler = self.compatible_schedulers[scheduler_name].from_config(self.scheduler_config)
        self.pipeline.scheduler = scheduler

    def generate_image(
        self,
        prompt,
        negative_prompt,
        image,
        mask,
        guidance_scale,
        scheduler,
        steps,
        seed,
        height,
        width,
        num_images,
    ):

        if seed == -1:
            # generate random seed
            seed = random.randint(0, 999999)

        self._set_scheduler(scheduler)

        if self.device == "mps":
            generator = torch.manual_seed(seed)
            num_images = 1
        else:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        output_images = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
            generator=generator,
            height=height,
            width=width,
        ).images

        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "scheduler": scheduler,
            "steps": steps,
            "seed": seed,
        }
        metadata_json = json.dumps(metadata)

        # Implement metadata saving logic without using utils.save_images()
        with open(f"{self.output_path}/inpainting.png", "wb") as f:
            output_images[0].save(f, format="PNG")
            png_info = PngInfo()
            png_info.add_text("inpainting", metadata_json)
            f.seek(0)
            f.write(png_info.tobytes())

        torch.cuda.empty_cache()
        return output_images, metadata_json


def main():
    st.title("Inpainting App")

    # Create an instance of the Inpainting class
    inpainting_model = Inpainting(model="diffusers/stable-diffusion-xl-1.0-inpainting-0.1", device="cuda")

    # Streamlit UI elements
    prompt = st.text_area("Prompt", "", help="Prompt for the image generation")
    negative_prompt = st.text_area("Negative Prompt", "", help="The prompt not to guide image generation")
    uploaded_file = st.file_uploader("Image:", type=["png", "jpg", "jpeg"], help="Image size must match model's image size. Usually: 512 or 768")

    if uploaded_file is not None:
        # Streamlit Canvas
        col1, col2 = st.columns(2)
        with col1:
            drawing_mode = st.selectbox("Drawing tool:", ("freedraw", "rect", "circle"))
            stroke_width = st.slider("Stroke width: ", 1, 25, 8)

        # Process uploaded image
        pil_image = Image.open(uploaded_file).convert("RGB")
        img_height, img_width = pil_image.size
        canvas_result = st_canvas(
            fill_color="rgb(255, 255, 255)",
            stroke_width=stroke_width,
            stroke_color="#FFF",
            background_color="#000",
            background_image=pil_image,
            update_streamlit=True,
            drawing_mode=drawing_mode,
            height=768,
            width=768,
            key="inpainting_canvas",
        )

        with col2:
            submit = st.button("Generate")

        if canvas_result.image_data is not None and len(canvas_result.json_data["objects"]) > 0 and submit:
            mask_npy = canvas_result.image_data[:, :, 3]
            mask_pil = Image.fromarray(mask_npy).convert("RGB")
            mask_pil = mask_pil.resize((img_width, img_height), resample=Image.LANCZOS)
            with st.spinner("Generating..."):
                output_images, metadata = inpainting_model.generate_image(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=pil_image,
                    mask=mask_pil,
                    guidance_scale=7.5,  # Adjust as needed
                    scheduler="EulerAncestralDiscreteScheduler",  # Adjust as needed
                    steps=50,  # Adjust as needed
                    seed=42,  # Adjust as needed
                    height=img_height,
                    width=img_width,
                    num_images=1,  # Adjust as needed
                )

            st.image(output_images[0], caption='Generated Image', use_column_width=True)

if __name__ == "__main__":
    main()
    
