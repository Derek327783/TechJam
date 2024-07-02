from diffusers import StableDiffusionPipeline
import torch

class _Image_generation_service():
    instance = None
    def get_image(self,prompt):
        pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
        pipeline.load_lora_weights("../Weights/pytorch_lora_weights.safetensors", weight_name="pytorch_lora_weights.safetensors")
        image = pipeline(prompt)
        return image

def Image_generation_service():
    if _Image_generation_service.instance == None:
        _Image_generation_service.instance = _Image_generation_service()
    return _Image_generation_service.instance

# if __name__ == "__main__":
#     image_service = Image_generation_service()
#     gen_image = image_service.get_image("A calm piece of music")
#     print(gen_image)