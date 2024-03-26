import argparse
import logging
import os
import os.path as osp
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.checkpoint
from torchvision import transforms
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from PIL import Image
from transformers import CLIPVisionModelWithProjection

from models.unet_2d_condition import UNet2DConditionModel
from models.unet_3d import UNet3DConditionModel
from models.mutual_self_attention import ReferenceAttentionControl
from models.guidance_encoder import GuidanceEncoder
from models.champ_model import ChampModel

from pipelines.pipeline_aggregation import MultiGuidance2LongVideoPipeline

from utils.video_utils import resize_tensor_frames, save_videos_grid, pil_list_to_tensor

GUIDANCE_TYPES=["depth", "normal", "semantic_map", "dwpose"]

def setup_savedir():
    time_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    savedir = f"results/exp-{time_str}"
    os.makedirs(savedir, exist_ok=True)
    return savedir


def setup_guidance_encoder():
    guidance_encoder_group = dict()

    for guidance_type in GUIDANCE_TYPES:
        guidance_encoder_group[guidance_type] = GuidanceEncoder(
            guidance_embedding_channels=320,
            guidance_input_channels=3,
            block_out_channels=[16, 32, 96, 256],
        ).to(device="cuda", dtype=torch.float16)

    return guidance_encoder_group


def process_semantic_map(semantic_map_path: Path):
    image_name = semantic_map_path.name
    mask_path = semantic_map_path.parent.parent / "mask" / image_name
    semantic_array = np.array(Image.open(semantic_map_path))
    mask_array = np.array(Image.open(mask_path).convert("RGB"))
    semantic_pil = Image.fromarray(np.where(mask_array > 0, semantic_array, 0))

    return semantic_pil

def combine_guidance_data(motion):
    guidance_data_folder = motion

    guidance_pil_group = dict()
    for guidance_type in GUIDANCE_TYPES:
        guidance_pil_group[guidance_type] = []
        for guidance_image_path in sorted(
            Path(osp.join(guidance_data_folder, guidance_type)).iterdir()
        ):
            # Add black background to semantic map
            if guidance_type == "semantic_map":
                guidance_pil_group[guidance_type] += [
                    process_semantic_map(guidance_image_path)
                ]
            else:
                guidance_pil_group[guidance_type] += [
                    Image.open(guidance_image_path).convert("RGB")
                ]

    # get video length from the first guidance sequence
    first_guidance_length = len(list(guidance_pil_group.values())[0])
    # ensure all guidance sequences are of equal length
    assert all(
        len(sublist) == first_guidance_length
        for sublist in list(guidance_pil_group.values())
    )

    return guidance_pil_group, first_guidance_length

def inference(
    vae,
    image_enc,
    model,
    scheduler,
    ref_image_pil,
    guidance_pil_group,
    video_length,
    width,
    height,
    seed=42,
    n_steps=20,
    guidance_scale=3.5,
    device="cuda",
    dtype=torch.float16,
):
    reference_unet = model.reference_unet
    denoising_unet = model.denoising_unet
    guidance_encoder_group = {
        f"guidance_encoder_{g}": getattr(model, f"guidance_encoder_{g}")
        for g in GUIDANCE_TYPES
    }

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    pipeline = MultiGuidance2LongVideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        **guidance_encoder_group,
        scheduler=scheduler,
    )
    pipeline = pipeline.to(device, dtype)

    video = pipeline(
        ref_image_pil,
        guidance_pil_group,
        width,
        height,
        video_length,
        num_inference_steps=n_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).videos

    del pipeline
    torch.cuda.empty_cache()

    return video

def main(
  image='example_data/ref_images/ref-01.png',
  motion='example_data/motions/motion-01',
  width=512,
  height=512,
  seed=42,
  n_steps=20,
  guidance_scale=3.5,
):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    save_dir = setup_savedir()
    logging.info(f"Running inference ...")

    # setup pretrained models
    weight_dtype = torch.float16

    noise_scheduler = DDIMScheduler(
      rescale_betas_zero_snr=True,
      timestep_spacing="trailing",
      prediction_type="v_prediction",
      num_train_timesteps=1000,
      beta_start=0.00085,
      beta_end=0.012,
      beta_schedule="linear",
      steps_offset=1,
      clip_sample=False,
    )

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        "pretrained_models/image_encoder",
    ).to(dtype=weight_dtype, device="cuda")

    vae = AutoencoderKL.from_pretrained("pretrained_models/sd-vae-ft-mse").to(
        dtype=weight_dtype, device="cuda"
    )

    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        "pretrained_models/stable-diffusion-v1-5",
        "pretrained_models/champ/motion_module.pth",
        subfolder="unet",
        unet_additional_kwargs={
          "use_inflated_groupnorm": True,
          "unet_use_cross_frame_attention": False,
          "unet_use_temporal_attention": False,
          "use_motion_module": True,
          "motion_module_resolutions": [1, 2, 4, 8],
          "motion_module_mid_block": True,
          "motion_module_decoder_only": False,
          "motion_module_type": "Vanilla",
          "motion_module_kwargs": {
            "num_attention_heads": 8,
            "num_transformer_block": 1,
            "attention_block_types": ["Temporal_Self", "Temporal_Self"],
            "temporal_position_encoding": True,
            "temporal_position_encoding_max_len": 32,
            "temporal_attention_dim_div": 1,
          },
        },
    ).to(dtype=weight_dtype, device="cuda")

    reference_unet = UNet2DConditionModel.from_pretrained(
        "pretrained_models/stable-diffusion-v1-5",
        subfolder="unet",
    ).to(device="cuda", dtype=weight_dtype)

    guidance_encoder_group = setup_guidance_encoder()

    ckpt_dir = "pretrained_models/champ"
    denoising_unet.load_state_dict(
        torch.load(
            osp.join(ckpt_dir, f"denoising_unet.pth"),
            map_location="cpu",
        ),
        strict=False,
    )
    reference_unet.load_state_dict(
        torch.load(
            osp.join(ckpt_dir, f"reference_unet.pth"),
            map_location="cpu",
        ),
        strict=False,
    )

    for guidance_type, guidance_encoder_module in guidance_encoder_group.items():
        guidance_encoder_module.load_state_dict(
            torch.load(
                osp.join(ckpt_dir, f"guidance_encoder_{guidance_type}.pth"),
                map_location="cpu",
            ),
            strict=False,
        )

    reference_control_writer = ReferenceAttentionControl(
        reference_unet,
        do_classifier_free_guidance=False,
        mode="write",
        fusion_blocks="full",
    )
    reference_control_reader = ReferenceAttentionControl(
        denoising_unet,
        do_classifier_free_guidance=False,
        mode="read",
        fusion_blocks="full",
    )

    model = ChampModel(
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        reference_control_writer=reference_control_writer,
        reference_control_reader=reference_control_reader,
        guidance_encoder_group=guidance_encoder_group,
    ).to("cuda", dtype=weight_dtype)

    if is_xformers_available():
        reference_unet.enable_xformers_memory_efficient_attention()
        denoising_unet.enable_xformers_memory_efficient_attention()
    else:
        raise ValueError(
            "xformers is not available. Make sure it is installed correctly"
        )

    ref_image_path = image
    ref_image_pil = Image.open(ref_image_path)
    ref_image_w, ref_image_h = ref_image_pil.size

    guidance_pil_group, video_length = combine_guidance_data()

    result_video_tensor = inference(
        vae=vae,
        image_enc=image_enc,
        model=model,
        scheduler=noise_scheduler,
        ref_image_pil=ref_image_pil,
        guidance_pil_group=guidance_pil_group,
        video_length=video_length,
        width=width,
        height=height,
        device="cuda",
        dtype=weight_dtype,
        seed=seed,
        n_steps=n_steps,
        guidance_scale=guidance_scale,
    )  # (1, c, f, h, w)

    result_video_tensor = resize_tensor_frames(
        result_video_tensor, (ref_image_h, ref_image_w)
    )
    anim = osp.join(save_dir, "animation.mp4")
    save_videos_grid(result_video_tensor, anim)

    ref_video_tensor = transforms.ToTensor()(ref_image_pil)[None, :, None, ...].repeat(
        1, 1, video_length, 1, 1
    )
    guidance_video_tensor_lst = []
    for guidance_pil_lst in guidance_pil_group.values():
        guidance_video_tensor_lst += [
            pil_list_to_tensor(guidance_pil_lst, size=(ref_image_h, ref_image_w))
        ]
    guidance_video_tensor = torch.stack(guidance_video_tensor_lst, dim=0)

    grid_video = torch.cat([ref_video_tensor, result_video_tensor], dim=0)
    grid_video_wguidance = torch.cat(
        [ref_video_tensor, result_video_tensor, guidance_video_tensor], dim=0
    )

    grid = osp.join(save_dir, "grid.mp4")
    save_videos_grid(grid_video, grid)
    grid_wguidance = osp.join(save_dir, "grid_wguidance.mp4")
    save_videos_grid(grid_video_wguidance, grid_wguidance)

    logging.info(f"Inference completed, results saved in {save_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/inference.yaml")
    args = parser.parse_args()

    main()
