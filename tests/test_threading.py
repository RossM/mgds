import os
import sys
import time

sys.path.append(os.getcwd())

from src.mgds.DebugDataLoaderModules import SaveImage, DecodeVAE
from src.mgds.DiffusersDataLoaderModules import *
from src.mgds.GenericDataLoaderModules import *
from src.mgds.MGDS import MGDS, TrainDataLoader, OutputPipelineModule
from src.mgds.TransformersDataLoaderModules import *

DEVICE = 'cuda'
DTYPE = torch.float32
BATCH_SIZE = 4
NUM_WORKERS = 4


def test():
    depth_model_path = '..\\models\\diffusers-base\\sd-v2-0-depth'

    vae = AutoencoderKL.from_pretrained(os.path.join(depth_model_path, 'vae')).to(DEVICE)
    image_depth_processor = DPTImageProcessor.from_pretrained(os.path.join(depth_model_path, 'feature_extractor'))
    depth_estimator = DPTForDepthEstimation.from_pretrained(os.path.join(depth_model_path, 'depth_estimator')).to(DEVICE)
    tokenizer = CLIPTokenizer.from_pretrained(os.path.join(depth_model_path, 'tokenizer'))

    input_modules = [
        CollectPaths(concept_in_name='concept', path_in_name='path', name_in_name='name', path_out_name='image_path', concept_out_name='concept', extensions=['.png', '.jpg'], include_postfix=None, exclude_postfix=['-masklabel'], include_subdirectories_in_name='concept.include_subdirectories'),
        ModifyPath(in_name='image_path', out_name='mask_path', postfix='-masklabel', extension='.png'),
        ModifyPath(in_name='image_path', out_name='prompt_path', postfix='', extension='.txt'),
        LoadImage(path_in_name='image_path', image_out_name='image', range_min=-1.0, range_max=1.0),
        GenerateImageLike(image_in_name='image', image_out_name='mask', color=255, range_min=0, range_max=1, channels=1),
        LoadImage(path_in_name='mask_path', image_out_name='mask', range_min=0, range_max=1, channels=1),
        GenerateDepth(path_in_name='image_path', image_out_name='depth', image_depth_processor=image_depth_processor, depth_estimator=depth_estimator),
        RandomCircularMaskShrink(mask_name='mask', shrink_probability=1.0, shrink_factor_min=0.2, shrink_factor_max=1.0, enabled_in_name='concept.random_circular_crop'),
        RandomMaskRotateCrop(mask_name='mask', additional_names=['image', 'depth'], min_size=512, min_padding_percent=10, max_padding_percent=30, max_rotate_angle=20, enabled_in_name='concept.random_mask_rotate_crop'),
        CalcAspect(image_in_name='image', resolution_out_name='original_resolution'),
        AspectBucketing(target_resolution=512, resolution_in_name='original_resolution', scale_resolution_out_name='scale_resolution', crop_resolution_out_name='crop_resolution', possible_resolutions_out_name='possible_resolutions', quantization=8),
        ScaleCropImage(image_in_name='image', scale_resolution_in_name='scale_resolution', crop_resolution_in_name='crop_resolution', enable_crop_jitter_in_name='concept.enable_crop_jitter', image_out_name='image', crop_offset_out_name='crop_offset'),
        ScaleCropImage(image_in_name='mask', scale_resolution_in_name='scale_resolution', crop_resolution_in_name='crop_resolution', enable_crop_jitter_in_name='concept.enable_crop_jitter', image_out_name='mask', crop_offset_out_name='crop_offset'),
        ScaleCropImage(image_in_name='depth', scale_resolution_in_name='scale_resolution', crop_resolution_in_name='crop_resolution', enable_crop_jitter_in_name='concept.enable_crop_jitter', image_out_name='depth', crop_offset_out_name='crop_offset'),
        LoadText(path_in_name='prompt_path', text_out_name='prompt'),
        GenerateMaskedConditioningImage(image_in_name='image', mask_in_name='mask', image_out_name='conditioning_image', image_range_min=0, image_range_max=1),
        RandomFlip(names=['image', 'mask', 'depth', 'conditioning_image'], enabled_in_name='concept.random_flip'),
        EncodeVAE(in_name='image', out_name='latent_image_distribution', vae=vae),
        Downscale(in_name='mask', out_name='latent_mask', factor=8),
        EncodeVAE(in_name='conditioning_image', out_name='latent_conditioning_image_distribution', vae=vae),
        Downscale(in_name='depth', out_name='latent_depth', factor=8),
        Tokenize(in_name='prompt', tokens_out_name='tokens', mask_out_name='tokens_mask', max_token_length=77, tokenizer=tokenizer),
        DiskCache(cache_dir='cache', split_names=['latent_image_distribution', 'latent_mask', 'latent_conditioning_image_distribution', 'latent_depth', 'tokens'], aggregate_names=['crop_resolution']),
        SampleVAEDistribution(in_name='latent_image_distribution', out_name='latent_image', mode='mean'),
        SampleVAEDistribution(in_name='latent_conditioning_image_distribution', out_name='latent_conditioning_image', mode='mean'),
        RandomLatentMaskRemove(latent_mask_name='latent_mask', latent_conditioning_image_name='latent_conditioning_image', replace_probability=0.1, vae=vae, possible_resolutions_in_name='possible_resolutions')
    ]

    debug_modules = [
        DecodeVAE(in_name='latent_image', out_name='decoded_image', vae=vae),
        DecodeVAE(in_name='latent_conditioning_image', out_name='decoded_conditioning_image', vae=vae),
        SaveImage(image_in_name='decoded_image', original_path_in_name='image_path', path='debug', in_range_min=-1, in_range_max=1),
        SaveImage(image_in_name='mask', original_path_in_name='image_path', path='debug', in_range_min=0, in_range_max=1),
        SaveImage(image_in_name='decoded_conditioning_image', original_path_in_name='image_path', path='debug', in_range_min=-1, in_range_max=1),
        # SaveImage(image_in_name='depth', original_path_in_name='image_path', path='debug', in_range_min=-1, in_range_max=1),
        # SaveImage(image_in_name='latent_mask', original_path_in_name='image_path', path='debug', in_range_min=0, in_range_max=1),
        # SaveImage(image_in_name='latent_depth', original_path_in_name='image_path', path='debug', in_range_min=-1, in_range_max=1),
    ]

    output_modules = [
        AspectBatchSorting(resolution_in_name='crop_resolution', names=['latent_image', 'latent_conditioning_image', 'latent_mask', 'latent_depth', 'tokens'], batch_size=BATCH_SIZE, sort_resolutions_for_each_epoch=True),
        OutputPipelineModule(names=['latent_image', 'latent_conditioning_image', 'latent_mask', 'latent_depth', 'tokens'])
    ]

    #threading_test()

    ds = MGDS(
        device=torch.device(DEVICE),
        dtype=DTYPE,
        allow_mixed_precision=False,
        concepts=[
            {
                'name': 'DS',
                'path': '..\\datasets\\dataset2-100',
                'random_circular_crop': True,
                'random_mask_rotate_crop': True,
                'random_flip': True,
                'include_subdirectories': True,
            },
        ],
        settings={},
        definition=[
            input_modules,
            #debug_modules,
            output_modules
        ],
        batch_size=BATCH_SIZE,
        seed=42,
        initial_epoch=0,
        initial_epoch_sample=10,
        num_workers=NUM_WORKERS,
    )
    dl = TrainDataLoader(
        ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )


    # for epoch in range(10):
    #     ds.start_next_epoch()
    #     for batch in tqdm(dl):
    #         pass


def threading_test():
    length = 100
    current_progress = 0
    in_lock = threading.Lock()
    out_lock = threading.Lock()
    progress_bar = tqdm(total=length)

    def __cache_one_item(index: int):
        time.sleep(1)
        with out_lock:
            print(index)
            progress_bar.update(1)

    def __worker():
        nonlocal current_progress
        stopped = False

        while not stopped:
            with in_lock:
                index = current_progress
                current_progress += 1

            if index >= length:
                stopped = True
            else:
                __cache_one_item(index)

    with ThreadPoolExecutor(max_workers=10) as pool:
        for i in range(10):
            pool.submit(__worker)

if __name__ == '__main__':
    test()
