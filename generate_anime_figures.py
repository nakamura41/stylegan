# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for reproducing the figures of the StyleGAN paper using pre-trained generators."""

import os
import pickle
import scipy
import moviepy.editor
import numpy as np

import config
import dnnlib.tflib as tflib

# ----------------------------------------------------------------------------
# Helpers for loading and using pre-trained generators.

anime_faces_256_url = 'networks/weights/sgan-anime-faces-256-network-snapshot-008126.pkl'

synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)

_Gs_cache = dict()

grid_size = [5, 3]
image_shrink = 1
image_zoom = 1
duration_sec = 30.0
smoothing_sec = 1.0
mp4_fps = 60
mp4_codec = 'libx264'
mp4_bitrate = '16M'
random_seed = 404
mp4_file = 'results/random_grid_%s.mp4' % random_seed
minibatch_size = 8


def load_Gs(url):
    if url not in _Gs_cache:
        with open(url, 'rb') as f:
            _G, _D, Gs = pickle.load(f)
        _Gs_cache[url] = Gs
    return _Gs_cache[url]


def gen_animated_faces(Gs):
    num_frames = int(np.rint(duration_sec * mp4_fps))
    random_state = np.random.RandomState(random_seed)

    # Generate latent vectors
    shape = [num_frames, np.prod(grid_size)] + Gs.input_shape[1:]  # [frame, image, channel, component]
    all_latents = random_state.randn(*shape).astype(np.float32)
    all_latents = scipy.ndimage.gaussian_filter(all_latents, [smoothing_sec * mp4_fps] + [0] * len(Gs.input_shape),
                                                mode='wrap')
    all_latents /= np.sqrt(np.mean(np.square(all_latents)))

    # Frame generation func for moviepy.
    def make_frame(t):
        frame_idx = int(np.clip(np.round(t * mp4_fps), 0, num_frames - 1))
        latents = all_latents[frame_idx]
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = Gs.run(latents, None, truncation_psi=0.7,
                        randomize_noise=False, output_transform=fmt)

        grid = create_image_grid(images, grid_size)
        if image_zoom > 1:
            grid = scipy.ndimage.zoom(grid, [image_zoom, image_zoom, 1], order=0)
        if grid.shape[2] == 1:
            grid = grid.repeat(3, 2)  # grayscale => RGB
        return grid

    video_clip = moviepy.editor.VideoClip(make_frame, duration=duration_sec)
    video_clip.write_videofile(mp4_file, fps=mp4_fps, codec=mp4_codec, bitrate=mp4_bitrate)


def create_image_grid(images, grid_size=None):
    assert images.ndim == 3 or images.ndim == 4
    num, img_h, img_w, channels = images.shape

    if grid_size is not None:
        grid_w, grid_h = tuple(grid_size)
    else:
        grid_w = max(int(np.ceil(np.sqrt(num))), 1)
        grid_h = max((num - 1) // grid_w + 1, 1)

    grid = np.zeros([grid_h * img_h, grid_w * img_w, channels], dtype=images.dtype)
    for idx in range(num):
        x = (idx % grid_w) * img_w
        y = (idx // grid_w) * img_h
        grid[y: y + img_h, x: x + img_w] = images[idx]
    return grid


# ----------------------------------------------------------------------------
# Main program.

def main():
    tflib.init_tf()
    os.makedirs(config.result_dir, exist_ok=True)
    gen_animated_faces(load_Gs(anime_faces_256_url))
    # draw_uncurated_result_figure(os.path.join(config.result_dir, 'figure02-uncurated-anime-faces-256.png'), load_Gs(anime_faces_256_url), cx=0, cy=0, cw=256, ch=256, rows=3, lods=[0, 1, 2, 2, 3, 3], seed=5)
    # draw_style_mixing_figure(os.path.join(config.result_dir, 'figure03-style-mixing.png'), load_Gs(url_ffhq), w=1024, h=1024, src_seeds=[639,701,687,615,2268], dst_seeds=[888,829,1898,1733,1614,845], style_ranges=[range(0,4)]*3+[range(4,8)]*2+[range(8,18)])
    # draw_noise_detail_figure(os.path.join(config.result_dir, 'figure04-noise-detail.png'), load_Gs(url_ffhq), w=1024, h=1024, num_samples=100, seeds=[1157,1012])
    # draw_noise_components_figure(os.path.join(config.result_dir, 'figure05-noise-components.png'), load_Gs(url_ffhq), w=1024, h=1024, seeds=[1967,1555], noise_ranges=[range(0, 18), range(0, 0), range(8, 18), range(0, 8)], flips=[1])
    # draw_truncation_trick_figure(os.path.join(config.result_dir, 'figure08-truncation-trick.png'), load_Gs(url_ffhq), w=1024, h=1024, seeds=[91,388], psis=[1, 0.7, 0.5, 0, -0.5, -1])
    # draw_uncurated_result_figure(os.path.join(config.result_dir, 'figure10-uncurated-bedrooms.png'), load_Gs(url_bedrooms), cx=0, cy=0, cw=256, ch=256, rows=5, lods=[0,0,1,1,2,2,2], seed=0)
    # draw_uncurated_result_figure(os.path.join(config.result_dir, 'figure11-uncurated-cars.png'), load_Gs(url_cars), cx=0, cy=64, cw=512, ch=384, rows=4, lods=[0,1,2,2,3,3], seed=2)
    # draw_uncurated_result_figure(os.path.join(config.result_dir, 'figure12-uncurated-cats.png'), load_Gs(url_cats), cx=0, cy=0, cw=256, ch=256, rows=5, lods=[0,0,1,1,2,2,2], seed=1)


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
