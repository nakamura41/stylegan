# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for reproducing the figures of the StyleGAN paper using pre-trained generators."""

import os
import pickle

import moviepy.editor
import numpy as np
import scipy

import config
import dnnlib.tflib as tflib

# ----------------------------------------------------------------------------
# Helpers for loading and using pre-trained generators.

anime_faces_256_url = 'networks/weights/sgan-anime-faces-256-network-snapshot-013000.pkl'

synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)

_Gs_cache = dict()

grid_size = [1, 1]
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


def animate_latents(seed_latents, num_frames):
    inc = 0.0001
    all_latents = []
    for frame in range(num_frames):
        new_latents = seed_latents
        new_latents[0] = new_latents[0] - inc
        all_latents.append(new_latents)
        seed_latents = new_latents
    return np.array(all_latents)


def gen_animated_faces(Gs):
    num_frames = int(np.rint(duration_sec * mp4_fps))
    random_state = np.random.RandomState(random_seed)

    # Generate latent vectors
    first_latents = random_state.randn(*Gs.input_shape[1:]).astype(np.float32)
    all_latents = animate_latents(first_latents, num_frames)
    all_latents = all_latents.reshape((num_frames, 1, Gs.input_shape[1:][0]))

    # Frame generation func for moviepy.
    def make_frame(t):
        frame_idx = int(np.clip(np.round(t * mp4_fps), 0, num_frames - 1))
        latents = all_latents[frame_idx]
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = Gs.run(latents, None, truncation_psi=0.5,
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


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
