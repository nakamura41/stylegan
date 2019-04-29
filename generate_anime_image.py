# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for reproducing the figures of the StyleGAN paper using pre-trained generators."""

import argparse
import copy
import os
import pickle

import numpy as np
import scipy

import config
import dnnlib.tflib as tflib
from PIL import Image

# ----------------------------------------------------------------------------
# Helpers for loading and using pre-trained generators.

anime_faces_256_url = 'networks/weights/sgan-anime-faces-256-network-snapshot-013000.pkl'

synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)

_Gs_cache = dict()
IMAGE_ZOOM = 2


def load_Gs(url):
    if url not in _Gs_cache:
        with open(url, 'rb') as f:
            _G, _D, Gs = pickle.load(f)
        _Gs_cache[url] = Gs
    return _Gs_cache[url]


def zoom_image(image, zoom):
    return scipy.ndimage.zoom(image, [zoom, zoom, 1], order=0)


def generate_latents(Gs, random_seed):
    random_state = np.random.RandomState(random_seed)
    return random_state.randn(*Gs.input_shape[1:]).astype(np.float32)


def gen_face(Gs, latents, random_seed, latent_modifier=0, latent_modifier_position=-1):
    if latent_modifier != 0 and latent_modifier_position >= 0:
        print("latent_modifier: {}".format(latent_modifier))
        print("latent_modifier_position: {}".format(latent_modifier_position))
        latents[latent_modifier_position] = latents[latent_modifier_position] + latent_modifier
        filename = "output/images/image_{}_{}_{}.png".format(random_seed, latent_modifier, latent_modifier_position)
    else:
        filename = "output/images/image_{}.png".format(random_seed)

    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    latents = latents.reshape((1, latents.shape[0]))
    image = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=False, output_transform=fmt)[0]
    image = zoom_image(image, IMAGE_ZOOM)
    img = Image.fromarray(image, 'RGB')
    img.save(filename)
    print("Image is saved to {}".format(filename))


def load_model():
    os.makedirs(config.result_dir, exist_ok=True)
    return load_Gs(anime_faces_256_url)


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    tflib.init_tf()

    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-latents", "-gl", type=int, help="Generate Latents")
    parser.add_argument("--use-latents", "-ul", type=int, help="Use existing Latents")
    parser.add_argument("--latent-modifier", "-lm", type=float, help="Latent modifier", default=0)
    parser.add_argument("--latent-modifier-position", "-lp", type=int, help="Latent modifier position", default=-1)
    args = parser.parse_args()

    if args.generate_latents:
        random_seed = args.generate_latents
        print("Generate latents, random seed={}".format(random_seed))
        Gs = load_model()
        latents = generate_latents(Gs, random_seed)
        filename = "output/latents/latents_{}".format(random_seed)
        np.save(filename, latents)
        print("latents is saved on {}".format(filename))
    if args.use_latents:
        print("Use existing latents")
        random_seed = args.use_latents
        filename = "output/latents/latents_{}.npy".format(random_seed)
        latents = np.load(filename)
        Gs = load_model()
        gen_face(Gs, latents, random_seed, args.latent_modifier, args.latent_modifier_position)

# ----------------------------------------------------------------------------
