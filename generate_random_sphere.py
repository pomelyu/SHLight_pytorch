import argparse
from pathlib import Path

import numpy as np
import torch
from pytorch3d.renderer import (FoVOrthographicCameras, MeshRasterizer,
                                MeshRenderer, RasterizationSettings,
                                SoftPhongShader, Textures,
                                look_at_view_transform)
from pytorch3d.renderer.blending import BlendParams
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere

from sh_lights import SphericalHarmonicsLights
from utils import mkdir, save_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_folder", default="results")
    args = parser.parse_args()

    IMAGE_SIZE = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mesh = ico_sphere(3, device=device)
    verts, faces = mesh.get_mesh_verts_faces(0)
    textures = Textures(verts_rgb=[torch.ones_like(verts)])
    meshes = Meshes(verts=[verts], faces=[faces], textures=textures)

    print(f"vertices: {meshes.verts_list()[0].shape}")
    print(f"faces: {meshes.faces_list()[0].shape}")

    # Render the mesh
    ## Create Camera
    R, T = look_at_view_transform(dist=2.7, elev=0, azim=0)
    cameras = FoVOrthographicCameras(device=device, R=R, T=T)

    sh_params = torch.rand(1, 9, 3, device=device)
    print(f"sh_params: {sh_params[0].shape}")

    ## Create Light
    lights = SphericalHarmonicsLights(device=device, sh_params=sh_params)

    ## Setup render
    rasterize_settings = RasterizationSettings(
        image_size=IMAGE_SIZE,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=rasterize_settings,
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
            blend_params=BlendParams(background_color=(0., 0., 0.))
        ),
    )

    result = renderer(meshes.extend(len(cameras)))
    result = np.clip(result.cpu().numpy(), 0, 1)
    result = np.concatenate(result, axis=1)
    result, alpha = result[..., :-1], result[..., -1:]
    alpha = np.repeat(alpha, 3, axis=-1)

    vis = np.concatenate([result, alpha], axis=1)
    vis = (vis * 255).astype(np.uint8)

    save_folder = mkdir(args.save_folder)
    save_image(save_folder / f"sphere_random.jpg", vis)


if __name__ == "__main__":
    main()
