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
from utils import load_image, load_json, mkdir, resize, save_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("id")
    parser.add_argument("--bip2017", default="resources/BIP2017")
    parser.add_argument("--save_folder", default="results")
    args = parser.parse_args()

    IMAGE_SIZE = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mesh = ico_sphere(3, device=device)
    verts, faces = mesh.get_mesh_verts_faces(0)
    # ambient color in BIP2017, see resources/BIP2017/sphere/sphereAmbient.png
    verts_rgb = np.repeat([[184, 131, 105]], len(verts), axis=0) / 255.
    textures = Textures(verts_rgb=[torch.Tensor(verts_rgb).to(device)])
    meshes = Meshes(verts=[verts], faces=[faces], textures=textures)

    print(f"vertices: {meshes.verts_list()[0].shape}")
    print(f"faces: {meshes.faces_list()[0].shape}")

    # Render the mesh
    ## Create Camera
    R, T = look_at_view_transform(dist=2.7, elev=0, azim=0)
    cameras = FoVOrthographicCameras(device=device, R=R, T=T)

    sh_id = args.id
    dataroot = Path(args.bip2017)
    info = load_json(dataroot / "parameters" / f"face_{sh_id}.rps")
    sh_params = info["environmentMap"]["coefficients"]
    sh_params = torch.Tensor(sh_params).unsqueeze(0)
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

    gt = load_image(dataroot / "renderings" / f"face_{sh_id}.png")
    gt = resize(gt, (IMAGE_SIZE, IMAGE_SIZE))

    save_folder = mkdir(args.save_folder)
    save_image(save_folder / f"sphere_{sh_id}.jpg", np.concatenate([vis, gt], axis=1))

    print("Save results to", save_folder / f"sphere_{sh_id}.jpg")

if __name__ == "__main__":
    main()
