import os
import shutil

import cv2
import numpy as np
import torch
from typing_extensions import override

import folder_paths
import nodes
from comfy_api.latest import ComfyExtension, io

from . import lav, rig
from .segment import segment

WEB_DIRECTORY = "web"
PACK_DIR = os.path.dirname(os.path.abspath(__file__))
AvatarSegments = io.Custom("LIVE_AVATAR_SEGMENTS")
Avatar = io.Custom("LIVE_AVATAR")
SeeThroughParts = io.Custom("SEETHROUGH_PARTS")

def segment_preview(seg):
    label, names, alpha = rig.partition(seg)
    colors = np.random.default_rng(7).integers(50, 255, (len(names), 3))
    out = seg["image"].copy() * 0.35
    for i in range(len(names)):
        out[label == i] += colors[i] / 255 * 0.65
    return torch.from_numpy(out.clip(0, 1).astype(np.float32))[None]


class LiveAvatarSegment(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LiveAvatarSegment",
            display_name="Live Avatar Segment",
            category="Live Avatar",
            description="Splits a character into body and face parts with WeVisOne-VL.",
            inputs=[
                io.Clip.Input("clip", tooltip="WeVisOne-VL from the WeVisOne-VL Loader."),
                io.Image.Input("image"),
                io.Mask.Input("alpha", optional=True, tooltip="Character matte. Without it WeVisOne's person matte is used."),
            ],
            outputs=[AvatarSegments.Output(display_name="segments"), io.Image.Output(display_name="preview")],
        )

    @classmethod
    def execute(cls, clip, image, alpha=None) -> io.NodeOutput:
        seg = segment(nodes.NODE_CLASS_MAPPINGS["WeVisOneVLPerceive"], clip, image[:1, ..., :3], alpha)
        return io.NodeOutput(seg, segment_preview(seg))


class LiveAvatarHeadMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LiveAvatarHeadMask",
            display_name="Live Avatar Head Mask",
            category="Live Avatar",
            description="Matte of the head alone, for reconstructing it in 3D.",
            inputs=[AvatarSegments.Input("segments")],
            outputs=[io.Mask.Output(display_name="mask")],
        )

    @classmethod
    def execute(cls, segments) -> io.NodeOutput:
        return io.NodeOutput(torch.from_numpy(rig.head_mask(segments))[None])


class LiveAvatarPlate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LiveAvatarPlate",
            display_name="Live Avatar Plate",
            category="Live Avatar",
            description="Prepares the body hidden behind arms and hair for an inpaint pass.",
            inputs=[AvatarSegments.Input("segments")],
            outputs=[io.Image.Output(display_name="image"), io.Mask.Output(display_name="mask")],
        )

    @classmethod
    def execute(cls, segments) -> io.NodeOutput:
        image, mask = rig.plate_request(segments)
        return io.NodeOutput(torch.from_numpy(image.astype(np.float32))[None], torch.from_numpy(mask.astype(np.float32))[None])


class LiveAvatarBuild(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LiveAvatarBuild",
            display_name="Live Avatar Build",
            category="Live Avatar",
            description="Builds layers, meshes, bones, morphs and clips from the segments.",
            inputs=[
                AvatarSegments.Input("segments"),
                SeeThroughParts.Input("seethrough_parts", optional=True, tooltip="Hidden pixels and depth from See-through. Without it hidden pixels are extended from the visible ones."),
                io.Image.Input("depth", optional=True, tooltip="Depth map, near is white. Adds relief to the layers and replaces See-through's depth."),
                io.Image.Input("plate", optional=True, tooltip="Inpainted image from Live Avatar Plate. Paints the body hidden behind arms and hair."),
                io.Splat.Input("head_splat", optional=True, tooltip="Splat of the head alone. Takes over from the layers in wide head turns."),
                io.Boolean.Input("splat_hair", default=False, tooltip="Draw the hair with the splat at every angle: no hair transition, but softer than the drawing when facing front."),
            ],
            outputs=[Avatar.Output(display_name="avatar")],
        )

    @classmethod
    def execute(cls, segments, splat_hair=False, seethrough_parts=None, depth=None, plate=None, head_splat=None) -> io.NodeOutput:
        hidden = None
        height, width = segments["image"].shape[:2]
        if depth is not None:
            depth = cv2.resize(depth[0, ..., 0].cpu().numpy(), (width, height), interpolation=cv2.INTER_AREA)
        if seethrough_parts is not None:
            hidden = rig.seethrough_hidden(seethrough_parts, width, height)
            if depth is None:
                depth = rig.seethrough_depth(seethrough_parts, width, height)
        if plate is not None:
            plate = (plate[0, ..., :3].cpu().numpy() * 255).round().astype(np.uint8)
        if head_splat is not None:
            head_splat = tuple(t[0].float().cpu().numpy() for t in (head_splat.positions, head_splat.scales, head_splat.rotations, head_splat.opacities, head_splat.sh))
        return io.NodeOutput(rig.build_avatar(segments, hidden, depth, plate, head_splat, splat_hair))


class LiveAvatarSave(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LiveAvatarSave",
            display_name="Live Avatar Save",
            category="Live Avatar",
            description="Saves a .lav file for the three.js runtime and shows it in a live viewer.",
            is_output_node=True,
            inputs=[Avatar.Input("avatar"), io.String.Input("filename_prefix", default="live_avatar/avatar")],
            outputs=[],
        )

    @classmethod
    def execute(cls, avatar, filename_prefix) -> io.NodeOutput:
        folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())
        name = f"{filename}_{counter:05}_.lav"
        with open(os.path.join(folder, name), "wb") as f:
            f.write(lav.encode(avatar))
        return io.NodeOutput(ui={"live_avatar": [{"filename": name, "subfolder": subfolder, "type": "output"}]})


class LiveAvatarExport(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LiveAvatarExport",
            display_name="Live Avatar Export",
            category="Live Avatar",
            description="Writes a self-contained web package: the avatar, its runtime and a page for apps and OBS.",
            is_output_node=True,
            inputs=[
                Avatar.Input("avatar"),
                io.String.Input("filename_prefix", default="live_avatar_export/avatar"),
                io.Boolean.Input("webcam_tracking", default=True, tooltip="Include MediaPipe for webcam tracking. Adds about 18 MB."),
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, avatar, filename_prefix, webcam_tracking) -> io.NodeOutput:
        folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())
        name = f"{filename}_{counter:05}_"
        package = os.path.join(folder, name)
        web = os.path.join(PACK_DIR, "web")
        for part in ("runtime", "three", "viewer") + (("mediapipe",) if webcam_tracking else ()):
            shutil.copytree(os.path.join(web, part), os.path.join(package, part))
        template = os.path.join(PACK_DIR, "export_template")
        for part in ("serve.py", "README.txt"):
            shutil.copy(os.path.join(template, part), package)
        with open(os.path.join(template, "index.html")) as f:
            page = f.read().replace("__NO_TRACKING__", "false" if webcam_tracking else "true")
        with open(os.path.join(package, "index.html"), "w") as f:
            f.write(page)
        with open(os.path.join(package, "avatar.lav"), "wb") as f:
            f.write(lav.encode(avatar))
        shutil.make_archive(package, "zip", package)
        return io.NodeOutput(ui={"live_avatar_export": [{"filename": name + ".zip", "subfolder": subfolder, "type": "output"}]})


class LiveAvatarExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [LiveAvatarSegment, LiveAvatarHeadMask, LiveAvatarPlate, LiveAvatarBuild, LiveAvatarSave, LiveAvatarExport]


async def comfy_entrypoint() -> LiveAvatarExtension:
    return LiveAvatarExtension()
