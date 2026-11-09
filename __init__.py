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
from .motion import motion_clip
from .segment import segment
from .variants import VARIANTS, import_variant, variant_request

WEB_DIRECTORY = "web"
PACK_DIR = os.path.dirname(os.path.abspath(__file__))
AvatarSegments = io.Custom("LIVE_AVATAR_SEGMENTS")
Avatar = io.Custom("LIVE_AVATAR")
SeeThroughParts = io.Custom("SEETHROUGH_PARTS")
AvatarVariants = io.Custom("LIVE_AVATAR_VARIANTS")
MHRPoseData = io.Custom("MHR_POSE_DATA")
LAV_FOLDER = "live_avatar"


def lav_files():
    root = os.path.join(folder_paths.get_output_directory(), LAV_FOLDER)
    files = []
    for folder, _, names in os.walk(root):
        files += [os.path.relpath(os.path.join(folder, n), root).replace(os.sep, "/") for n in names if n.endswith(".lav")]
    return sorted(files)

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


class LiveAvatarVariantMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LiveAvatarVariantMask",
            display_name="Live Avatar Variant Mask",
            category="Live Avatar",
            description="Head crop and mask for inpainting one drawn variant: closed or smiling eyes, or a vowel mouth.",
            inputs=[AvatarSegments.Input("segments"), io.Combo.Input("variant", options=VARIANTS)],
            outputs=[io.Image.Output(display_name="image"), io.Mask.Output(display_name="mask")],
        )

    @classmethod
    def execute(cls, segments, variant) -> io.NodeOutput:
        image, mask = variant_request(segments, variant)
        return io.NodeOutput(torch.from_numpy(image.astype(np.float32))[None], torch.from_numpy(mask)[None])


class LiveAvatarVariants(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LiveAvatarVariants",
            display_name="Live Avatar Variants",
            category="Live Avatar",
            description="Collects inpainted variants for Live Avatar Build. Only each variant's masked area is used.",
            inputs=[AvatarSegments.Input("segments")] + [io.Image.Input(name, optional=True) for name in VARIANTS],
            outputs=[AvatarVariants.Output(display_name="variants")],
        )

    @classmethod
    def execute(cls, segments, **painted) -> io.NodeOutput:
        return io.NodeOutput({name: import_variant(segments, name, image[0, ..., :3].cpu().numpy()) for name, image in painted.items()})


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
                AvatarVariants.Input("variants", optional=True, tooltip="Drawn closed and smiling eyes and vowel mouths from Live Avatar Variants."),
                io.Boolean.Input("splat_hair", default=False, tooltip="Draw the hair with the splat at every angle: no hair transition, but softer than the drawing when facing front."),
            ],
            outputs=[Avatar.Output(display_name="avatar")],
        )

    @classmethod
    def execute(cls, segments, splat_hair=False, seethrough_parts=None, depth=None, plate=None, head_splat=None, variants=None) -> io.NodeOutput:
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
        return io.NodeOutput(rig.build_avatar(segments, hidden, depth, plate, head_splat, splat_hair, variants))


class LiveAvatarLoad(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LiveAvatarLoad",
            display_name="Live Avatar Load",
            category="Live Avatar",
            description="Loads a saved .lav from output/live_avatar, to add clips or export it again without rebuilding.",
            inputs=[io.Combo.Input("file", options=lav_files())],
            outputs=[Avatar.Output(display_name="avatar")],
        )

    @classmethod
    def execute(cls, file) -> io.NodeOutput:
        root = os.path.realpath(os.path.join(folder_paths.get_output_directory(), LAV_FOLDER))
        path = os.path.realpath(os.path.join(root, file))
        if os.path.commonpath([root, path]) != root or not path.endswith(".lav"):
            raise ValueError(f"Live Avatar: invalid file {file}")
        return io.NodeOutput(lav.load(path))

    @classmethod
    def fingerprint_inputs(cls, file):
        path = os.path.join(folder_paths.get_output_directory(), LAV_FOLDER, file)
        return os.path.getmtime(path) if os.path.isfile(path) else ""


class LiveAvatarMotion(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LiveAvatarMotion",
            display_name="Live Avatar Motion",
            category="Live Avatar",
            description="Adds a clip retargeted from captured body motion: MHR pose data from InstantHMR or SAM 3D Body.",
            inputs=[
                Avatar.Input("avatar"),
                MHRPoseData.Input("mhr_pose_data"),
                io.String.Input("clip_name", default="dance"),
                io.Float.Input("fps", default=30.0, min=1.0, max=240.0, step=0.01, tooltip="Frame rate of the video, for example from Get Video Components."),
                io.Boolean.Input("loop", default=False),
                io.Int.Input("person", default=0, min=0, max=63, advanced=True, tooltip="Which person of the pose data to follow."),
            ],
            outputs=[Avatar.Output(display_name="avatar")],
        )

    @classmethod
    def execute(cls, avatar, mhr_pose_data, clip_name, fps, loop, person=0) -> io.NodeOutput:
        if not clip_name.strip():
            raise ValueError("Live Avatar: the clip needs a name.")
        bones, params = lav.rig_of(avatar)
        return io.NodeOutput(lav.add_clip(avatar, clip_name.strip(), motion_clip(bones, params, mhr_pose_data, fps, loop, person)))


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
        return [LiveAvatarSegment, LiveAvatarHeadMask, LiveAvatarPlate, LiveAvatarVariantMask, LiveAvatarVariants, LiveAvatarBuild,
                LiveAvatarLoad, LiveAvatarMotion, LiveAvatarSave, LiveAvatarExport]


async def comfy_entrypoint() -> LiveAvatarExtension:
    return LiveAvatarExtension()
