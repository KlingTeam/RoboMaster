# ------------------------------------------------------------------------
# Copyright (c) 2023-present, PhyscalX. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Gradio ``ImagePrompter`` component."""

from __future__ import annotations

from typing import Optional, List, TypedDict, Union, Literal

import numpy as np
import gradio
from gradio.data_classes import FileData, GradioModel
from gradio_client.documentation import document, set_documentation_group
from PIL import Image as _Image  # using _ to minimize namespace pollution

set_documentation_group("component")


class PromptData(GradioModel):
    image: FileData
    points: List[List[float]]


class PromptValue(TypedDict):
    image: Optional[Union[np.ndarray, _Image.Image, str]]
    points: Optional[List[List[float]]]


@document()
class ImagePrompter(gradio.Image):
    """Create an image prompter to upload images and process point/box prompts."""

    data_model = PromptData

    def __init__(
        self,
        value: str | _Image.Image | np.ndarray | None = None,
        *,
        height: int | None = None,
        width: int | None = None,
        image_mode: Literal[
            "1", "L", "P", "RGB", "RGBA", "CMYK", "YCbCr", "LAB", "HSV", "I", "F"
        ] = "RGB",
        sources: list[Literal["upload", "clipboard"]] | None = None,
        type: Literal["numpy", "pil", "filepath"] = "numpy",
        label: str | None = None,
        every: float | None = None,
        show_label: bool | None = None,
        show_download_button: bool = True,
        container: bool = True,
        scale: int | None = None,
        min_width: int = 160,
        interactive: bool | None = None,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        render: bool = True,
        show_share_button: bool | None = None,
    ):
        """
        Parameters:
            value: A PIL Image, numpy array, path or URL for the default value. If callable, it will be called set the initial value.
            height: Height of the displayed image in pixels.
            width: Width of the displayed image in pixels.
            image_mode: "RGB" if color, or "L" if black and white. See https://pillow.readthedocs.io/en/stable/handbook/concepts.html.
            sources: List of sources for the image.
            type: The format the image is converted before being passed into the prediction function.
            label: The label for this component.
            every: If `value` is a callable, run the function 'every' number of seconds while the client connection is open.
            show_label: if True, will display label.
            show_download_button: If True, will display button to download image.
            container: If True, will place the component in a container - providing some extra padding around the border.
            scale: relative width compared to adjacent Components in a Row. Should be an integer.
            min_width: minimum pixel width, will wrap if not sufficient screen space to satisfy this value.
            interactive: if True, will allow users to upload and edit an image; if False, can only be used to display images.
            visible: If False, component will be hidden.
            streaming: If True when used in a `live` interface, will automatically stream webcam feed. Only valid is source is 'webcam'.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM.
            render: If False, component will not render be rendered in the Blocks context.
            mirror_webcam: If True webcam will be mirrored. Default is True.
            show_share_button: If True, show a share icon that allows user to share outputs to Hugging Face Spaces Discussions.
        """
        super(ImagePrompter, self).__init__(
            value=value,
            height=height,
            width=width,
            image_mode=image_mode,
            sources=["upload", "clipboard"] if sources is None else sources,
            type=type,
            label=label,
            every=every,
            show_label=show_label,
            show_download_button=show_download_button,
            container=container,
            scale=scale,
            min_width=min_width,
            interactive=interactive,
            visible=visible,
            elem_id=elem_id,
            elem_classes=elem_classes,
            render=render,
            show_share_button=show_share_button,
        )

    def preprocess(self, x: PromptData) -> PromptValue | None:
        if x is None:
            return x
        im = super().preprocess(x.image)
        points = [point[:2] for point in x.points]
        np.save('coord.npy',np.array(points).astype(np.int32))
        return {"image": im, "points": points}

    def postprocess(self, y: PromptValue) -> PromptData | None:
        if y is None:
            return None
        image, points = y.get("image", None), y.get("points", [])
        return PromptData(image=super().postprocess(image), points=points)

    def as_example(self, y: PromptValue) -> str | None:
        if y is None:
            return None
        return self.move_resource_to_block_cache(y.get("image", None))
