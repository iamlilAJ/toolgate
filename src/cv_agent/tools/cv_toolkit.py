"""Tools for CV Agent."""

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, ValidationInfo, field_validator

from cv_agent.core.registries import tool_registry
from cv_agent.tools.base import McpToolBase

# OCR ###

#
# class OCRInput(BaseModel):
#     image_url: str = Field(
#         description="Complete image access URL with protocol header and full path for ocr."
#     )
#
#
# class OCRTool(McpToolBase):
#     def __init__(
#         self,
#         mcp_server_url: str,
#         name: str = "ocr",
#         description: str | None = None,
#         args_schema: type[BaseModel] = OCRInput,
#     ) -> None:
#         if description is None:
#             description = "Perform OCR to an image."
#
#         super().__init__(mcp_server_url, name, description, args_schema)
#
#     async def invoke(self, image_url: str):
#         async with self._client as client:
#             result = await client.call_tool("ocr_image_from_url", {"image_url": image_url})
#
#         return result
#
#
# @tool_registry.register("ocr")
# def get_mcp_ocr_tool(
#     server_url: str,
#     name: str = "ocr",
#     description: str | None = None,
#     args_schema: type[BaseModel] = OCRInput,
# ) -> StructuredTool:
#     return OCRTool(server_url, name, description, args_schema).langchain_tool
#

# MinerU OCR


class MinerU_OCRInput(BaseModel):
    image_url: str = Field(
        description="Complete image access URL with protocol header and full path for ocr."
    )


class MinerU_OCRTool(McpToolBase):
    def __init__(
        self,
        mcp_server_url: str,
        name: str = "ocr",
        description: str | None = None,
        args_schema: type[BaseModel] = MinerU_OCRInput,
    ) -> None:
        if description is None:
            description = "Perform OCR to an image. "
        self.mcp_server_url = mcp_server_url
        super().__init__(mcp_server_url, name, description, args_schema)

    async def invoke(self, image_url: str):
        if "node1" in self.mcp_server_url:
            tool_name = "ocr_image_with_mineru"
        else:
            tool_name = "parse_pdf_file_parse_post"
        async with self._client as client:
            result = await client.call_tool(tool_name, {"urls": [image_url], "lang_list": ["en"]})

        return result


@tool_registry.register("ocr")
def get_mcp_mineru_ocr_tool(
    server_url: str,
    name: str = "ocr",
    description: str | None = None,
    args_schema: type[BaseModel] = MinerU_OCRInput,
) -> StructuredTool:
    return MinerU_OCRTool(server_url, name, description, args_schema).langchain_tool


# Detection ###


class DetectionInput(BaseModel):
    image_url: str = Field(
        description="Complete image access URL with protocol header and full path for detection."
    )

    # Required: Double-nested list of target object phrases
    targets: list[str] = Field(
        description="A single list of simple English nouns or noun phrases for all target objects. "
        "All labels must be placed in a single inner list.",
        examples=["a cat", "a mouse"],
    )


class DetectionTool(McpToolBase):
    def __init__(
        self,
        mcp_server_url: str,
        name: str = "detection",
        description: str | None = None,
        args_schema: type[BaseModel] = DetectionInput,
    ) -> None:
        if description is None:
            description = (
                "Detect objects in an image based on natural language labels.\n\n"
                "❗IMPORTANT FOR LLM CALLERS:\n"
                "This model works best when object descriptions are in the canonical format: "
                "'[modifier] [noun]' (e.g., 'red car', 'glass bottle', 'with hat dog').\n"
                "Before calling this tool, REWRITE any user description into this form. Examples:\n"
                "- 'a car that is red' → 'red car'\n"
                "- 'a bottle made of glass' → 'glass bottle'\n"
                "- 'a dog wearing sunglasses' → 'wearing sunglasses dog'\n"
                "- 'an old wooden chair' → 'old wooden chair' (already good)\n\n"
                "DO NOT pass original complex sentences, relative clauses, or post-noun modifiers. "
                "Only send concise, pre-modified labels."
            )

        super().__init__(mcp_server_url, name, description, args_schema)

    async def invoke(self, image_url: str, targets: list[str]):
        async with self._client as client:
            result = await client.call_tool(
                "llmdet_detection_llmdet_detection_post", {"image_url": image_url, "text": targets}
            )
            # if result.content and result.content[0].type == "text":
            #     data = json.loads(result.content[0].text)
            #     if data.get("status") == "success" and data.get("count") == 0:
            #         data["output_image"] = None
            #         # Re-serialize the modified data
            #         result.content[0].text = json.dumps(data)

        return result


@tool_registry.register("detection")
def get_mcp_detection_tool(
    server_url: str,
    name: str = "detection",
    description: str | None = None,
    args_schema: type[BaseModel] = DetectionInput,
) -> StructuredTool:
    return DetectionTool(server_url, name, description, args_schema).langchain_tool


# Detection ###


# ----------------------------------------------------
# NEW: Detection (Crop Mix) Tool
# ----------------------------------------------------


class DetectionCropMixTool(McpToolBase):
    """
    Client wrapper for the new high-recall 'llmdet_crop_mix...' tool.
    It uses the *same input schema* as the standard detection tool.
    """

    def __init__(
        self,
        mcp_server_url: str,
        name: str = "detection_small_object",
        description: str | None = None,
        args_schema: type[BaseModel] = DetectionInput,  # Re-uses the same input
    ) -> None:
        if description is None:
            description = (
                "High-recall detection for SMALL objects. This tool crops the image into "
                "many small pieces to find small targets. "
                "WARNING: This tool may return multiple overlapping boxes for large objects."
                "Detect objects in an image based on natural language labels.\n\n"
                "❗IMPORTANT FOR LLM CALLERS:\n"
                "This model works best when object descriptions are in the canonical format: "
                "'[modifier] [noun]' (e.g., 'red car', 'glass bottle', 'with hat dog').\n"
                "Before calling this tool, REWRITE any user description into this form. Examples:\n"
                "- 'a car that is red' → 'red car'\n"
                "- 'a bottle made of glass' → 'glass bottle'\n"
                "- 'a dog wearing sunglasses' → 'wearing sunglasses dog'\n"
                "- 'an old wooden chair' → 'old wooden chair' (already good)\n\n"
                "DO NOT pass original complex sentences, relative clauses, or post-noun modifiers. "
                "Only send concise, pre-modified labels."
            )

        super().__init__(mcp_server_url, name, description, args_schema)

    async def invoke(self, image_url: str, targets: list[str]):
        async with self._client as client:
            result = await client.call_tool(
                "llmdet_detection_crop_mix_llmdet_detection_crop_mix_post",
                {"image_url": image_url, "text": targets},
            )
            # if result.content and result.content[0].type == "text":
            #     data = json.loads(result.content[0].text)
            #     if data.get("status") == "success" and data.get("count") == 0:
            #         data["output_image"] = None
            #         # Re-serialize the modified data
            #         result.content[0].text = json.dumps(data)

        return result


@tool_registry.register("detection_small_object")
def get_mcp_detection_crop_mix_tool(
    server_url: str,
    name: str = "detection_small_object",
    description: str | None = None,
    args_schema: type[BaseModel] = DetectionInput,
) -> StructuredTool:
    return DetectionCropMixTool(server_url, name, description, args_schema).langchain_tool


# ----------------------------------------------------
# Segmentation Tool
# ----------------------------------------------------


class SegmentationInput(BaseModel):
    """Input schema for the segmentation_segmentation_post tool."""

    image_url: str = Field(
        description="Complete image access URL with protocol header and full path "
        "for segmentation.",
    )

    text_prompt: str = Field(
        description="Text description of the object to be segmented from the image.",
    )


class SegmentationTool(McpToolBase):
    """
    'segmentation_segmentation_post'
    """

    def __init__(
        self,
        mcp_server_url: str,
        name: str = "segmentation",
        description: str | None = None,
        args_schema: type[BaseModel] = SegmentationInput,
    ) -> None:
        if description is None:
            description = "Generate segmentation maps (masks) from an image based on a text prompt."

        super().__init__(mcp_server_url, name, description, args_schema)

    async def invoke(self, image_url: str, text_prompt: str):
        REMOTE_TOOL_NAME = "segmentation_segmentation_post"

        payload = {
            "image_url": image_url,
            "text_prompt": text_prompt,
        }

        async with self._client as client:
            result = await client.call_tool(REMOTE_TOOL_NAME, payload)

        return result


@tool_registry.register("segmentation")
def get_mcp_segmentation_tool(
    server_url: str,
    name: str = "segmentation",
    description: str | None = None,
    args_schema: type[BaseModel] = SegmentationInput,
) -> StructuredTool:
    return SegmentationTool(server_url, name, description, args_schema).langchain_tool


# ----------------------------------------------------
#  Cropping Tool (crop_image_tool_crop_image_post)
# ----------------------------------------------------


class CropImageInput(BaseModel):
    image_url: str = Field(...)
    # Accept coordinates as a list
    coordinates: list[float] = Field(description="List of 4 coordinates [x1, y1, x2, y2]")

    # These will be populated by the validator, not by the LLM
    x1: int = Field(exclude=True, default=0)
    y1: int = Field(exclude=True, default=0)
    x2: int = Field(exclude=True, default=0)
    y2: int = Field(exclude=True, default=0)

    @field_validator("coordinates")
    @classmethod
    def convert_coordinates(cls, v: list[float], values: "ValidationInfo") -> list[float]:
        if len(v) != 4:
            raise ValueError("Coordinates list must contain exactly 4 floats.")

        # Populate the individual fields
        values.data["x1"] = int(v[0])
        values.data["y1"] = int(v[1])
        values.data["x2"] = int(v[2])
        values.data["y2"] = int(v[3])
        return v


class CroppingTool(McpToolBase):
    """
    Client wrapper for the remote 'crop_image_tool_crop_image_post' tool.
    This tool is robust: it accepts a list of float coordinates and
    handles the conversion to integers internally.
    """

    def __init__(
        self,
        mcp_server_url: str,
        name: str = "crop_image_tool_crop_image_post",
        description: str | None = None,
        # --- CHANGE 1: Use the new, robust schema ---
        args_schema: type[BaseModel] = CropImageInput,
    ) -> None:
        if description is None:
            # --- CHANGE 2: Update description to match the new schema ---
            description = (
                "Precise Image Cropping Tool. "
                "Crops an image based on a list of 4 float coordinates: [x1, y1, x2, y2]."
            )

        super().__init__(mcp_server_url, name, description, args_schema)

    # --- CHANGE 3: Update the invoke method ---
    async def invoke(self, image_url: str, coordinates: list[float]):
        """
        This method is called by LangChain *after* the CropImageInput
        model has successfully validated the input.
        """
        REMOTE_TOOL_NAME = "crop_image_tool_crop_image_post"

        # The validation has already run, so we can just re-do the
        # simple conversion to integers to build the payload.
        x1 = int(coordinates[0])
        y1 = int(coordinates[1])
        x2 = int(coordinates[2])
        y2 = int(coordinates[3])

        # This payload matches what your *remote* API expects
        payload = {"image_url": image_url, "x1": x1, "y1": y1, "x2": x2, "y2": y2}

        async with self._client as client:
            return await client.call_tool(REMOTE_TOOL_NAME, payload)


@tool_registry.register("cropping")
def get_mcp_cropping_tool(
    server_url: str,
    name: str = "cropping",
    description: str | None = None,
    args_schema: type[BaseModel] = CropImageInput,
) -> StructuredTool:
    return CroppingTool(server_url, name, description, args_schema).langchain_tool


# ----------------------------------------------------
#  Depth Estimation Tool (depth_estimation_depth_estimation_post)
# ----------------------------------------------------


class DepthEstimationInput(BaseModel):
    """Input schema for the depth_estimation_depth_estimation_post tool."""

    image_url: str = Field(
        description="Complete image access URL with protocol header and full path."
    )


class DepthEstimationTool(McpToolBase):
    """Client wrapper for the remote 'depth_estimation_depth_estimation_post' tool."""

    def __init__(
        self,
        mcp_server_url: str,
        name: str = "depth_estimation_depth_estimation_post",
        description: str | None = None,  #
        args_schema: type[BaseModel] = DepthEstimationInput,
    ) -> None:
        if description is None:
            description = "Generate colored depth maps from a single input image."

        super().__init__(mcp_server_url, name, description, args_schema)

    async def invoke(self, image_url: str):
        REMOTE_TOOL_NAME = "depth_estimation_depth_estimation_post"
        payload = {"image_url": image_url}
        async with self._client as client:
            return await client.call_tool(REMOTE_TOOL_NAME, payload)


@tool_registry.register("depth_estimation")
def get_mcp_depth_estimation_tool(
    server_url: str,
    name: str = "depth_estimation",
    description: str | None = None,
    args_schema: type[BaseModel] = DepthEstimationInput,
) -> StructuredTool:
    return DepthEstimationTool(server_url, name, description, args_schema).langchain_tool


# ----------------------------------------------------
# Bounding Box Drawing Tool (bbox_drawing_tool_bbox_drawing_post)
# ----------------------------------------------------


class BBoxDrawingInfo(BaseModel):
    """Internal schema for a single box in BBoxDrawingInput."""

    bbox: list[int] = Field(
        description="Bounding box coordinates [x1, y1, x2, y2].",
    )
    color: str = Field(description="Box color: color names (red), RGB (#RRGGBB), or 'auto'.")
    label: str | None = Field(
        description="Object label text (displayed above box if provided).", default=None
    )  # 使用 str | None


class BBoxDrawingInput(BaseModel):
    """Input schema for the bbox_drawing_tool_bbox_drawing_post tool."""

    image_url: str = Field(
        description="Complete image access URL with protocol header and full path."
    )
    boxes: list[BBoxDrawingInfo] = Field(
        description="List of bounding box information with coordinates, colors and labels."
    )
    line_thickness: int = Field(description="Bounding box line thickness in pixels (1-10).")
    font_size: int = Field(description="Label text size in pixels (10-100).")


class BBoxDrawingTool(McpToolBase):
    """Client wrapper for the remote 'bbox_drawing_tool_bbox_drawing_post' tool."""

    def __init__(
        self,
        mcp_server_url: str,
        name: str = "bbox_drawing_tool_bbox_drawing_post",
        description: str | None = None,
        args_schema: type[BaseModel] = BBoxDrawingInput,
    ) -> None:
        if description is None:
            description = (
                "Draw bounding boxes with custom colors and labels (Input: structured boxes)."
            )

        super().__init__(mcp_server_url, name, description, args_schema)

    async def invoke(
        self,
        image_url: str,
        boxes: list[BBoxDrawingInfo],
        line_thickness: int | None = None,
        font_size: int | None = None,
    ):
        REMOTE_TOOL_NAME = "bbox_drawing_tool_bbox_drawing_post"

        # 仅将非 None 的可选参数添加到 payload
        payload = {"image_url": image_url, "boxes": boxes}
        if line_thickness is not None:
            payload["line_thickness"] = line_thickness
        if font_size is not None:
            payload["font_size"] = font_size

        async with self._client as client:
            return await client.call_tool(REMOTE_TOOL_NAME, payload)


@tool_registry.register("bbox_drawing")
def get_mcp_bbox_drawing_tool(
    server_url: str,
    name: str = "bbox_drawing",
    description: str | None = None,
    args_schema: type[BaseModel] = BBoxDrawingInput,
) -> StructuredTool:
    return BBoxDrawingTool(server_url, name, description, args_schema).langchain_tool


# ----------------------------------------------------
# 5. Bounding Box Center Labeling Tool (bbox_center_labeling_tool_bbox_center_labeling_post)
# ----------------------------------------------------


class BBoxCenterLabelingInput(BaseModel):
    """Input schema for the bbox_center_labeling_tool_bbox_center_labeling_post tool."""

    image_url: str = Field(
        description="Complete image access URL with protocol header and full path."
    )
    bboxes: list[list[int]] = Field(
        description="List of bounding boxes in [x1, y1, x2, y2] format (pixel coordinates)."
    )
    font_size: int = Field(description="Font size for numeric labels (10-200).")


class BBoxCenterLabelingTool(McpToolBase):
    """Client wrapper for the remote 'bbox_center_labeling_tool_bbox_center_labeling_post' tool."""

    def __init__(
        self,
        mcp_server_url: str,
        name: str = "bbox_center_labeling_tool_bbox_center_labeling_post",
        description: str | None = None,  # 使用 str | None
        args_schema: type[BaseModel] = BBoxCenterLabelingInput,
    ) -> None:
        if description is None:
            description = "Draw sequential numeric labels at bounding box centers (Input: bboxes)."

        super().__init__(mcp_server_url, name, description, args_schema)

    async def invoke(self, image_url: str, bboxes: list[list[int]], font_size: int | None = None):
        REMOTE_TOOL_NAME = "bbox_center_labeling_tool_bbox_center_labeling_post"
        payload = {"image_url": image_url, "bboxes": bboxes}
        if font_size is not None:
            payload["font_size"] = font_size

        async with self._client as client:
            return await client.call_tool(REMOTE_TOOL_NAME, payload)


@tool_registry.register("bbox_center_labeling")
def get_mcp_bbox_center_labeling_tool(
    server_url: str,
    name: str = "bbox_center_labeling",
    description: str | None = None,
    args_schema: type[BaseModel] = BBoxCenterLabelingInput,
) -> StructuredTool:
    return BBoxCenterLabelingTool(server_url, name, description, args_schema).langchain_tool
