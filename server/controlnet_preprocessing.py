"""
Drives ControlNet preprocessing for a request object.

For each attachment carrying source_asset_ref + preprocess:
  1. Resolve source bytes from the AssetStore.
  2. Invoke the named preprocessor.
  3. Store the emitted control map as a control_map asset.
  4. Backfill attachment.map_asset_ref for downstream execution paths.
  5. Return a ControlNetArtifactRef describing the emitted artifact.

Attachments that already carry map_asset_ref pass through unchanged.
"""

from typing import Optional

from server.asset_store import AssetStore
from server.controlnet_models import ControlNetArtifactRef
from server.controlnet_preprocessors import DEFAULT_REGISTRY, PreprocessorRegistry


def preprocess_controlnet_attachments(
    req,
    store: AssetStore,
    registry: Optional[PreprocessorRegistry] = None,
) -> list[ControlNetArtifactRef]:
    if registry is None:
        registry = DEFAULT_REGISTRY

    attachments = getattr(req, "controlnets", None) or []
    artifacts: list[ControlNetArtifactRef] = []

    for index, attachment in enumerate(attachments):
        if attachment.source_asset_ref is None or attachment.preprocess is None:
            continue

        source_ref = attachment.source_asset_ref
        try:
            source_entry = store.resolve(source_ref)
        except KeyError as exc:
            raise ValueError(f"source_asset_ref {source_ref!r} not found or evicted") from exc

        preprocessor = registry.get(attachment.preprocess.id)
        if preprocessor is None:
            raise ValueError(f"unknown preprocessor {attachment.preprocess.id!r}")

        result = preprocessor.run(source_entry.data, attachment.preprocess.options)
        metadata = {
            "attachment_id": attachment.attachment_id,
            "control_type": attachment.control_type,
            "source_asset_ref": source_ref,
            "preprocessor_id": result.preprocessor_id,
            "width": result.width,
            "height": result.height,
            "media_type": result.media_type,
        }
        new_ref = store.insert("control_map", result.image_bytes, metadata)
        normalized_attachment = attachment.model_copy(
            update={
                "map_asset_ref": new_ref,
                "source_asset_ref": None,
                "preprocess": None,
            }
        )
        attachments[index] = normalized_attachment

        artifacts.append(
            ControlNetArtifactRef(
                attachment_id=normalized_attachment.attachment_id,
                asset_ref=new_ref,
                control_type=normalized_attachment.control_type,
                preprocessor_id=result.preprocessor_id,
                source_asset_ref=source_ref,
            )
        )

    return artifacts
