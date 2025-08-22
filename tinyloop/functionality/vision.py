import base64
import io
import mimetypes
import os
from typing import Any, Union
from urllib.parse import urlparse

import requests
from PIL import Image as PILImage


class Image:
    def __init__(
        self,
        *,
        from_url: str = None,
        from_pil: PILImage.Image = None,
        from_file: str = None,
    ):
        """
        Initialize Image with different input sources.

        Args:
            from_url: Initialize from URL (keyword-only)
            from_pil: Initialize from PIL Image (keyword-only)
            from_file: Initialize from file path (keyword-only)
        """
        sources = [from_url, from_pil, from_file]
        provided_sources = sum(x is not None for x in sources)

        if provided_sources != 1:
            raise ValueError(
                "Exactly one of from_url, from_pil, or from_file must be provided"
            )

        if from_url is not None:
            self.url = from_url
            self.mime_type = self._guess_mime_type_from_url(from_url)
        elif from_pil is not None:
            self.url, self.mime_type = self._encode_pil_image(from_pil)
        elif from_file is not None:
            self.url, self.mime_type = self._encode_image_from_file(from_file)

    @classmethod
    def from_url(cls, image_url: str):
        """Create Image instance from URL."""
        return cls(from_url=image_url)

    @classmethod
    def from_PIL(cls, image: PILImage.Image):
        """Create Image instance from PIL Image."""
        return cls(from_pil=image)

    @classmethod
    def from_file(cls, file_path: str):
        """Create Image instance from file path."""
        return cls(from_file=file_path)

    def format(self) -> list[dict[str, Any]]:
        """Format image for API consumption with URL and MIME type."""
        return [
            {
                "type": "image_url",
                "image_url": {"url": self.url, "mime_type": self.mime_type},
            }
        ]

    def _is_url(self, string: str) -> bool:
        """Check if a string is a valid URL."""
        try:
            result = urlparse(string)
            return all([result.scheme in ("http", "https", "gs"), result.netloc])
        except ValueError:
            return False

    def _guess_mime_type_from_url(self, url: str) -> str:
        """Guess MIME type from URL."""
        if url.startswith("data:"):
            # Extract MIME type from data URI
            return url.split(";")[0].split(":")[1]
        elif self._is_url(url):
            # Try to guess from URL path
            mime_type, _ = mimetypes.guess_type(url)
            return mime_type or "image/jpeg"  # Default fallback
        else:
            return "image/jpeg"  # Default fallback

    def _encode_image_from_file(self, file_path: str) -> tuple[str, str]:
        """Encode a file from a file path to a base64 data URI with MIME type."""
        with open(file_path, "rb") as file:
            file_data = file.read()

        # Use mimetypes to guess directly from the file path
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None:
            raise ValueError(f"Could not determine MIME type for file: {file_path}")

        encoded_data = base64.b64encode(file_data).decode("utf-8")
        data_url = f"data:{mime_type};base64,{encoded_data}"
        return data_url, mime_type

    def _encode_pil_image(
        self, image: PILImage.Image, format: str = None
    ) -> tuple[str, str]:
        """Encode a PIL Image object to a base64 data URI with MIME type."""
        buffered = io.BytesIO()
        file_format = format or image.format or "PNG"
        image.save(buffered, format=file_format)

        # Get the correct MIME type using the image format
        file_extension = file_format.lower()
        mime_type, _ = mimetypes.guess_type(f"file.{file_extension}")
        if mime_type is None:
            # Fallback MIME types for common formats
            format_to_mime = {
                "png": "image/png",
                "jpg": "image/jpeg",
                "jpeg": "image/jpeg",
                "gif": "image/gif",
                "bmp": "image/bmp",
                "webp": "image/webp",
            }
            mime_type = format_to_mime.get(file_extension, "image/png")

        encoded_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
        data_url = f"data:{mime_type};base64,{encoded_data}"
        return data_url, mime_type

    def __str__(self):
        """String representation of the Image."""
        if "base64" in self.url:
            len_base64 = len(self.url.split("base64,")[1])
            return f"Image(url=data:{self.mime_type};base64,<IMAGE_BASE_64_ENCODED({len_base64})>, mime_type='{self.mime_type}')"
        return f"Image(url='{self.url}', mime_type='{self.mime_type}')"

    def __repr__(self):
        """Detailed representation of the Image."""
        return self.__str__()


def is_url(string: str) -> bool:
    """Check if a string is a valid URL."""
    try:
        result = urlparse(string)
        return all([result.scheme in ("http", "https", "gs"), result.netloc])
    except ValueError:
        return False


def is_image(obj) -> bool:
    """Check if the object is an image or a valid media file reference."""
    if isinstance(obj, PILImage.Image):
        return True
    if isinstance(obj, str):
        if obj.startswith("data:"):
            return True
        elif os.path.isfile(obj):
            return True
        elif is_url(obj):
            return True
    return False


def encode_image(
    image: Union[str, bytes, PILImage.Image, dict], download_images: bool = False
) -> tuple[str, str]:
    """
    Encode an image or file to a base64 data URI with MIME type detection.

    Args:
        image: The image or file to encode. Can be a PIL Image, file path, URL, or data URI.
        download_images: Whether to download images from URLs.

    Returns:
        tuple: (data_uri, mime_type) The data URI of the file and its MIME type.

    Raises:
        ValueError: If the file type is not supported.
    """
    if isinstance(image, dict) and "url" in image:
        url = image["url"]
        mime_type = _guess_mime_type_from_url(url)
        return url, mime_type
    elif isinstance(image, str):
        if image.startswith("data:"):
            # Already a data URI, extract MIME type
            mime_type = image.split(";")[0].split(":")[1]
            return image, mime_type
        elif os.path.isfile(image):
            # File path
            return _encode_image_from_file(image)
        elif is_url(image):
            # URL
            if download_images:
                return _encode_image_from_url(image)
            else:
                # Return the URL as is with guessed MIME type
                mime_type = _guess_mime_type_from_url(image)
                return image, mime_type
        else:
            raise ValueError(f"Unsupported file string: {image}")
    elif isinstance(image, PILImage.Image):
        # PIL Image
        return _encode_pil_image(image)
    elif isinstance(image, bytes):
        # Raw bytes
        img = PILImage.open(io.BytesIO(image))
        return _encode_pil_image(img)
    elif isinstance(image, Image):
        return image.url, image.mime_type
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")


def _guess_mime_type_from_url(url: str) -> str:
    """Guess MIME type from URL."""
    if url.startswith("data:"):
        # Extract MIME type from data URI
        return url.split(";")[0].split(":")[1]
    elif is_url(url):
        # Try to guess from URL path
        mime_type, _ = mimetypes.guess_type(url)
        return mime_type or "image/jpeg"  # Default fallback
    else:
        return "image/jpeg"  # Default fallback


def _encode_image_from_file(file_path: str) -> tuple[str, str]:
    """Encode a file from a file path to a base64 data URI with MIME type."""
    with open(file_path, "rb") as file:
        file_data = file.read()

    # Use mimetypes to guess directly from the file path
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        raise ValueError(f"Could not determine MIME type for file: {file_path}")

    encoded_data = base64.b64encode(file_data).decode("utf-8")
    data_url = f"data:{mime_type};base64,{encoded_data}"
    return data_url, mime_type


def _encode_image_from_url(image_url: str) -> tuple[str, str]:
    """Encode a file from a URL to a base64 data URI with MIME type."""
    response = requests.get(image_url)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")

    # Use the content type from the response headers if available
    if content_type:
        mime_type = content_type
    else:
        # Try to guess MIME type from URL
        mime_type, _ = mimetypes.guess_type(image_url)
        if mime_type is None:
            raise ValueError(f"Could not determine MIME type for URL: {image_url}")

    encoded_data = base64.b64encode(response.content).decode("utf-8")
    data_url = f"data:{mime_type};base64,{encoded_data}"
    return data_url, mime_type


def _encode_pil_image(image: PILImage.Image, format: str = None) -> tuple[str, str]:
    """Encode a PIL Image object to a base64 data URI with MIME type."""
    buffered = io.BytesIO()
    file_format = format or image.format or "PNG"
    image.save(buffered, format=file_format)

    # Get the correct MIME type using the image format
    file_extension = file_format.lower()
    mime_type, _ = mimetypes.guess_type(f"file.{file_extension}")
    if mime_type is None:
        # Fallback MIME types for common formats
        format_to_mime = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "gif": "image/gif",
            "bmp": "image/bmp",
            "webp": "image/webp",
        }
        mime_type = format_to_mime.get(file_extension, "image/png")

    encoded_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
    data_url = f"data:{mime_type};base64,{encoded_data}"
    return data_url, mime_type
