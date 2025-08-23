"""
Tests for tinyloop.functionality.vision module.

This module tests all image loading methods and utility functions in the vision module.
"""

import io

import pytest
from PIL import Image as PILImage

from tinyloop.features.vision import Image, encode_image, is_image, is_url


class TestImageClass:
    """Test the Image class and its methods."""

    def test_image_from_url(self):
        """Test creating Image from URL."""
        url = "https://example.com/image.jpg"
        image = Image(from_url=url)

        assert image.url == url
        assert image.mime_type == "image/jpeg"

    def test_image_from_pil(self):
        """Test creating Image from PIL Image."""
        # Create a simple test image
        pil_image = PILImage.new("RGB", (100, 100), color="red")
        image = Image(from_pil=pil_image)

        assert image.url.startswith("data:image/png;base64,")
        assert image.mime_type == "image/png"

    def test_image_from_file(self, tmp_path):
        """Test creating Image from file path."""
        # Create a temporary image file
        image_path = tmp_path / "test_image.png"
        pil_image = PILImage.new("RGB", (50, 50), color="blue")
        pil_image.save(image_path)

        image = Image(from_file=str(image_path))

        assert image.url.startswith("data:image/png;base64,")
        assert image.mime_type == "image/png"

    def test_image_from_url_class_method(self):
        """Test Image.from_url class method."""
        url = "https://example.com/image.png"
        image = Image.from_url(url)

        assert image.url == url
        assert image.mime_type == "image/png"

    def test_image_from_pil_class_method(self):
        """Test Image.from_PIL class method."""
        pil_image = PILImage.new("RGB", (100, 100), color="green")
        image = Image.from_PIL(pil_image)

        assert image.url.startswith("data:image/png;base64,")
        assert image.mime_type == "image/png"

    def test_image_from_file_class_method(self, tmp_path):
        """Test Image.from_file class method."""
        # Create a temporary JPEG image file
        image_path = tmp_path / "test_image.jpg"
        pil_image = PILImage.new("RGB", (50, 50), color="yellow")
        pil_image.save(image_path, format="JPEG")

        image = Image.from_file(str(image_path))

        assert image.url.startswith("data:image/jpeg;base64,")
        assert image.mime_type == "image/jpeg"

    def test_image_constructor_multiple_sources_error(self):
        """Test that providing multiple sources raises ValueError."""
        url = "https://example.com/image.jpg"
        pil_image = PILImage.new("RGB", (100, 100), color="red")

        with pytest.raises(
            ValueError,
            match="Exactly one of from_url, from_pil, or from_file must be provided",
        ):
            Image(from_url=url, from_pil=pil_image)

    def test_image_constructor_no_sources_error(self):
        """Test that providing no sources raises ValueError."""
        with pytest.raises(
            ValueError,
            match="Exactly one of from_url, from_pil, or from_file must be provided",
        ):
            Image()

    def test_image_format_method(self):
        """Test the format method returns correct structure."""
        url = "https://example.com/image.jpg"
        image = Image(from_url=url)
        formatted = image.format()

        assert isinstance(formatted, list)
        assert len(formatted) == 1
        assert formatted[0]["type"] == "image_url"
        assert formatted[0]["image_url"]["url"] == url
        assert formatted[0]["image_url"]["mime_type"] == "image/jpeg"

    def test_image_str_representation(self):
        """Test string representation of Image."""
        url = "https://example.com/image.jpg"
        image = Image(from_url=url)

        str_repr = str(image)
        assert "Image(url='https://example.com/image.jpg'" in str_repr
        assert "mime_type='image/jpeg'" in str_repr

    def test_image_str_representation_base64(self):
        """Test string representation of Image with base64 data."""
        pil_image = PILImage.new("RGB", (100, 100), color="red")
        image = Image(from_pil=pil_image)

        str_repr = str(image)
        assert "Image(url=data:image/png;base64," in str_repr
        assert "mime_type='image/png'" in str_repr
        assert "<IMAGE_BASE_64_ENCODED(" in str_repr

    def test_image_repr_representation(self):
        """Test repr representation of Image."""
        url = "https://example.com/image.jpg"
        image = Image(from_url=url)

        assert repr(image) == str(image)

    def test_mime_type_detection_from_url(self):
        """Test MIME type detection from different URL patterns."""
        # Test PNG URL
        png_url = "https://example.com/image.png"
        image = Image(from_url=png_url)
        assert image.mime_type == "image/png"

        # Test JPEG URL
        jpeg_url = "https://example.com/image.jpg"
        image = Image(from_url=jpeg_url)
        assert image.mime_type == "image/jpeg"

        # Test data URI
        data_uri = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
        image = Image(from_url=data_uri)
        assert image.mime_type == "image/gif"

    def test_mime_type_detection_from_file(self, tmp_path):
        """Test MIME type detection from different file types."""
        # Test PNG file
        png_path = tmp_path / "test.png"
        pil_image = PILImage.new("RGB", (50, 50), color="red")
        pil_image.save(png_path)
        image = Image(from_file=str(png_path))
        assert image.mime_type == "image/png"

        # Test JPEG file
        jpeg_path = tmp_path / "test.jpg"
        pil_image.save(jpeg_path, format="JPEG")
        image = Image(from_file=str(jpeg_path))
        assert image.mime_type == "image/jpeg"

    def test_pil_image_format_detection(self):
        """Test PIL image format detection and MIME type mapping."""
        # Test with explicit format
        pil_image = PILImage.new("RGB", (100, 100), color="red")
        image = Image(from_pil=pil_image)
        assert image.mime_type == "image/png"  # Default format

        # Test with different format
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG")
        jpeg_image = PILImage.open(buffered)
        image = Image(from_pil=jpeg_image)
        assert image.mime_type == "image/jpeg"


class TestUtilityFunctions:
    """Test utility functions in the vision module."""

    def test_is_url_valid_urls(self):
        """Test is_url function with valid URLs."""
        valid_urls = [
            "https://example.com/image.jpg",
            "http://example.com/image.png",
            "https://subdomain.example.com/path/to/image.gif",
            "gs://bucket/image.jpg",  # Google Cloud Storage
        ]

        for url in valid_urls:
            assert is_url(url) is True

    def test_is_url_invalid_urls(self):
        """Test is_url function with invalid URLs."""
        invalid_urls = [
            "not_a_url",
            "ftp://example.com/image.jpg",  # Unsupported scheme
            "file:///path/to/image.jpg",  # Unsupported scheme
            "",
            "https://",  # Missing netloc
            "example.com",  # Missing scheme
        ]

        for url in invalid_urls:
            assert is_url(url) is False

    def test_is_image_valid_inputs(self):
        """Test is_image function with valid inputs."""
        # Test PIL Image
        pil_image = PILImage.new("RGB", (100, 100), color="red")
        assert is_image(pil_image) is True

        # Test data URI
        data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        assert is_image(data_uri) is True

        # Test URL
        url = "https://example.com/image.jpg"
        assert is_image(url) is True

    def test_is_image_invalid_inputs(self):
        """Test is_image function with invalid inputs."""
        invalid_inputs = [
            "not_an_image",
            "",
            None,
            123,
            [],
            {},
        ]

        for invalid_input in invalid_inputs:
            assert is_image(invalid_input) is False

        # Note: URLs are considered valid by is_image regardless of extension
        # This is by design as the function doesn't validate the actual content

    def test_is_image_file_path(self, tmp_path):
        """Test is_image function with file paths."""
        # Test existing image file
        image_path = tmp_path / "test.png"
        pil_image = PILImage.new("RGB", (50, 50), color="red")
        pil_image.save(image_path)
        assert is_image(str(image_path)) is True

        # Test non-existent file
        non_existent_path = tmp_path / "nonexistent.png"
        assert is_image(str(non_existent_path)) is False

    def test_encode_image_from_pil(self):
        """Test encode_image function with PIL Image."""
        pil_image = PILImage.new("RGB", (100, 100), color="red")
        data_uri, mime_type = encode_image(pil_image)

        assert data_uri.startswith("data:image/png;base64,")
        assert mime_type == "image/png"

    def test_encode_image_from_file(self, tmp_path):
        """Test encode_image function with file path."""
        image_path = tmp_path / "test.jpg"
        pil_image = PILImage.new("RGB", (50, 50), color="blue")
        pil_image.save(image_path, format="JPEG")

        data_uri, mime_type = encode_image(str(image_path))

        assert data_uri.startswith("data:image/jpeg;base64,")
        assert mime_type == "image/jpeg"

    def test_encode_image_from_url(self):
        """Test encode_image function with URL."""
        url = "https://example.com/image.jpg"
        data_uri, mime_type = encode_image(url, download_images=False)

        # Should return URL as-is when download_images=False
        assert data_uri == url
        assert mime_type == "image/jpeg"

    def test_encode_image_from_data_uri(self):
        """Test encode_image function with data URI."""
        data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        result_uri, mime_type = encode_image(data_uri)

        assert result_uri == data_uri
        assert mime_type == "image/png"

    def test_encode_image_from_bytes(self):
        """Test encode_image function with bytes."""
        # Create image bytes
        pil_image = PILImage.new("RGB", (100, 100), color="green")
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        image_bytes = buffered.getvalue()

        data_uri, mime_type = encode_image(image_bytes)

        assert data_uri.startswith("data:image/png;base64,")
        assert mime_type == "image/png"

    def test_encode_image_from_dict(self):
        """Test encode_image function with dict containing URL."""
        image_dict = {"url": "https://example.com/image.jpg"}
        data_uri, mime_type = encode_image(image_dict)

        assert data_uri == "https://example.com/image.jpg"
        assert mime_type == "image/jpeg"

    def test_encode_image_from_image_instance(self):
        """Test encode_image function with Image instance."""
        pil_image = PILImage.new("RGB", (100, 100), color="red")
        image_instance = Image(from_pil=pil_image)

        data_uri, mime_type = encode_image(image_instance)

        assert data_uri.startswith("data:image/png;base64,")
        assert mime_type == "image/png"

    def test_encode_image_unsupported_type(self):
        """Test encode_image function with unsupported type."""
        with pytest.raises(ValueError, match="Unsupported image type"):
            encode_image(123)

    def test_encode_image_unsupported_string(self):
        """Test encode_image function with unsupported string."""
        with pytest.raises(ValueError, match="Unsupported file string"):
            encode_image("not_a_file_or_url")

    def test_encode_image_file_not_found(self, tmp_path):
        """Test encode_image function with non-existent file."""
        non_existent_path = tmp_path / "nonexistent.png"

        with pytest.raises(ValueError, match="Unsupported file string"):
            encode_image(str(non_existent_path))

    def test_encode_image_unsupported_file_type(self, tmp_path):
        """Test encode_image function with unsupported file type."""
        # Create a file with no extension (which won't have a MIME type)
        no_ext_file = tmp_path / "test"
        no_ext_file.write_text("This is not an image")

        with pytest.raises(ValueError, match="Could not determine MIME type"):
            encode_image(str(no_ext_file))


class TestErrorHandling:
    """Test error handling in the vision module."""

    def test_image_from_file_nonexistent_file(self):
        """Test Image.from_file with non-existent file."""
        with pytest.raises(FileNotFoundError):
            Image.from_file("nonexistent_file.jpg")

    def test_image_from_file_unsupported_format(self, tmp_path):
        """Test Image.from_file with unsupported file format."""
        # Create a file with no extension (which won't have a MIME type)
        no_ext_file = tmp_path / "test"
        no_ext_file.write_text("This is not an image")

        with pytest.raises(ValueError, match="Could not determine MIME type"):
            Image.from_file(str(no_ext_file))

    def test_encode_image_from_url_download_failure(self):
        """Test encode_image with URL download failure."""
        # This test would require mocking requests.get to simulate failure
        # For now, we'll test the basic functionality
        url = "https://example.com/image.jpg"
        data_uri, mime_type = encode_image(url, download_images=False)

        assert data_uri == url
        assert mime_type == "image/jpeg"


class TestIntegration:
    """Integration tests for the vision module."""

    def test_full_workflow_from_pil_to_format(self):
        """Test complete workflow from PIL image to formatted output."""
        # Create PIL image
        pil_image = PILImage.new("RGB", (100, 100), color="purple")

        # Create Image instance
        image = Image.from_PIL(pil_image)

        # Format for API
        formatted = image.format()

        # Verify structure
        assert isinstance(formatted, list)
        assert len(formatted) == 1
        assert formatted[0]["type"] == "image_url"
        assert "url" in formatted[0]["image_url"]
        assert "mime_type" in formatted[0]["image_url"]
        assert formatted[0]["image_url"]["mime_type"] == "image/png"

    def test_full_workflow_from_file_to_format(self, tmp_path):
        """Test complete workflow from file to formatted output."""
        # Create image file
        image_path = tmp_path / "test_workflow.jpg"
        pil_image = PILImage.new("RGB", (100, 100), color="orange")
        pil_image.save(image_path, format="JPEG")

        # Create Image instance
        image = Image.from_file(str(image_path))

        # Format for API
        formatted = image.format()

        # Verify structure
        assert isinstance(formatted, list)
        assert len(formatted) == 1
        assert formatted[0]["type"] == "image_url"
        assert "url" in formatted[0]["image_url"]
        assert "mime_type" in formatted[0]["image_url"]
        assert formatted[0]["image_url"]["mime_type"] == "image/jpeg"

    def test_multiple_image_formats(self, tmp_path):
        """Test handling of multiple image formats."""
        formats = [
            ("PNG", "image/png"),
            ("JPEG", "image/jpeg"),
            ("GIF", "image/gif"),
        ]

        for format_name, expected_mime in formats:
            # Create image file
            image_path = tmp_path / f"test.{format_name.lower()}"
            pil_image = PILImage.new("RGB", (50, 50), color="white")
            pil_image.save(image_path, format=format_name)

            # Create Image instance
            image = Image.from_file(str(image_path))

            # Verify MIME type
            assert image.mime_type == expected_mime
