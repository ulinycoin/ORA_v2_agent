"""Unit tests: SSRF validation in fetch_url."""
import pytest
from ora_v2.skills.builtins.fetch_url import _validate_url


def test_blocks_localhost():
    with pytest.raises(ValueError, match="Blocked host"):
        _validate_url("http://localhost/admin")


def test_blocks_127():
    with pytest.raises(ValueError, match="Blocked host"):
        _validate_url("http://127.0.0.1:8080/secret")


def test_blocks_169_254():
    with pytest.raises(ValueError, match="Blocked host"):
        _validate_url("http://169.254.169.254/latest/meta-data/")


def test_blocks_10_x():
    with pytest.raises(ValueError, match="Blocked host"):
        _validate_url("http://10.0.0.1/internal")


def test_blocks_192_168():
    with pytest.raises(ValueError, match="Blocked host"):
        _validate_url("http://192.168.1.1/router")


def test_blocks_172_16():
    with pytest.raises(ValueError, match="Blocked host"):
        _validate_url("http://172.16.0.1/")


def test_blocks_ftp_scheme():
    with pytest.raises(ValueError, match="Unsupported scheme"):
        _validate_url("ftp://example.com/file.txt")


def test_allows_https():
    _validate_url("https://example.com/page")


def test_allows_http():
    _validate_url("http://openai.com/docs")
