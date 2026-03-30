"""src/compat.py 유닛 테스트"""
import os
import sys
import pytest
from src.compat import (
    normalize_path, safe_realpath, path_is_under,
    fix_encoding_name, get_file_extension, find_executable,
    get_subprocess_kwargs, should_unload_model,
)


class TestNormalizePath:
    def test_basic_string(self):
        assert normalize_path("/some/path/file.txt") == "/some/path/file.txt"

    def test_pathlib_input(self):
        from pathlib import Path
        result = normalize_path(Path("/some/path"))
        assert isinstance(result, str)

    def test_korean_filename(self):
        result = normalize_path("/경로/한글파일.txt")
        assert "한글파일" in result


class TestPathIsUnder:
    def test_child_under_parent(self, tmp_path):
        parent = str(tmp_path)
        child_dir = tmp_path / "sub"
        child_dir.mkdir()
        child_file = child_dir / "file.txt"
        child_file.touch()
        assert path_is_under(str(child_file), parent) is True

    def test_child_outside_parent(self, tmp_path):
        import tempfile
        other = tempfile.mkdtemp()
        child = os.path.join(other, "file.txt")
        open(child, 'w').close()
        assert path_is_under(child, str(tmp_path)) is False


class TestFixEncodingName:
    def test_euc_kr_to_cp949(self):
        assert fix_encoding_name("EUC-KR") == "cp949"
        assert fix_encoding_name("euc-kr") == "cp949"

    def test_utf8_passthrough(self):
        assert fix_encoding_name("utf-8") == "utf-8"

    def test_none_fallback(self):
        assert fix_encoding_name(None) == "utf-8"
        assert fix_encoding_name("") == "utf-8"


class TestGetFileExtension:
    def test_normal(self):
        assert get_file_extension("report.pdf") == "pdf"

    def test_dotless(self):
        assert get_file_extension("README") == ""

    def test_dotfile(self):
        assert get_file_extension(".gitignore") == "gitignore"

    def test_multiple_dots(self):
        assert get_file_extension("archive.tar.gz") == "gz"

    def test_uppercase(self):
        assert get_file_extension("REPORT.PDF") == "pdf"


class TestShouldUnloadModel:
    def test_mps(self):
        assert should_unload_model("mps") is True

    def test_cpu(self):
        assert should_unload_model("cpu") is False

    def test_cuda(self):
        assert should_unload_model("cuda") is False


class TestGetSubprocessKwargs:
    def test_returns_dict(self):
        result = get_subprocess_kwargs()
        assert isinstance(result, dict)
        if sys.platform == 'win32':
            assert 'creationflags' in result
        else:
            assert result == {}
