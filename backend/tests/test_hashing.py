import hashlib

from app.services.hashing import is_valid_sha256, normalize_sha256, sha256_bytes, sha256_file

ABC = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
EMPTY = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_known_vectors():
    assert sha256_bytes(b"abc") == ABC
    assert sha256_bytes(b"") == EMPTY


def test_file_hash_matches_bytes_hash(tmp_path):
    data = b"some apk-like bytes" * 1000
    path = tmp_path / "a.bin"
    path.write_bytes(data)
    assert sha256_file(path) == sha256_bytes(data) == hashlib.sha256(data).hexdigest()


def test_hash_is_independent_of_chunk_size(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(bytes(range(256)) * 50)
    assert sha256_file(path, chunk_size=1) == sha256_file(path, chunk_size=7) == sha256_file(path)


def test_hash_is_stable_across_calls(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"stable")
    assert sha256_file(path) == sha256_file(path)


def test_different_content_gives_different_hash(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"one")
    b.write_bytes(b"two")
    assert sha256_file(a) != sha256_file(b)


def test_empty_file_hash(tmp_path):
    path = tmp_path / "empty"
    path.write_bytes(b"")
    assert sha256_file(path) == EMPTY


def test_validation_helpers():
    assert is_valid_sha256(ABC)
    assert not is_valid_sha256(ABC.upper())  # strict: lowercase only
    assert not is_valid_sha256(ABC[:-1])
    assert not is_valid_sha256("g" * 64)
    assert not is_valid_sha256(None)
    assert normalize_sha256("  " + ABC.upper() + " ") == ABC
    assert normalize_sha256("nope") is None
    assert normalize_sha256(123) is None
