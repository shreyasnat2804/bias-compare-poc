"""Tests for biascompare.inputs: ModelSpec, Post, load_posts."""

from __future__ import annotations

import json

import pytest

from biascompare.inputs import ModelSpec, Post, load_posts

# ---------------------------------------------------------------------------
# ModelSpec
# ---------------------------------------------------------------------------


def test_model_spec_defaults_and_label():
    spec = ModelSpec(name="unitary/toxic-bert")
    assert spec.toxic_label == "toxic"
    assert spec.label == "toxic-bert"


def test_model_spec_label_is_last_path_component():
    assert ModelSpec(name="s-nlp/roberta_toxicity_classifier").label == "roberta_toxicity_classifier"
    assert ModelSpec(name="just-a-name").label == "just-a-name"


def test_model_spec_rejects_empty_name():
    with pytest.raises(ValueError):
        ModelSpec(name="")


def test_model_spec_rejects_whitespace_only_name():
    with pytest.raises(ValueError):
        ModelSpec(name="   ")


def test_model_spec_rejects_empty_toxic_label():
    with pytest.raises(ValueError):
        ModelSpec(name="unitary/toxic-bert", toxic_label="")


def test_model_spec_resolved_token_prefers_explicit_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "env-token")
    spec = ModelSpec(name="m", token="explicit-token")
    assert spec.resolved_token() == "explicit-token"


def test_model_spec_resolved_token_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "env-token")
    spec = ModelSpec(name="m")
    assert spec.resolved_token() == "env-token"


def test_model_spec_resolved_token_none_when_absent(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    spec = ModelSpec(name="m")
    assert spec.resolved_token() is None


def test_model_spec_to_public_dict_never_includes_token():
    spec = ModelSpec(name="unitary/toxic-bert", toxic_label="toxic", token="super-secret")
    public = spec.to_public_dict()
    assert public == {"name": "unitary/toxic-bert", "toxic_label": "toxic"}
    assert "token" not in public
    assert "super-secret" not in json.dumps(public)


def test_model_spec_repr_never_includes_token():
    spec = ModelSpec(name="unitary/toxic-bert", token="super-secret")
    assert "super-secret" not in repr(spec)


def test_model_spec_equality_ignores_token():
    a = ModelSpec(name="m", toxic_label="toxic", token="token-a")
    b = ModelSpec(name="m", toxic_label="toxic", token="token-b")
    assert a == b


# ---------------------------------------------------------------------------
# Post
# ---------------------------------------------------------------------------


def test_post_valid_construction():
    post = Post.from_legacy(id="p01", platform="twitter", text="hello", gender="female")
    assert post.intended_toxicity is None


def test_post_rejects_invalid_gender():
    with pytest.raises(ValueError):
        Post.from_legacy(id="p01", platform="twitter", text="hello", gender="other")


def test_post_rejects_empty_id():
    with pytest.raises(ValueError):
        Post.from_legacy(id="", platform="twitter", text="hello", gender="male")


def test_post_rejects_empty_text():
    with pytest.raises(ValueError):
        Post.from_legacy(id="p01", platform="twitter", text="", gender="male")


def test_post_rejects_empty_platform():
    with pytest.raises(ValueError):
        Post.from_legacy(id="p01", platform="", text="hello", gender="male")


# ---------------------------------------------------------------------------
# load_posts
# ---------------------------------------------------------------------------


def _write_jsonl(path, records) -> None:
    lines = []
    for record in records:
        if isinstance(record, str):
            lines.append(record)
        else:
            lines.append(json.dumps(record))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_posts_parses_valid_file(tmp_path):
    path = tmp_path / "posts.jsonl"
    _write_jsonl(
        path,
        [
            {"id": "p01", "platform": "twitter", "text": "hi", "gender": "female"},
            {"id": "p02", "platform": "reddit", "text": "hey", "gender": "male", "intended_toxicity": "clean"},
        ],
    )
    posts = load_posts(path)
    assert isinstance(posts, tuple)
    assert len(posts) == 2
    assert posts[0].id == "p01"
    assert posts[1].intended_toxicity == "clean"


def test_load_posts_skips_blank_lines(tmp_path):
    path = tmp_path / "posts.jsonl"
    path.write_text(
        '\n{"id": "p01", "platform": "twitter", "text": "hi", "gender": "female"}\n\n   \n',
        encoding="utf-8",
    )
    posts = load_posts(path)
    assert len(posts) == 1


def test_load_posts_invalid_json_raises_with_line_number(tmp_path):
    path = tmp_path / "posts.jsonl"
    _write_jsonl(
        path,
        [
            {"id": "p01", "platform": "twitter", "text": "hi", "gender": "female"},
            "{not valid json",
        ],
    )
    with pytest.raises(ValueError, match="line 2"):
        load_posts(path)


def test_load_posts_missing_required_field_raises(tmp_path):
    path = tmp_path / "posts.jsonl"
    _write_jsonl(path, [{"id": "p01", "platform": "twitter", "gender": "female"}])
    with pytest.raises(ValueError, match="missing required field"):
        load_posts(path)


def test_load_posts_invalid_gender_raises(tmp_path):
    path = tmp_path / "posts.jsonl"
    _write_jsonl(path, [{"id": "p01", "platform": "twitter", "text": "hi", "gender": "nonbinary"}])
    with pytest.raises(ValueError):
        load_posts(path)


def test_load_posts_duplicate_ids_raise(tmp_path):
    path = tmp_path / "posts.jsonl"
    _write_jsonl(
        path,
        [
            {"id": "p01", "platform": "twitter", "text": "hi", "gender": "female"},
            {"id": "p01", "platform": "reddit", "text": "hey", "gender": "male"},
        ],
    )
    with pytest.raises(ValueError, match="Duplicate post id"):
        load_posts(path)


def test_load_posts_empty_file_raises(tmp_path):
    path = tmp_path / "posts.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        load_posts(path)


def test_load_posts_blank_only_file_raises(tmp_path):
    path = tmp_path / "posts.jsonl"
    path.write_text("\n\n   \n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_posts(path)


def test_load_posts_missing_file_raises():
    with pytest.raises(ValueError):
        load_posts("/no/such/file.jsonl")


def test_load_posts_non_object_line_raises(tmp_path):
    path = tmp_path / "posts.jsonl"
    _write_jsonl(path, [["not", "an", "object"]])
    with pytest.raises(ValueError, match="JSON object"):
        load_posts(path)


# ---------------------------------------------------------------------------
# v1 rows (docs/DATA.md §2.3)
# ---------------------------------------------------------------------------


def _v1(**overrides):
    record = {"id": "r1", "text": "He said hello.", "attributes": {"gender": "male"}}
    record.update(overrides)
    return record


def test_load_posts_parses_v1_rows(tmp_path):
    path = tmp_path / "rows.jsonl"
    _write_jsonl(path, [_v1(label="toxic", meta={"platform": "reddit", "source": "pilot"})])
    (post,) = load_posts(path)
    assert post.attributes == {"gender": "male"}
    assert post.gender == "male" and post.platform == "reddit"
    assert post.label == "toxic"
    assert post.meta["source"] == "pilot"
    assert post.lang == "en"


def test_load_posts_accepts_mixed_v1_and_legacy_rows(tmp_path):
    path = tmp_path / "rows.jsonl"
    legacy = {"id": "old", "platform": "twitter", "text": "She waved.", "gender": "female", "extra": 1}
    _write_jsonl(path, [_v1(), legacy])
    v1_post, legacy_post = load_posts(path)
    assert v1_post.platform is None
    assert legacy_post.attributes == {"gender": "female"}
    assert legacy_post.meta == {"platform": "twitter", "intended_toxicity": None, "extra": 1}


def test_load_posts_v1_rejects_unknown_fields_and_points_to_meta(tmp_path):
    path = tmp_path / "rows.jsonl"
    _write_jsonl(path, [_v1(atributes={"gender": "male"})])
    with pytest.raises(ValueError, match=r"Line 1 .*unknown field\(s\) \['atributes'\].*'meta'"):
        load_posts(path)


def test_load_posts_v1_requires_the_axis_attribute(tmp_path):
    path = tmp_path / "rows.jsonl"
    _write_jsonl(path, [_v1(attributes={"race": "x"})])
    with pytest.raises(ValueError, match="no 'gender' entry in attributes"):
        load_posts(path)


def test_load_posts_v1_checks_the_group_against_the_lexicon(tmp_path):
    path = tmp_path / "rows.jsonl"
    _write_jsonl(path, [_v1(attributes={"gender": "nonbinary"})])
    with pytest.raises(ValueError, match=r"must be one of \['male', 'female'\]"):
        load_posts(path)


def test_load_posts_v1_rejects_unsupported_language(tmp_path):
    path = tmp_path / "rows.jsonl"
    _write_jsonl(path, [_v1(lang="de")])
    with pytest.raises(ValueError, match="only English"):
        load_posts(path)


def test_load_posts_unknown_axis_raises(tmp_path):
    path = tmp_path / "rows.jsonl"
    _write_jsonl(path, [_v1()])
    with pytest.raises(ValueError, match="Unknown axis 'age'"):
        load_posts(path, axis="age")


def test_post_mappings_are_read_only_copies():
    attributes = {"gender": "male"}
    post = Post(id="r1", text="He said hi.", attributes=attributes)
    attributes["gender"] = "female"
    assert post.gender == "male"
    with pytest.raises(TypeError):
        post.attributes["gender"] = "female"
    assert hash(post) == hash(Post(id="r1", text="He said hi.", attributes={"gender": "male"}))


# ---------------------------------------------------------------------------
# ModelSpec: type, label_map, revision, token_env, settings
# ---------------------------------------------------------------------------


def test_model_spec_new_fields_round_trip_into_public_dict():
    spec = ModelSpec(
        name="org/m", label_map={"toxic": "hate", "borderline": ["offensive"]}, revision="abc",
        settings={"batch_size": 8}, token="secret",
    )
    assert spec.label_map == {"toxic": ("hate",), "borderline": ("offensive",)}
    assert spec.to_public_dict() == {
        "name": "org/m", "toxic_label": "toxic",
        "label_map": {"toxic": ["hate"], "borderline": ["offensive"]},
        "revision": "abc", "settings": {"batch_size": 8},
    }
    assert "secret" not in json.dumps(spec.to_public_dict())
    assert hash(spec) == hash(ModelSpec(
        name="org/m", label_map={"toxic": "hate", "borderline": ["offensive"]}, revision="abc",
        settings={"batch_size": 8},
    ))


def test_model_spec_token_env_takes_precedence_over_hf_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "generic")
    monkeypatch.setenv("TEAM_TOKEN", "team")
    assert ModelSpec(name="m", token_env="TEAM_TOKEN").resolved_token() == "team"
    monkeypatch.delenv("TEAM_TOKEN")
    assert ModelSpec(name="m", token_env="TEAM_TOKEN").resolved_token() is None


def test_model_spec_rejects_bad_label_map_and_settings():
    with pytest.raises(ValueError, match="label_map"):
        ModelSpec(name="m", label_map={})
    with pytest.raises(ValueError, match="settings"):
        ModelSpec(name="m", settings={"layers": [1, 2]})
