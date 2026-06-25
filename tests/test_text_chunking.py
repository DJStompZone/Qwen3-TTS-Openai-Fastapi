# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""Tests for long-input TTS chunking."""

from api.services.text_processing import split_text_for_tts


def test_split_text_for_tts_keeps_short_input_unchanged():
    assert split_text_for_tts("Hello world.", min_chars=20, max_chars=70) == ["Hello world."]


def test_split_text_for_tts_prefers_sentence_boundaries():
    text = "This is the first sentence. This is the second sentence. This is the third sentence."

    chunks = split_text_for_tts(text, min_chars=20, max_chars=35)

    assert chunks == [
        "This is the first sentence.",
        "This is the second sentence.",
        "This is the third sentence.",
    ]
    assert all(len(chunk) <= 35 for chunk in chunks)


def test_split_text_for_tts_falls_back_to_word_boundaries():
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"

    chunks = split_text_for_tts(text, min_chars=10, max_chars=20)

    assert " ".join(chunks) == text
    assert all(len(chunk) <= 20 for chunk in chunks)
