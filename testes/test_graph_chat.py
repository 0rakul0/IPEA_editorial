from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from editorial_docx.graph_chat import _parse_comments


def test_parse_comments_accepts_json_fenced_block():
    raw = """```json
    [
      {
        "category": "gramatica_ortografia",
        "message": "Erro de crase",
        "paragraph_index": 1,
        "issue_excerpt": "a nivel",
        "suggested_fix": "em nível"
      }
    ]
    ```"""

    comments = _parse_comments(raw, agent="gramatica_ortografia")

    assert len(comments) == 1
    assert comments[0].message == "Erro de crase"
    assert comments[0].paragraph_index == 1


def test_parse_comments_accepts_wrapped_comments_key():
    raw = """
    {
      "comments": [
        {
          "category": "gramatica_ortografia",
          "message": "Ajustar concordância",
          "paragraph_index": 2
        }
      ]
    }
    """

    comments = _parse_comments(raw, agent="gramatica_ortografia")

    assert len(comments) == 1
    assert comments[0].message == "Ajustar concordância"
    assert comments[0].paragraph_index == 2


def test_parse_comments_returns_empty_list_for_empty_payload():
    assert _parse_comments("", agent="gramatica_ortografia") == []
