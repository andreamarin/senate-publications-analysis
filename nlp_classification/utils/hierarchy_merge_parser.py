"""Parse BERTopic hierarchical topic tree text and build merge topic groups.

Expects the ASCII tree format produced by BERTopic / manual review, e.g.::

    ├─parent_label >>>>> MERGE
    │    ├─■──child_terms ── Topic: 12
    │    └─■──other_terms ── Topic: 34

Each line marked with ``>>>>> MERGE`` defines one merge group: every
``Topic: <id>`` leaf in that node's subtree is collected into the same sublist.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TextIO

MERGE_MARKER = ">>>>> MERGE"
BRANCH_PATTERN = re.compile(r"([├└])─\s*(.*)$")
TOPIC_PATTERN = re.compile(r"Topic:\s*(-?\d+)")
# BERTopic trees indent one level per 5 columns (``│    `` or ``     ``).
INDENT_STEP = 5


@dataclass
class _TreeNode:
    depth: int
    label: str
    is_merge: bool
    topics: list[int]


def _line_depth(line: str) -> int | None:
    """Return 1-based tree depth, or None for blank / root-only lines."""
    stripped = line.strip()
    if not stripped or stripped == ".":
        return None

    match = BRANCH_PATTERN.search(line)
    if not match:
        return None

    # Use horizontal position of the branch marker, not only ``│`` count.
    # BERTopic mixes ``│    `` prefixes with space-only ``     `` indentation.
    return match.start() // INDENT_STEP + 1


def _parse_tree_line(line: str) -> tuple[int, str, bool, int | None] | None:
    """Parse one tree line into depth, label, merge flag, and optional topic id."""
    depth = _line_depth(line)
    if depth is None:
        return None

    match = BRANCH_PATTERN.search(line)
    assert match is not None
    rest = match.group(2).strip()

    is_merge = MERGE_MARKER in rest
    if is_merge:
        rest = rest.split(MERGE_MARKER, 1)[0].strip()

    label = rest.removeprefix("■──").strip()
    if not label and "Topic:" in match.group(2):
        label = match.group(2).split("──", 1)[-1].split("Topic:")[0].strip()

    topic_match = TOPIC_PATTERN.search(line)
    topic_id = int(topic_match.group(1)) if topic_match else None
    return depth, label, is_merge, topic_id


def build_merge_topic_list_from_tree(
    tree: str,
    *,
    include_nested: bool = True,
) -> list[list[int]]:
    """Build merge groups from a hierarchical topic tree string.

    Parameters
    ----------
    tree
        Full multi-line ASCII tree text.
    include_nested
        When True (default), each ``>>>>> MERGE`` node yields its own sublist,
        even if it is nested under another merge node. When False, a topic is
        assigned only to the nearest merge ancestor.

    Returns
    -------
    list[list[int]]
        One sublist per merge node (pre-order), each containing topic ids to
        merge together.
    """
    groups = build_merge_groups_from_tree(tree, include_nested=include_nested)
    return [group.topics for group in groups]


@dataclass
class MergeTopicGroup:
    """Topics to merge for one ``>>>>> MERGE`` node."""

    label: str
    depth: int
    topics: list[int]


def build_merge_groups_from_tree(
    tree: str,
    *,
    include_nested: bool = True,
) -> list[MergeTopicGroup]:
    """Like :func:`build_merge_topic_list_from_tree` but keeps merge labels."""
    stack: list[_TreeNode] = []
    groups: list[MergeTopicGroup] = []

    for raw_line in tree.splitlines():
        parsed = _parse_tree_line(raw_line)
        if parsed is None:
            continue

        depth, label, is_merge, topic_id = parsed

        while stack and stack[-1].depth >= depth:
            stack.pop()

        if topic_id is not None:
            if include_nested:
                for ancestor in stack:
                    if ancestor.is_merge and topic_id not in ancestor.topics:
                        ancestor.topics.append(topic_id)
            else:
                for ancestor in reversed(stack):
                    if ancestor.is_merge:
                        if topic_id not in ancestor.topics:
                            ancestor.topics.append(topic_id)
                        break

        node = _TreeNode(
            depth=depth,
            label=label,
            is_merge=is_merge,
            topics=[topic_id] if topic_id is not None and is_merge else [],
        )
        if is_merge:
            groups.append(
                MergeTopicGroup(label=label, depth=depth, topics=node.topics)
            )
        stack.append(node)

    return [g for g in groups if g.topics]


def build_merge_topic_list_from_file(
    path: str,
    *,
    include_nested: bool = True,
) -> list[list[int]]:
    """Load tree text from a file and return merge topic groups."""
    with open(path, encoding="utf-8") as fp:
        return build_merge_topic_list_from_tree(
            fp.read(),
            include_nested=include_nested,
        )


def print_merge_topic_list(
    tree: str | TextIO,
    *,
    include_nested: bool = True,
) -> None:
    """Print a Python-ready ``merge_topic_list`` literal to stdout."""
    if not isinstance(tree, str):
        tree = tree.read()

    groups = build_merge_groups_from_tree(tree, include_nested=include_nested)
    print("merge_topic_list = [")
    for group in groups:
        comment = group.label.replace('"', '\\"')
        print(f"    {group.topics},  # {comment}")
    print("]")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            "Usage: python -m nlp_classification.utils.hierarchy_merge_parser "
            "<hierarchy_tree.txt>",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print_merge_topic_list(open(sys.argv[1], encoding="utf-8"))
