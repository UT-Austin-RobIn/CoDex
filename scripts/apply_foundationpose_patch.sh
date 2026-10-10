#!/usr/bin/env bash
# Ensure the vendored FoundationPose fixes are present without overwriting a
# developer's tracked changes. Called by install.sh before installing the fork.
set -e

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <checkout> <base-commit> <patched-commit> <patch>" >&2
    exit 2
fi

checkout="$1"
base_commit="$2"
patched_commit="$3"
patch="$4"
head_commit="$(git -C "$checkout" rev-parse HEAD)"
tracked_edits="$(git -C "$checkout" status --porcelain --untracked-files=no)"

if [ "$head_commit" = "$patched_commit" ]; then
    if [ -n "$tracked_edits" ]; then
        echo "ERROR: FoundationPose has tracked edits on the patched commit; inspect them first." >&2
        exit 1
    fi
    echo "FoundationPose memory fixes are already committed locally."
elif [ "$head_commit" = "$base_commit" ]; then
    if [ -z "$tracked_edits" ]; then
        git -C "$checkout" apply --check "$patch"
        git -C "$checkout" apply "$patch"
        echo "Applied FoundationPose memory fixes to pinned public commit."
    elif git -C "$checkout" diff HEAD | cmp -s "$patch" -; then
        echo "FoundationPose memory fixes are already applied."
    else
        echo "ERROR: FoundationPose has tracked edits that do not exactly match the release patch." >&2
        exit 1
    fi
else
    echo "ERROR: FoundationPose checkout is at $head_commit." >&2
    echo "Expected public base $base_commit or patched commit $patched_commit." >&2
    exit 1
fi
