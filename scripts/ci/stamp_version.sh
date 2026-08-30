#!/usr/bin/env bash
# Stamp the package version from a release tag.
#
#   scripts/ci/stamp_version.sh <version>       # e.g. 0.7.0, 0.7.0-beta.2
#
# Rewrites `[package] version` in Cargo.toml. numr is a single leaf crate with
# no internal path deps, so there is nothing else to pin.
#
# No-ops when Cargo.toml already carries the target version, which keeps
# re-running a stage idempotent.

set -euo pipefail

VERSION="${1:?usage: stamp_version.sh <version>}"

CURRENT=$(cargo metadata --no-deps --format-version=1 \
    | jq -r '.packages[] | select(.name == "numr") | .version')

if [[ "$VERSION" == "$CURRENT" ]]; then
    echo "Version already $VERSION — nothing to stamp."
    exit 0
fi

# First `version = "..."` in the file is [package].
perl -i -pe 'if (!$done && /^version = "/) { s/^version = ".*"/version = "'"$VERSION"'"/; $done=1 }' Cargo.toml

echo "Stamped package version: $CURRENT -> $VERSION"
