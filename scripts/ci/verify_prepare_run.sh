#!/usr/bin/env bash
# Gate the maintainer-supplied Release Prepare run before any of its artifacts
# are downloaded. The run must be a successful, tag-pushed execution of the
# release prepare workflow for the exact commit the release tag resolves to.
#
# Provenance only. Nothing here inspects the crate tarball; that is checked by
# verify_package_contents.sh during the prepare run itself, before it is ever
# uploaded. The tarball is trusted on the strength of this gate — a run pinned
# to the tag commit, the prepare workflow file, and a successful conclusion —
# so there is deliberately no digest comparison anywhere in the release path.

set -euo pipefail

usage='usage: verify_prepare_run.sh <run.json> <commit>'

die() {
  printf 'prepare run: %s\n' "$*" >&2
  exit 1
}

run_json="${1:?$usage}"
commit="${2:?$usage}"

test -f "$run_json" || die "missing workflow-run metadata: $run_json"
[[ "$commit" =~ ^[0-9a-f]{40}$ ]] || die "invalid release commit: $commit"
jq -e . "$run_json" >/dev/null 2>&1 ||
  die "workflow-run metadata is not valid JSON: $run_json"

# Checked field-by-field rather than as one boolean so a rejection names the
# field that disagreed — the maintainer needs to know whether they picked the
# wrong run, the wrong tag, or a run that has not finished.
fields=(head_sha path event status conclusion)
expected=(
  "$commit"
  ".github/workflows/release-prepare.yml"
  push
  completed
  success
)

index=0
while IFS= read -r actual; do
  test "$index" -lt "${#fields[@]}" || die "unexpected workflow-run metadata shape"
  test "$actual" = "${expected[$index]}" ||
    die "run ${fields[$index]} is '$actual', expected '${expected[$index]}'"
  index=$((index + 1))
done < <(
  jq -r '[.head_sha, .path, .event, .status, .conclusion]
         | map(if . == null then "<absent>" else tostring end)
         | .[]' "$run_json"
)

test "$index" -eq "${#fields[@]}" || die "incomplete workflow-run metadata: $run_json"

printf 'prepare run: ok\n'
