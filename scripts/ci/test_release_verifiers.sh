#!/usr/bin/env bash
# Regression tests for the release verifiers.
#
# Generated data only — no network, no tracked fixtures, no assertions about
# workflow YAML layout. Every negative case asserts the reason for the
# rejection, so a verifier that breaks in an unrelated way (bad arity, missing
# jq, a typo in a guard) cannot pass by merely exiting nonzero.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
verify_run="$repo_root/scripts/ci/verify_prepare_run.sh"
verify_contents="$repo_root/scripts/ci/verify_package_contents.sh"
changelog_section="$repo_root/scripts/ci/changelog_section.sh"

work_dir="$(mktemp -d)"
trap 'rm -rf "$work_dir"' EXIT

version="0.2.0-beta.1"
commit="0123456789abcdef0123456789abcdef01234567"
other_commit="ffffffffffffffffffffffffffffffffffffffff"
log="$work_dir/verifier.log"
failures=0

report_log() {
  sed 's/^/       /' "$log" >&2
}

expect_ok() {
  local label="$1"
  shift
  if "$@" >"$log" 2>&1; then
    printf 'ok   %s\n' "$label"
  else
    printf 'FAIL %s: expected success\n' "$label" >&2
    report_log
    failures=$((failures + 1))
  fi
}

expect_failure() {
  local label="$1" want="$2"
  shift 2
  if "$@" >"$log" 2>&1; then
    printf 'FAIL %s: expected failure, got success\n' "$label" >&2
    failures=$((failures + 1))
  elif grep -qF -- "$want" "$log"; then
    printf 'ok   %s\n' "$label"
  else
    printf 'FAIL %s: expected message containing: %s\n' "$label" "$want" >&2
    report_log
    failures=$((failures + 1))
  fi
}

# A successful prepare run for $commit, optionally mutated by a jq filter.
write_run() {
  local out="$1" filter="${2:-.}"
  jq -n --arg head_sha "$commit" \
    '{head_sha: $head_sha,
      path: ".github/workflows/release-prepare.yml",
      event: "push",
      status: "completed",
      conclusion: "success"}' |
    jq "$filter" >"$out"
}

# A well-formed numr-<version>/ package tree, optionally missing one item.
# $1: destination directory to build the tree under (removed/recreated).
# $2: label of the item to omit (build.rs | cu | cuh | wgsl | sobol | changelog | readme | license), or empty for none.
build_tree() {
  local root="$1" omit="${2:-}"
  rm -rf "$root"
  local top="$root/numr-${version}"
  mkdir -p "$top/src/runtime/cpu/kernels" "$top/src/runtime/cuda/kernels" "$top/src/runtime/wgpu/shaders"

  test "$omit" = "build.rs" || : >"$top/build.rs"
  test "$omit" = "cu" || : >"$top/src/runtime/cuda/kernels/binary.cu"
  test "$omit" = "cuh" || : >"$top/src/runtime/cuda/kernels/common.cuh"
  test "$omit" = "wgsl" || : >"$top/src/runtime/wgpu/shaders/unary.wgsl"
  test "$omit" = "sobol" || : >"$top/src/runtime/cpu/kernels/sobol_data.bin"
  test "$omit" = "changelog" || : >"$top/CHANGELOG.md"
  test "$omit" = "readme" || : >"$top/README.md"
  test "$omit" = "license" || : >"$top/LICENSE"
}

# Package $1 (a numr-<version>/ parent dir) into crate tarball $2.
make_crate() {
  local root="$1" crate="$2"
  tar czf "$crate" -C "$root" "numr-${version}"
}

# ── prepare-run provenance ──────────────────────────────────────────────────

run_json="$work_dir/run.json"
write_run "$run_json"

expect_ok "prepare run: successful tag build" \
  bash "$verify_run" "$run_json" "$commit"

expect_failure "prepare run: different tag commit" "run head_sha is" \
  bash "$verify_run" "$run_json" "$other_commit"

write_run "$work_dir/failed.json" '.conclusion = "failure"'
expect_failure "prepare run: unsuccessful conclusion" "run conclusion is 'failure'" \
  bash "$verify_run" "$work_dir/failed.json" "$commit"

write_run "$work_dir/in-progress.json" '.status = "in_progress" | .conclusion = null'
expect_failure "prepare run: still running" "run status is 'in_progress'" \
  bash "$verify_run" "$work_dir/in-progress.json" "$commit"

write_run "$work_dir/other-workflow.json" '.path = ".github/workflows/test.yml"'
expect_failure "prepare run: different workflow" "run path is" \
  bash "$verify_run" "$work_dir/other-workflow.json" "$commit"

write_run "$work_dir/dispatched.json" '.event = "workflow_dispatch"'
expect_failure "prepare run: not a tag push" "run event is 'workflow_dispatch'" \
  bash "$verify_run" "$work_dir/dispatched.json" "$commit"

write_run "$work_dir/truncated.json" 'del(.conclusion)'
expect_failure "prepare run: absent field" "run conclusion is '<absent>'" \
  bash "$verify_run" "$work_dir/truncated.json" "$commit"

printf 'not json' >"$work_dir/garbage.json"
expect_failure "prepare run: malformed metadata" "not valid JSON" \
  bash "$verify_run" "$work_dir/garbage.json" "$commit"

expect_failure "prepare run: absent metadata file" "missing workflow-run metadata" \
  bash "$verify_run" "$work_dir/nonexistent.json" "$commit"

expect_failure "prepare run: malformed commit" "invalid release commit" \
  bash "$verify_run" "$run_json" "HEAD"

# ── package contents ────────────────────────────────────────────────────────

pkg_root="$work_dir/pkg"
crate="$work_dir/numr-${version}.crate"

build_tree "$pkg_root"
make_crate "$pkg_root" "$crate"
expect_ok "package contents: well-formed crate" \
  bash "$verify_contents" "$crate"

# The *.cu case is the reason this script exists: a missing CUDA source
# publishes cleanly under default features and only breaks a downstream
# `--features cuda` build, which no runner here compiles to catch.
build_tree "$pkg_root" "cu"
make_crate "$pkg_root" "$crate"
expect_failure "package contents: missing CUDA source (*.cu)" \
  "packaged crate is missing CUDA kernel sources (*.cu)" \
  bash "$verify_contents" "$crate"

build_tree "$pkg_root" "build.rs"
make_crate "$pkg_root" "$crate"
expect_failure "package contents: missing build.rs" \
  "packaged crate is missing build.rs" \
  bash "$verify_contents" "$crate"

build_tree "$pkg_root" "cuh"
make_crate "$pkg_root" "$crate"
expect_failure "package contents: missing CUDA header (*.cuh)" \
  "packaged crate is missing CUDA kernel headers (*.cuh)" \
  bash "$verify_contents" "$crate"

build_tree "$pkg_root" "wgsl"
make_crate "$pkg_root" "$crate"
expect_failure "package contents: missing WGSL shader" \
  "packaged crate is missing WGSL shaders (*.wgsl)" \
  bash "$verify_contents" "$crate"

build_tree "$pkg_root" "sobol"
make_crate "$pkg_root" "$crate"
expect_failure "package contents: missing sobol_data.bin" \
  "packaged crate is missing sobol_data.bin" \
  bash "$verify_contents" "$crate"

build_tree "$pkg_root" "changelog"
make_crate "$pkg_root" "$crate"
expect_failure "package contents: missing CHANGELOG.md" \
  "packaged crate is missing CHANGELOG.md" \
  bash "$verify_contents" "$crate"

build_tree "$pkg_root" "readme"
make_crate "$pkg_root" "$crate"
expect_failure "package contents: missing README.md" \
  "packaged crate is missing README.md" \
  bash "$verify_contents" "$crate"

build_tree "$pkg_root" "license"
make_crate "$pkg_root" "$crate"
expect_failure "package contents: missing LICENSE" \
  "packaged crate is missing LICENSE" \
  bash "$verify_contents" "$crate"

expect_failure "package contents: absent tarball" "crate tarball not found" \
  bash "$verify_contents" "$work_dir/nonexistent.crate"

# ── changelog section extraction ────────────────────────────────────────────
#
# changelog_section.sh reads CHANGELOG.md from the current directory, so each
# case runs in a subshell `cd`ed into its own scratch directory — the
# harness's own cwd is never disturbed.

changelog_dir="$work_dir/changelog"
mkdir -p "$changelog_dir/basic"

cat >"$changelog_dir/basic/CHANGELOG.md" <<'EOF'
# Changelog

## [1.2.3]

Added a thing.
Fixed another thing.
EOF

expect_ok "changelog: success extracts version section" \
  bash -c 'cd "$1" && bash "$2" 1.2.3 "$1/out.md"' _ "$changelog_dir/basic" "$changelog_section"
grep -qF "Added a thing." "$changelog_dir/basic/out.md" ||
  { printf 'FAIL changelog: success extracts version section: body missing from output\n' >&2; failures=$((failures + 1)); }

mkdir -p "$changelog_dir/two-sections"
cat >"$changelog_dir/two-sections/CHANGELOG.md" <<'EOF'
# Changelog

## [1.2.3]

Current release notes.

## [1.2.2]

Older release notes that must not leak into the newer section.
EOF

# Extraction must stop at the next `## [` heading — the case with the most
# real risk of regression (an off-by-one in the awk state machine would pull
# the older section's text into the newer one).
expect_ok "changelog: stops at next version heading" \
  bash -c 'cd "$1" && bash "$2" 1.2.3 "$1/out.md"' _ "$changelog_dir/two-sections" "$changelog_section"
if grep -qF "Older release notes" "$changelog_dir/two-sections/out.md"; then
  printf 'FAIL changelog: stops at next version heading: next section leaked into output\n' >&2
  failures=$((failures + 1))
else
  printf 'ok   changelog: stops at next version heading (no leak)\n'
fi

mkdir -p "$changelog_dir/trailing"
printf '# Changelog\n\n## [1.2.3]\n\nRelease notes.\n\n\n---\n\n' >"$changelog_dir/trailing/CHANGELOG.md"

# A section always opens with the blank line between the heading and its
# body, so only the LAST line is checked here — that is what the trailing
# blank-line/`---`-rule stripping loop actually promises to remove.
expect_ok "changelog: trailing blanks and --- rule stripped" \
  bash -c 'cd "$1" && bash "$2" 1.2.3 "$1/out.md"' _ "$changelog_dir/trailing" "$changelog_section"
last_line="$(tail -n 1 "$changelog_dir/trailing/out.md")"
if [[ "$last_line" =~ ^[[:space:]]*$ ]] || [[ "$last_line" =~ ^-{3,}[[:space:]]*$ ]]; then
  printf 'FAIL changelog: trailing blanks and --- rule stripped: furniture remained in output\n' >&2
  failures=$((failures + 1))
else
  printf 'ok   changelog: trailing blanks and --- rule stripped\n'
fi

mkdir -p "$changelog_dir/missing-version"
cat >"$changelog_dir/missing-version/CHANGELOG.md" <<'EOF'
# Changelog

## [1.0.0]

Initial release.
EOF

expect_failure "changelog: no section for requested version" \
  "has no non-empty '## [9.9.9]' section" \
  bash -c 'cd "$1" && bash "$2" 9.9.9 "$1/out.md"' _ "$changelog_dir/missing-version" "$changelog_section"

mkdir -p "$changelog_dir/empty-section"
cat >"$changelog_dir/empty-section/CHANGELOG.md" <<'EOF'
# Changelog

## [1.2.3]

## [1.2.2]

Older notes.
EOF

expect_failure "changelog: section present but empty" \
  "has no non-empty '## [1.2.3]' section" \
  bash -c 'cd "$1" && bash "$2" 1.2.3 "$1/out.md"' _ "$changelog_dir/empty-section" "$changelog_section"

mkdir -p "$changelog_dir/no-file"
expect_failure "changelog: no CHANGELOG.md at all" \
  "CHANGELOG.md is required to cut a release" \
  bash -c 'cd "$1" && bash "$2" 1.2.3 "$1/out.md"' _ "$changelog_dir/no-file" "$changelog_section"

# ────────────────────────────────────────────────────────────────────────────

if test "$failures" -ne 0; then
  printf 'release verifier tests: %d test(s) failed\n' "$failures" >&2
  exit 1
fi

printf 'release verifier tests: passed\n'
