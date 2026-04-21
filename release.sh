#!/usr/bin/env bash
set -euo pipefail
BUMP_TYPE="${1:-patch}"
CURRENT_VERSION=$(grep -m1 'version = ' pyproject.toml | cut -d'"' -f2)  # noqa: shell var
echo "🔍 当前版本: $CURRENT_VERSION | 升级: $BUMP_TYPE"
IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT_VERSION"
case "$BUMP_TYPE" in
  major) MAJOR=$((MAJOR+1)); MINOR=0; PATCH=0 ;;
  minor) MINOR=$((MINOR+1)); PATCH=0 ;;
  patch) PATCH=$((PATCH+1)) ;;
esac
NEW_VERSION="$MAJOR.$MINOR.$PATCH"
sed -i.bak "s/version = \"$CURRENT_VERSION\"/version = \"$NEW_VERSION\"/" pyproject.toml && rm -f pyproject.toml.bak
git add pyproject.toml
git commit -m "chore: release v$NEW_VERSION"
git tag -a "v$NEW_VERSION" -m "Release v$NEW_VERSION"
git push origin main --tags
echo "✅ 发布完成: v$NEW_VERSION"
