# Release Process for unxt Workspace

This workspace contains three packages that can be released:

- `unxt` - the main package
- `unxt-api` - abstract dispatch API
- `unxt-hypothesis` - hypothesis testing strategies

## Versioning Strategy

All packages use **hatch-vcs** for automatic version detection from git tags, with automated tag creation for synchronized releases.

### Tag Format and Auto-Creation

**Coordinator Tags (for synchronized releases):**

- Push a shared tag: `vX.Y.0` (e.g., `v1.8.0`)
- CI automatically creates package-specific tags:
  - `unxt-vX.Y.0`
  - `unxt-api-vX.Y.0`
  - `unxt-hypothesis-vX.Y.0`
- All three packages get version X.Y.0
- **Must be `.0` releases** (major/minor only)

**Package-Specific Tags (for bug-fix releases):**

- Manually create: `PACKAGE-vX.Y.Z` where Z > 0 (e.g., `unxt-api-v1.8.1`)
- Only that package is released
- Other packages are unaffected
- Example: Tag `unxt-api-v1.8.1` releases only unxt-api

### Validation Rules

- ✅ `vX.Y.0` - **Coordinator tag**: Auto-creates package tags for synchronized release
- ✅ `package-vX.Y.0` - **Package .0 release**: Must have corresponding `vX.Y.0` coordinator tag
- ✅ `package-vX.Y.Z` (Z > 0) - **Bug-fix release**: Independent package update
- ❌ `vX.Y.Z` (Z > 0) - **Forbidden**: Bug-fixes must use package-specific tags

This ensures synchronized major/minor releases while allowing independent bug-fixes.

## How Versioning Works

When you push a tag:

1. **If you push `vX.Y.0`** (coordinator tag):
   - CI validates it's a `.0` release
   - CI creates `unxt-vX.Y.0`, `unxt-api-vX.Y.0`, `unxt-hypothesis-vX.Y.0`
   - All package CD workflows trigger and build version X.Y.0

2. **If you push `package-vX.Y.Z`** (package-specific tag):
   - CI validates the tag format
   - For `.0` releases: verifies `vX.Y.0` coordinator tag exists
   - Only that package's CD workflow triggers
   - Package builds with version X.Y.Z

3. **Development versions** are created automatically between tags (e.g., `1.8.1.dev5+gabc1234`)

### Legacy Support

Tags from v1.10.x and earlier are grandfathered in and allowed without validation. Strict enforcement begins at v1.11.0.

## Release Workflows

### Scenario 1: Major/Minor Release (Synchronized All Packages)

Release all packages together with version `X.Y.0`:

```bash
# Create and push coordinator tag
git tag vX.Y.0 -m "Release all packages to X.Y.0"
git push origin vX.Y.0

# CI automatically:
# 1. Creates unxt-vX.Y.0, unxt-api-vX.Y.0, unxt-hypothesis-vX.Y.0
# 2. Triggers builds for all packages
# 3. Publishes to TestPyPI and PyPI

# No manual builds needed!
```

### Scenario 2: Bug-fix Release (Individual Package)

Release a specific package with version `X.Y.Z` (Z > 0):

```bash
# Create and push package-specific tag (e.g., for unxt-api)
git tag unxt-api-vX.Y.Z -m "Release unxt-api X.Y.Z bug-fix"
git push origin unxt-api-vX.Y.Z

# CI automatically:
# 1. Validates the tag
# 2. Builds only unxt-api
# 3. Publishes to TestPyPI and PyPI

# Other packages are unaffected
```

### Creating a GitHub Release

For synchronized releases (vX.Y.0):

1. Go to <https://github.com/GalacticDynamics/unxt/releases/new>
2. Choose the `vX.Y.0` tag (the coordinator tag)
3. Fill in release notes covering all packages
4. Publish the release → triggers CD for all packages

For package-specific bug-fix releases:

1. Go to <https://github.com/GalacticDynamics/unxt/releases/new>
2. Choose the `package-vX.Y.Z` tag
3. Fill in release notes for that package only
4. Publish the release → triggers CD for that package

### Manual Testing Before Release

To test locally without creating tags:

```bash
# Check current development versions
hatch version  # Or: git describe --tags

# Build locally
cd /path/to/package
uv build

# Test installation
uv pip install dist/*.whl --force-reinstall

# Publish to TestPyPI first
uv publish --repository testpypi
```

## Automation Details

The release process is fully automated via GitHub Actions:

1. **Tag Creation** (`.github/workflows/create-package-tags.yml`):
   - Triggers on `v*` tags
   - Validates it's a `.0` release
   - Creates package-specific tags automatically
   - Pushes all three package tags

2. **Package Builds** (`.github/workflows/cd-*.yml`):
   - Each package has its own CD workflow
   - Triggers on package-specific tags only (`unxt-v*`, `unxt-api-v*`, `unxt-hypothesis-v*`)
   - Validates tags with `scripts/validate_tag.py`
   - Builds and publishes to TestPyPI, then PyPI

3. **Version Detection**:
   - hatch-vcs uses standard `git describe` with package-specific `--match` patterns
   - Each package only sees its own tags
   - No custom scripts needed!

## Version Detection Details

Each package's `pyproject.toml` is configured with package-specific git describe matching:

Between releases, versions are automatically suffixed with development info:

- `0.1.1.dev5+gabc1234` - 5 commits after v0.1.0 tag, at commit abc1234

This happens automatically via hatch-vcs without any manual intervention.

### Bugfix Releases on Old Versions

Create a maintenance branch from an old tag and release from there:

```bash
# Create maintenance branch from old release
git checkout unxt-v1.7.0
git checkout -b maint-1.7.x

# Make bugfix changes
git commit -am "Fix critical bug in 1.7 series"

# Tag the bugfix release
git tag unxt-v1.7.1 -m "Bugfix release 1.7.1"
git push origin maint-1.7.x unxt-v1.7.1
```

2. **CD Workflows** (`.github/workflows/cd-*.yml`):
   - Each package has its own CD workflow
   - Triggers on package-specific tags only (`unxt-v*`, `unxt-api-v*`, `unxt-hypothesis-v*`)
   - Validates tags with `scripts/validate_tag.py`
   - Builds and publishes to TestPyPI, then PyPI

3. **Version Detection**:
   - hatch-vcs uses standard `git describe` with package-specific `--match` patterns
   - Each package only sees its own tags
   - No custom scripts needed!

## Version Detection Details

Each package's `pyproject.toml` is configured with:

```toml
[tool.hatch.version]
source = "vcs"

[tool.hatch.version.raw-options.scm.git]
describe_command = ["git", "describe", "--dirty", "--tags", "--long", "--match", "PACKAGE-v*"]
```

Where `PACKAGE` is:

- `unxt` for the main package
- `unxt-api` for unxt-api
- `unxt-hypothesis` for unxt-hypothesis

This ensures each package only considers its own tags when determining the version.

## Testing Versions Locally

You can test version detection without pushing tags:

```bash
# Create local test tags (DO NOT PUSH)
git tag unxt-api-v0.1.0 -m "Test tag"
git tag unxt-hypothesis-v0.1.0 -m "Test tag"
git tag unxt-v1.8.0 -m "Test tag"

# Check detected versions
cd /path/to/package
hatch version

# Or check in Python
uv run python -c "import unxt; print(unxt.__version__)"

# Delete test tags when done
git tag -d unxt-api-v0.1.0 unxt-hypothesis-v0.1.0 unxt-v1.8.0
```

The version will match the tag exactly (e.g., `0.1.0` for tag `unxt-api-v0.1.0`), and commits after the tag will show development versions like `0.1.1.dev1+gabc1234`.

## Troubleshooting

### CI fails with "package tags must have coordinator tag"

For `.0` releases, you must create the `vX.Y.0` coordinator tag first:

```bash
# Create coordinator tag
git tag v1.8.0 -m "Release 1.8.0"
git push origin v1.8.0

# CI will automatically create unxt-v1.8.0, unxt-api-v1.8.0, unxt-hypothesis-v1.8.0
```

### Wrong version being detected

Check which tags are present:

```bash
git tag -l "unxt-v*"
git tag -l "unxt-api-v*"
git tag -l "unxt-hypothesis-v*"

# See what git describe returns
git describe --tags --match "unxt-v*"
```

### Package not building after tag push

Verify the tag matches the expected pattern:

- Main package: `unxt-vX.Y.Z`
- API package: `unxt-api-vX.Y.Z`
- Hypothesis package: `unxt-hypothesis-vX.Y.Z`

Check GitHub Actions for error messages.

```bash
cd packages/unxt-hypothesis
uv run hatch version
```

Without any matching tags, the `fallback-version = "0.0.0"` is used.

## Best Practices

1. **Always test locally** before pushing tags:

   ```bash
   uv run nox -s test
   ```

2. **Use semantic versioning:**
   - MAJOR: incompatible API changes
   - MINOR: new functionality in a backward-compatible manner
   - PATCH: backward-compatible bug fixes

3. **Write release notes** when creating GitHub Releases

4. **Consider dependencies:** If `unxt-hypothesis` depends on a specific `unxt` version, update `packages/unxt-hypothesis/pyproject.toml` accordingly before release

5. **Tag from the right commit:** Ensure you're on the correct commit before creating tags

## Troubleshooting

### Version shows as "0.0.0"

This means hatch-vcs couldn't find a matching tag. Check:

- Are you in a git repository?
- Does a tag matching the pattern exist?
- Did you fetch all tags? (`git fetch --tags`)

### Build fails in CI

Check the workflow logs to see which package failed and why. Common issues:

- Missing dependencies in `build-system.requires`
- Incorrect tag format
- Build errors in the package code

### Wrong package published

Verify the tag matches the correct pattern exactly:

- `unxt-v1.5.0` (✓ correct)
- `v1.5.0` (✗ wrong - ambiguous)
- `unxt-1.5.0` (✗ wrong - missing 'v')
