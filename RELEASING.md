# Release Process for unxt Workspace

This workspace contains three packages that can be released:

- `unxt` - the main package
- `unxt-api` - abstract dispatch API
- `unxt-hypothesis` - hypothesis testing strategies

## Versioning Strategy

All packages use **hatch-vcs** for automatic version detection from git tags,
with a custom validation system that enforces the following strategy:

### Tag Format Rules

**Major/Minor Releases (all packages together):**

- Use shared tags: `vX.Y.0` (e.g., `v1.8.0`)
- Applies to **all** workspace packages simultaneously
- Required format: `.0` (must be a `.0` release)
- Example: Tag `v1.8.0` bumps unxt, unxt-api, and unxt-hypothesis all to 1.8.0

**Bug-fix Releases (individual packages only):**

- Use package-specific tags: `PACKAGE-vX.Y.Z` where Z > 0 (e.g.,
  `unxt-api-v1.8.1`)
- Applies to **only that package**
- Required format: Must have patch version > 0 (bug-fix release)
- Example: Tag `unxt-api-v1.8.1` bumps only unxt-api to 1.8.1
- Other packages are unaffected

### Validation Rules (Enforced)

- ✅ `vX.Y.0` - **Allowed**: Shared tag bumps all packages
- ❌ `vX.Y.0` (package-specific like `unxt-api-vX.Y.0`) - **Forbidden**: Use
  shared tag instead
- ✅ `unxt-api-vX.Y.Z` (Z > 0) - **Allowed**: Bug-fix for specific package
- ❌ `vX.Y.Z` (Z > 0, shared) - **Forbidden**: Must be package-specific for
  bug-fixes

This ensures clear separation: major/minor synchronized across all packages,
bug-fixes isolated per package.

## How Versioning Works

When you tag a commit:

1. The custom version determination script (`scripts/get_version.py`) validates
   all tags
2. Each package selects the latest applicable tag(s):
   - For major/minor: Uses shared `vX.Y.0` or its own package-specific version
     if newer
   - For bug-fix: Only uses package-specific tags with Z > 0
3. The version matches the selected tag (e.g., tag `v1.8.0` → all packages get
   `1.8.0`)
4. After the tag, development versions are created automatically (e.g.,
   `1.8.1.dev5+gabc1234`)

### Legacy Support

Tags from v1.10.x and earlier (including pre-1.0 tags) are grandfathered in and
allowed without validation. Strict enforcement begins at v1.11.0.

## Release Workflows

### Scenario 1: Major/Minor Release (Synchronized All Packages)

Tag and release all packages together with version `X.Y.0`:

```bash
# Create shared tag
git tag vX.Y.0 -m "Release all packages to X.Y.0"

# Push the tag
git push origin vX.Y.0

# Verify version detection
python scripts/get_version.py --main          # Should show X.Y.0
python scripts/get_version.py unxt-api       # Should show X.Y.0
python scripts/get_version.py unxt-hypothesis # Should show X.Y.0

# Build and publish
cd /path/to/unxt
uv build && uv publish

cd packages/unxt-api
uv build && uv publish

cd ../unxt-hypothesis
uv build && uv publish
```

### Scenario 2: Bug-fix Release (Individual Package)

Tag and release a specific package with version `X.Y.Z` (Z > 0):

```bash
# Create package-specific tag (e.g., for unxt-api)
git tag unxt-api-vX.Y.Z -m "Release unxt-api X.Y.Z"

# Push the tag
git push origin unxt-api-vX.Y.Z

# Verify version detection
python scripts/get_version.py unxt-api       # Should show X.Y.Z

# Build and publish
cd packages/unxt-api
uv build && uv publish

# Other packages are unaffected and should not be released
```

### Creating a GitHub Release

For shared releases:

1. Go to <https://github.com/GalacticDynamics/unxt/releases/new>
2. Choose the `vX.Y.0` tag
3. Fill in release notes covering all packages
4. Publish the release

For package-specific releases:

1. Go to <https://github.com/GalacticDynamics/unxt/releases/new>
2. Choose the `package-vX.Y.Z` tag
3. Fill in release notes for that package
4. Publish the release

### Release via Git Tags (Full Process)

1. **Prepare the release commit**
   - Review changes requiring release
   - Update CHANGELOG if maintained
   - Commit any final changes

2. **Create and push the appropriate tag:**

   For all packages (major/minor):

   ```bash
   git tag vX.Y.0 -m "Release all packages to X.Y.0"
   git push origin vX.Y.0
   ```

   For single package (bug-fix):

   ```bash
   git tag unxt-api-vX.Y.Z -m "Release unxt-api X.Y.Z"
   git push origin unxt-api-vX.Y.Z
   ```

3. **Verify versioning is correct:**

   ```bash
   python scripts/get_version.py --main
   python scripts/get_version.py unxt-api
   python scripts/get_version.py unxt-hypothesis
   ```

4. **Build and publish**

   ```bash
   cd /package/path
   uv build
   uv publish
   ```

### Manual Build and Publish

For testing without tagging:

```bash
# Version detection uses tags, so build without a tag for dev versions
uv build
uv publish --repository testpypi  # Test first
uv publish                         # Publish to production
```

## Version Validation

To validate that tags follow the new strategy:

```bash
# The script will show warnings for policy violations
python scripts/get_version.py --main

# For strict validation (useful in CI):
# Check exit codes and stderr for policy violations
```

Packages can be released independently at different versions:

```bash
# Release only unxt
git tag unxt-v1.8.0 -m "Release unxt 1.8.0"
git push origin unxt-v1.8.0

# Later, release only unxt-api
git tag unxt-api-v0.2.0 -m "Release unxt-api 0.2.0"
git push origin unxt-api-v0.2.0

# Even later, release only unxt-hypothesis
git tag unxt-hypothesis-v0.2.0 -m "Release unxt-hypothesis 0.2.0"
git push origin unxt-hypothesis-v0.2.0
```

### Synchronized Releases

To release all packages together (e.g., for coordinated changes):

```bash
# Create tags for all packages
git tag unxt-v1.8.0 -m "Release unxt 1.8.0"
git tag unxt-api-v0.2.0 -m "Release unxt-api 0.2.0"
git tag unxt-hypothesis-v0.2.0 -m "Release unxt-hypothesis 0.2.0"

# Push all tags
git push origin unxt-v1.8.0 unxt-api-v0.2.0 unxt-hypothesis-v0.2.0
```

Then build and publish each package individually.

### Development Versions

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

## Version Detection Details

Each package's `pyproject.toml` is configured with:

```toml
[tool.hatch.version]
  source = "vcs"
  raw-options = {
    git_describe_command = "git describe --dirty --tags --long --match '<pattern>'"
  }
```

Where `<pattern>` is:

- `unxt-api-v*` for unxt-api
- `unxt-hypothesis-v*` for unxt-hypothesis
- `v*` and `unxt-v*` for unxt (matches both formats)

This ensures each package only considers its own tags when determining the
version.

## Testing Versions Locally

You can test version detection without pushing tags to the remote:

```bash
# Create local test tags (DO NOT PUSH)
git tag unxt-api-v0.1.0 -m "Test tag"
git tag unxt-hypothesis-v0.1.0 -m "Test tag"
git tag unxt-v1.8.0 -m "Test tag"

# Build packages to see detected versions
uv sync

# Check detected versions
uv run python -c "import unxt; import unxt_api; print(f'unxt: {unxt.__version__}'); print(f'unxt_api: {unxt_api.__version__}')"

# Delete test tags when done
git tag -d unxt-api-v0.1.0 unxt-hypothesis-v0.1.0 unxt-v1.8.0
```

The version will match the tag exactly (e.g., `0.1.0` for tag
`unxt-api-v0.1.0`), and commits after the tag will show development versions
like `0.1.1.dev1+gabc1234`.

## CI/CD Workflow

The automated release workflow:

1. **Detects the package(s) to release** based on:
   - Git tag name (e.g., `unxt-v*` or `unxt-hypothesis-v*`)
   - GitHub Release tag name
   - For non-release events (PRs, pushes to main), builds all packages for
     testing

2. **Builds packages** using `hynek/build-and-inspect-python-package`:
   - Artifacts are uploaded with package-specific names (`Packages-unxt`,
     `Packages-unxt-hypothesis`)
   - Each package is built from its directory in `packages/`

3. **Publishes to TestPyPI first** (for releases only)

4. **Publishes to PyPI** after TestPyPI succeeds

## Version Discovery

To check what version would be assigned:

```bash
# Check unxt version
cd packages/unxt
uv run hatch version

# Check unxt-hypothesis version
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

4. **Consider dependencies:** If `unxt-hypothesis` depends on a specific `unxt`
   version, update `packages/unxt-hypothesis/pyproject.toml` accordingly before
   release

5. **Tag from the right commit:** Ensure you're on the correct commit before
   creating tags

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
