# Release Process for unxt Workspace

This workspace contains two packages that can be released independently:

- `unxt` - the main package
- `unxt-hypothesis` - hypothesis testing strategies for unxt

## Tag-Based Releases

Each package uses **package-prefixed tags** to determine its version via
hatch-vcs.

### Tag Format

- **unxt**: `unxt-v<version>` (e.g., `unxt-v1.5.0`)
- **unxt-hypothesis**: `unxt-hypothesis-v<version>` (e.g.,
  `unxt-hypothesis-v1.3.0`)

The version must follow PEP 440 format: `X.Y.Z` with optional suffixes like
`a1`, `b2`, `rc1`, `.post1`, `.dev0`, etc.

## Release Workflows

### Option 1: Release via Git Tags (Recommended)

1. **Create and push a tag:**

   ```bash
   # For unxt
   git tag unxt-v1.5.0
   git push origin unxt-v1.5.0

   # For unxt-hypothesis
   git tag unxt-hypothesis-v1.3.0
   git push origin unxt-hypothesis-v1.3.0
   ```

2. **GitHub Actions will automatically:**
   - Detect which package to release based on the tag
   - Build the package
   - Publish to TestPyPI
   - Publish to PyPI

### Option 2: Release via GitHub Releases

1. **Create a GitHub Release:**
   - Go to https://github.com/GalacticDynamics/unxt/releases/new
   - Choose or create a tag following the package-prefixed format
   - Fill in release notes
   - Publish the release

2. **GitHub Actions will automatically:**
   - Build and publish the package(s) corresponding to the tag

### Option 3: Manual Release

For testing or special cases:

```bash
# Build a specific package
uv run --directory packages/unxt hatch build
uv run --directory packages/unxt-hypothesis hatch build

# Or from the package directory
cd packages/unxt
uv run hatch build

# Publish (requires PyPI credentials)
uv run hatch publish
```

## Release Scenarios

### Independent Releases

Packages can be released independently at different versions:

```bash
# Release only unxt
git tag unxt-v1.5.0
git push origin unxt-v1.5.0

# Later, release only unxt-hypothesis
git tag unxt-hypothesis-v1.3.0
git push origin unxt-hypothesis-v1.3.0
```

### Synchronized Releases

To release both packages together (e.g., for coordinated changes):

```bash
# Create both tags
git tag unxt-v1.5.0
git tag unxt-hypothesis-v1.3.0

# Push both tags
git push origin unxt-v1.5.0 unxt-hypothesis-v1.3.0
```

The CD workflow will build and publish both packages.

### Bugfix Releases on Old Branches

Create a maintenance branch and tag from there:

```bash
# Create maintenance branch from old release
git checkout unxt-v1.4.0
git checkout -b maint-1.4.x

# Make bugfix changes
git commit -am "Fix bug in 1.4 series"

# Tag the bugfix release
git tag unxt-v1.4.1
git push origin maint-1.4.x unxt-v1.4.1
```

## CI/CD Workflow Details

The `.github/workflows/cd.yml` workflow:

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
