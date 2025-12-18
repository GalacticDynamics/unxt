# Release Process for unxt Workspace

This workspace contains three packages that can be released independently:

- `unxt` - the main package
- `unxt-api` - abstract dispatch API
- `unxt-hypothesis` - hypothesis testing strategies

## Versioning with hatch-vcs

All packages use **hatch-vcs** for automatic version detection from git tags.

### Tag Format

- **unxt**: `unxt-vX.Y.Z` (e.g., `unxt-v1.8.0`)
- **unxt-api**: `unxt-api-vX.Y.Z` (e.g., `unxt-api-v0.1.0`)
- **unxt-hypothesis**: `unxt-hypothesis-vX.Y.Z` (e.g., `unxt-hypothesis-v0.1.0`)

The version must follow PEP 440 format: `X.Y.Z` with optional suffixes like
`a1`, `b2`, `rc1`, `.post1`, `.dev0`, etc.

**Note**: Each package uses `git describe` with tag pattern matching to find
only its own tags. This allows independent versioning in the monorepo.

## How Versioning Works

When you tag a commit:

- The package version matches the tag (e.g., tag `unxt-api-v0.1.0` → version
  `0.1.0`)
- After the tag, development versions are created automatically (e.g.,
  `0.1.1.dev5+gabc1234`)
- Each package only looks at tags matching its pattern, ignoring tags for other
  packages

## Release Workflows

### Release via Git Tags (Recommended)

1. **Create and push a tag:**

   ```bash
   # For unxt (accepts either format)
   git tag unxt-v1.8.0 -m "Release unxt 1.8.0"
   # or
   git tag v1.8.0 -m "Release unxt 1.8.0"

   git push origin unxt-v1.8.0

   # For unxt-api
   git tag unxt-api-v0.2.0 -m "Release unxt-api 0.2.0"
   git push origin unxt-api-v0.2.0

   # For unxt-hypothesis
   git tag unxt-hypothesis-v0.2.0 -m "Release unxt-hypothesis 0.2.0"
   git push origin unxt-hypothesis-v0.2.0
   ```

2. **Build and publish manually:**

   ```bash
   # Build the package (version will be detected from tags)
   cd /path/to/package
   uv build

   # Publish to PyPI
   uv publish
   ```

3. **Create a GitHub Release:**
   - Go to https://github.com/GalacticDynamics/unxt/releases/new
   - Choose or create a tag following the package-prefixed format
   - Fill in release notes
   - Publish the release

### Manual Build and Publish

For all packages:

```bash
# From repository root for unxt
uv build
uv publish

# From package directory for workspace packages
cd packages/unxt-api
uv build
uv publish

cd ../unxt-hypothesis
uv build
uv publish
```

**Note**: Publishing requires PyPI credentials configured.

## Release Scenarios

### Independent Releases

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
