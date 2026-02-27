# Quick Release Reference for unxt Workspace

`scripts/get_version.py` prints a git-describe-like version string (not bare
`X.Y.Z`), e.g. `v1.8.0-0-gabc1234` or `unxt-api-v1.8.1-3-gdef5678-dirty`.

## Release Types

### All Packages Together (Major/Minor)

Use shared tag: `vX.Y.0` (e.g., `v1.8.0`)

```bash
# Create tag
git tag v1.8.0 -m "Release all packages to 1.8.0"

# Push and verify
git push origin v1.8.0
python scripts/get_version.py --main          # → v1.8.0-0-g<sha>
python scripts/get_version.py unxt-api        # → v1.8.0-0-g<sha>
python scripts/get_version.py unxt-hypothesis # → v1.8.0-0-g<sha>
```

### Single Package Bug-fix

Use package tag: `PACKAGE-vX.Y.Z` where Z > 0 (e.g., `unxt-api-v1.8.1`)

```bash
# Create tag (example: unxt-api)
git tag unxt-api-v1.8.1 -m "Release unxt-api 1.8.1 bug-fix"

# Push and verify
git push origin unxt-api-v1.8.1
python scripts/get_version.py unxt-api # → unxt-api-v1.8.1-0-g<sha>
```

## Command Reference

| Task                  | Command                                                                                       |
| --------------------- | --------------------------------------------------------------------------------------------- |
| Check package version | `python scripts/get_version.py PACKAGE`                                                       |
| Check all versions    | `for p in "" unxt-api unxt-hypothesis; do python scripts/get_version.py ${p:-"--main"}; done` |
| List all tags         | `git tag -l`                                                                                  |
| Create shared tag     | `git tag vX.Y.0`                                                                              |
| Create package tag    | `git tag PACKAGE-vX.Y.Z`                                                                      |
| Build package         | `cd <path> && uv build`                                                                       |
| Publish package       | `cd <path> && uv publish`                                                                     |

## Tag Format Rules

❌ **Don't use:**

- `v1.8.1` (shared tag with bug-fix)
- `unxt-api-v1.8.0` (package tag with .0)
- `unxt-v1.8.0` (use `v1.8.0` shared instead)

✅ **Use:**

- `v1.8.0` (shared, all packages)
- `unxt-api-v1.8.1` (package bug-fix, Z > 0)
- `unxt-hypothesis-v1.8.2` (package bug-fix, Z > 0)

## Full Release Workflow

### 1. Prepare

```bash
# Make sure you're on main and everything is committed
git status  # Should be clean
git pull origin main
```

### 2. Create Tag

```bash
# For all packages (major/minor)
git tag vX.Y.0 -m "Release all packages to X.Y.0"

# Or for single package (bug-fix)
git tag PACKAGE-vX.Y.Z -m "Release PACKAGE X.Y.Z"
```

### 3. Verify

```bash
# Check which packages get which versions
python scripts/get_version.py --main
python scripts/get_version.py unxt-api
python scripts/get_version.py unxt-hypothesis
```

### 4. Push

```bash
git push origin vX.Y.0  # or git push origin PACKAGE-vX.Y.Z
```

### 5. Create GitHub Release

- Go to <https://github.com/GalacticDynamics/unxt/releases/new>
- Select the tag you just pushed
- Add release notes
- Click "Publish release"

### 6. Build and Publish

For shared release (all packages):

```bash
# unxt (main)
cd /path/to/unxt
uv build && uv publish

# unxt-api
cd packages/unxt-api
uv build && uv publish

# unxt-hypothesis
cd packages/unxt-hypothesis
uv build && uv publish
```

For single package release:

```bash
# Only affected package
cd packages/PACKAGE
uv build && uv publish
```

## Testing Before Release

```bash
# Verify versions are correct
python scripts/get_version.py --main
python scripts/get_version.py unxt-api
python scripts/get_version.py unxt-hypothesis

# Build in test mode
cd /path/to/package
uv build

# Test with test PyPI (optional)
uv publish --repository testpypi
```

## Common Issues

**Q: How do I know which tag to use?** A: Ask:

- "Is this a coordinated release affecting all packages?" → Use `vX.Y.0`
- "Is this a bug-fix for one package only?" → Use `PACKAGE-vX.Y.Z` (Z > 0)

**Q: Can I release unxt-api-v1.8.0?** A: No. Use the shared tag `v1.8.0`
instead.

**Q: Can I release v1.8.1 (without package prefix)?** A: No. Use
package-specific tag `unxt-api-v1.8.1` instead.

**Q: What if I made a mistake with the tag?** A: Delete and recreate (only for
tags not yet released):

```bash
git tag -d OLD_TAG
git push origin :OLD_TAG  # Delete remote
git tag NEW_TAG
git push origin NEW_TAG
```

## References

- Full documentation: See [RELEASING.md](../RELEASING.md)
- Version system details: See
  [Versioning Strategy](../RELEASING.md#versioning-strategy)
- Main repo: <https://github.com/GalacticDynamics/unxt>
