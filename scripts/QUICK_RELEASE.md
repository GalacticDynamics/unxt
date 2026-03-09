# Quick Release Reference for unxt Workspace

All releases are automated via GitHub Actions - just push tags!

## Release Types

### All Packages Together (Major/Minor)

Use coordinator tag: `vX.Y.0` (e.g., `v1.8.0`)

```bash
# Create and push coordinator tag
git tag v1.8.0 -m "Release all packages to 1.8.0"
git push origin v1.8.0

# CD automatically:
# 1. Creates unxt-v1.8.0, unxt-api-v1.8.0, unxt-hypothesis-v1.8.0
# 2. Builds all packages
# 3. Publishes to TestPyPI and PyPI
```

### Single Package Bug-fix

Use package tag: `PACKAGE-vX.Y.Z` where Z > 0 (e.g., `unxt-api-v1.8.1`)

```bash
# Create and push package-specific tag (example: unxt-api)
git tag unxt-api-v1.8.1 -m "Release unxt-api 1.8.1 bug-fix"
git push origin unxt-api-v1.8.1

# CD automatically builds and publishes only unxt-api
```

## Command Reference

| Task                  | Command                                             |
| --------------------- | --------------------------------------------------- |
| Check package version | `cd /path/to/package && hatch version`              |
| List all tags         | `git tag -l`                                        |
| Create package tag    | `git tag PACKAGE-vX.Y.Z -m "Release PACKAGE X.Y.Z"` |
| Push tag              | `git push origin TAG`                               |

## Tag Format Rules

✅ **Coordinator tags** (synchronized releases):

- `v1.8.0` → CD creates `unxt-v1.8.0`, `unxt-api-v1.8.0`, `unxt-hypothesis-v1.8.0`
- Must be `.0` releases (major/minor only)

✅ **Package tags** (independent bug-fixes):

- `unxt-api-v1.8.1` (bug-fix for unxt-api only)
- `unxt-hypothesis-v1.8.2` (bug-fix for unxt-hypothesis only)

❌ **Invalid**:

- `v1.8.1` (bug-fixes must be package-specific)
- Manual creation of package `.0` tags (use coordinator tag instead)

## Full Release Workflow

### 1. Prepare

```bash
# Make sure you're on main and everything is committed
git status  # Should be clean
git pull origin main
```

### 2. Create and Push Tag

```bash
# For synchronized release (all packages)
git tag v1.8.0 -m "Release all packages to 1.8.0"
git push origin v1.8.0

# Or for single package bug-fix
git tag unxt-api-v1.8.1 -m "Release unxt-api 1.8.1"
git push origin unxt-api-v1.8.1
```

### 3. Monitor CD

- GitHub Actions automatically builds and publishes
- Check: <https://github.com/GalacticDynamics/unxt/actions>
- For coordinator tags: watch for 4 workflows (create-package-tags + 3 CD workflows)
- For package tags: watch for 1 CD workflow

### 4. Create GitHub Release (Optional)

- Go to <https://github.com/GalacticDynamics/unxt/releases/new>
- Select the tag you just pushed
- Add release notes
- Click "Publish release" (triggers additional CD run)

## Testing Before Release

```bash
# Check current version
cd /path/to/package
hatch version

# Create local test tag (don't push!)
git tag unxt-api-v0.1.0 -m "Test"
hatch version  # Should show 0.1.0

# Build locally
uv build

# Clean up test tag
git tag -d unxt-api-v0.1.0
```

## Common Questions

**Q: How do I release all packages together?** A: Push `vX.Y.0` coordinator tag. CD creates package tags automatically.

**Q: How do I release just one package?** A: Push `PACKAGE-vX.Y.Z` tag where Z > 0 for bug-fixes.

**Q: Can I use `unxt-v1.8.0` for synchronized releases?** A: No, use coordinator tag `v1.8.0` which auto-creates package tags.

**Q: What if I made a mistake with a tag?** A: Delete locally and remotely before any releases publish:

```bash
git tag -d OLD_TAG
git push origin :OLD_TAG  # Delete remote
git tag NEW_TAG
git push origin NEW_TAG
```

## References

- Full documentation: See [RELEASING.md](../RELEASING.md)
- Version system details: See [Versioning Strategy](../RELEASING.md#versioning-strategy)
- Main repo: <https://github.com/GalacticDynamics/unxt>
