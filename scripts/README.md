# Scripts Directory

Helper scripts for maintaining the hill-climber package.

## update_version.py

Utility script to update version numbers consistently across the package.

### Usage

**Check current versions:**
```bash
python scripts/update_version.py --check
```

**Update to new version:**
```bash
python scripts/update_version.py 0.2.0
```

This will:
- Validate the version format (semantic versioning)
- Update `hill_climber/__init__.py`
- Update `pyproject.toml`
- Display next steps for git commit and tag

### Version Format

Must follow semantic versioning: `MAJOR.MINOR.PATCH`

Examples:
- `0.1.0` - Initial development release
- `0.1.1` - Bug fix (patch)
- `0.2.0` - New features (minor)
- `1.0.0` - First stable release (major)

### Examples

```bash
# Check if versions are consistent
$ python scripts/update_version.py --check
__init__.py version:    0.1.0
pyproject.toml version: 0.1.0

✓ Versions are consistent!

# Update to new version
$ python scripts/update_version.py 0.2.0

Updating to version 0.2.0...
✓ Updated __init__.py to version 0.2.0
✓ Updated pyproject.toml to version 0.2.0

✓ Version update complete!

Next steps:
  1. Review changes: git diff
  2. Commit: git commit -am 'Bump version to 0.2.0'
  3. Tag: git tag v0.2.0
  4. Push: git push origin main --tags
```
