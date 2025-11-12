# Documentation

This directory contains the Sphinx documentation for the Hill Climber package.

## Building the Documentation Locally

To build the documentation on your machine:

```bash
cd docs
make html
```

The built documentation will be in `docs/build/html/`. Open `index.html` in your browser to view.

## Viewing the Documentation Online

The documentation is automatically built and deployed to GitHub Pages on every push to the main branch.

Visit: https://gperdrizet.github.io/hill_climber/

## Documentation Structure

- `source/index.rst` - Main landing page
- `source/installation.rst` - Installation instructions
- `source/quickstart.rst` - Quick start guide
- `source/user_guide.rst` - Comprehensive user guide
- `source/api.rst` - API reference (auto-generated)
- `source/notebooks.rst` - Example notebooks
- `source/advanced.rst` - Advanced topics and troubleshooting
- `source/conf.py` - Sphinx configuration

## Updating the Documentation

1. Edit the `.rst` files in `docs/source/`
2. Rebuild locally to preview: `make html`
3. Commit and push to main
4. GitHub Actions will automatically rebuild and deploy

## Dependencies

- sphinx
- sphinx-rtd-theme
- nbsphinx
- sphinx-autodoc-typehints

All installed via the main `requirements.txt` or during CI/CD.
