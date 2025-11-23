# Sphinx Documentation Implementation

This repository contains a Sphinx documentation system. The documentation is deployed to GitHub Pages with automatic builds via GitHub Actions.

## Documentation Files

**Core Documentation (docs/source/)**
- `conf.py` - Sphinx configuration with RTD theme, nbsphinx, autodoc
- `index.rst` - Main landing page with feature overview
- `installation.rst` - Installation guide with requirements and verification
- `quickstart.rst` - Quick start tutorial with working examples
- `user_guide.rst` - Comprehensive guide covering all concepts
- `api.rst` - API reference (auto-generated from docstrings)
- `notebooks.rst` - Example notebooks index with download links
- `advanced.rst` - Advanced topics and troubleshooting

**Build System**
- `docs/Makefile` - Sphinx build commands
- `docs/.gitignore` - Ignore build artifacts
- `docs/README.md` - Documentation maintenance guide
- `docs/source/_static/` - Static assets directory

**Deployment**
- `.github/workflows/docs.yml` - GitHub Actions workflow for automated deployment

## Features

- **Theme**: Read the Docs theme (mobile-friendly)
- **Notebook Integration**: Download links for all 6 notebooks
- **API Documentation**: Auto-generated from docstrings
- **Search Functionality**: Full-text search across all pages
- **Math Support**: MathJax/KaTeX for equations
- **Code Highlighting**: Syntax highlighting for all code blocks
- **Cross-References**: Internal linking between pages
- **Navigation**: Clear hierarchical structure
- **Responsive Design**: Works on mobile and desktop
- **Version Control**: Integrated with Git workflow

## Automated Deployment

### GitHub Actions Workflow

The `.github/workflows/docs.yml` file provides:

**Triggers**
- Push to `main` branch
- Pull requests to `main`

**Build Process**
1. Checkout repository
2. Set up Python 3.12
3. Install Sphinx and extensions
4. Install package dependencies
5. Build HTML documentation
6. Create `.nojekyll` file (bypass Jekyll processing)
7. Deploy to `gh-pages` branch

**Requirements Installed**
- sphinx
- sphinx-rtd-theme
- nbsphinx
- sphinx-autodoc-typehints
- All package requirements from `requirements.txt`

## Deployment Instructions

### 1. Enable GitHub Pages

In your GitHub repository:

1. Go to **Settings** → **Pages**
2. Under "Source", select **Deploy from a branch**
3. Select branch: **gh-pages**
4. Select folder: **/ (root)**
5. Click **Save**

### 2. Push to GitHub

```bash
git add docs/ .github/workflows/docs.yml DOCUMENTATION_SETUP.md
git commit -m "Add Sphinx documentation with automated deployment"
git push origin main
```

### 3. Monitor Deployment

1. Go to **Actions** tab in GitHub
2. Watch "Build and Deploy Documentation" workflow
3. First run will create `gh-pages` branch automatically
4. Subsequent pushes trigger automatic rebuilds

### 4. Access Documentation

After deployment completes (2-3 minutes):

**URL**: `https://gperdrizet.github.io/hill_climber/`

## Local Development

### Build Locally

```bash
cd docs
make html
```

Output: `docs/build/html/index.html`

### View Locally

```bash
cd docs/build/html
python -m http.server 8000
```

Then open: `http://localhost:8000`

### Clean Build

```bash
cd docs
make clean
make html
```

## Maintenance

### Updating Documentation

1. **Edit content**: Modify `.rst` files in `docs/source/`
2. **Test locally**: Run `make html` to preview changes
3. **Commit changes**: `git add docs/source/ && git commit -m "Update docs"`
4. **Push**: `git push origin main`
5. **Auto-deploy**: GitHub Actions rebuilds and deploys automatically

### Adding New Pages

1. Create new `.rst` file in `docs/source/`
2. Add to `index.rst` table of contents:
   ```rst
   .. toctree::
      :maxdepth: 2
      
      new_page
   ```
3. Build and push

### Updating API Docs

API documentation auto-generates from docstrings. To update:

1. Modify docstrings in Python code
2. Rebuild documentation
3. Changes appear automatically

## Configuration Highlights

### Sphinx Extensions Used

```python
extensions = [
    'sphinx.ext.autodoc',          # API doc generation
    'sphinx.ext.napoleon',         # Google/NumPy docstring support
    'sphinx.ext.viewcode',         # Source code links
    'sphinx.ext.autosummary',      # Summary tables
    'sphinx.ext.mathjax',          # Math equations
    'nbsphinx',                    # Jupyter notebook support
]
```

### Theme Customization

- **Theme**: `sphinx_rtd_theme` (Read the Docs)
- **Responsive**: Mobile and desktop support
- **Dark mode**: Available via theme toggle
- **Sidebar navigation**: Collapsible sections

### Notebook Handling

- **Execution**: Disabled (`nbsphinx_execute = 'never'`)
- **Error handling**: Continue on errors
- **Timeout**: 60 seconds
- **Download links**: All notebooks downloadable

## File Structure

```
hill_climber/
├── .github/
│   └── workflows/
│       └── docs.yml                    # Auto-deployment workflow
├── docs/
│   ├── source/
│   │   ├── _static/                    # Static assets
│   │   ├── conf.py                     # Sphinx config
│   │   ├── index.rst                   # Main page
│   │   ├── installation.rst            # Install guide
│   │   ├── quickstart.rst              # Quick start
│   │   ├── user_guide.rst              # User guide
│   │   ├── api.rst                     # API reference
│   │   ├── notebooks.rst               # Notebooks index
│   │   └── advanced.rst                # Advanced topics
│   ├── build/                          # Generated HTML (gitignored)
│   ├── Makefile                        # Build commands
│   ├── .gitignore                      # Ignore build artifacts
│   └── README.md                       # Docs maintenance guide
└── DOCUMENTATION_SETUP.md              # This file
```

## Next Steps

1. **Review content**: Check documentation pages for accuracy
2. **Push to GitHub**: Deploy using instructions above
3. **Enable Pages**: Activate GitHub Pages in repository settings
4. **Share link**: Documentation will be live at `https://gperdrizet.github.io/hill_climber/`

## Testing

To verify everything works:

```bash
# Test local build
cd docs
make clean
make html
ls build/html/index.html  # Should exist

# Test content
python -m http.server 8000 --directory build/html
# Visit http://localhost:8000 and verify all pages load

# Test deployment workflow
git push origin main
# Check GitHub Actions tab for successful workflow execution
```

## Support

- **Sphinx Documentation**: https://www.sphinx-doc.org/
- **RTD Theme**: https://sphinx-rtd-theme.readthedocs.io/
- **nbsphinx**: https://nbsphinx.readthedocs.io/
- **GitHub Pages**: https://docs.github.com/pages
