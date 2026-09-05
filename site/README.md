# xyntetik.com

The public site is generated from this repository: `build_pages.py` copies
README sections and selected `docs/` pages into `runner/` at build time,
rewriting relative links to site pages where they exist and to GitHub
otherwise. The workflow `.github/workflows/pages.yml` runs the generator,
builds the Jekyll source in this directory and publishes it to GitHub Pages
on every push to `main` that touches `site/`, `docs/` or the README.

Nothing here may claim more than the repository does: the pages are copies.
Layout and the umbrella `index.md` are the only hand-written files. Generated
output (`runner/`, `_includes/build.html`) is ignored by git.
