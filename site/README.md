# xyntetik.com

The public site, built by `build.py` (standard library only) into `_site/`
and published to GitHub Pages by `.github/workflows/pages.yml` on every push
to `main` that touches `site/`.

- `pages/*.html`: one fragment per page with a leading `<!--meta -->` block
  (title, description, path, nav). The shell (head, header, footer) is
  applied by the generator.
- `assets/`: stylesheet, the small script (menu, reveal-on-scroll), the
  ensō mark, the Zenova star, favicon, and the two README recordings.
- `{{chart:name}}` in a page renders an inline SVG from the data tables at
  the top of `build.py`. Those tables are copies of numbers in the README
  and `docs/`; every caption names the source document and the date.
- `{{repo}}` and `{{hf}}` expand to the repository and Hugging Face URLs.

Rules the build enforces: no em dashes anywhere in the output (public-prose
rule), no placeholder tokens, no internal link without a page behind it.
The rule it cannot enforce, and the one that matters most: nothing on the
site may claim more than the README and docs do on the day it is built. If
the site and the README disagree, the README is right and the site has a
bug.

Local preview: `python3 site/build.py && python3 -m http.server -d site/_site 8000`.
