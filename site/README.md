# xyntetik.com

The public site, built by `build.py` (standard library only) into `_site/`
and published to GitHub Pages by `.github/workflows/pages.yml` on every push
to `main` that touches `site/`.

- `pages/*.html`: one fragment per page with a leading `<!--meta -->` block
  (title, description, path, nav). The shell (head, header, footer) is
  applied by the generator.
- `assets/`: stylesheet, the small script (menu, reveal-on-scroll), the
  twelve brand marks under `marks/` (xyntetik, runner, dojo, suite, lab, zenova; light and dark), favicon, the two README recordings and the two Dojo screenshots.
- `{{chart:name}}` in a page renders an inline SVG from the data tables at
  the top of `build.py`. Those tables are copies of numbers in the README
  and `docs/`; every caption names the source document and the date.
- `{{repo}}`, `{{dojo}}` and `{{hf}}` expand to the Runner repository, the Dojo repository and the Hugging Face URLs.
- `{{tiers:table}}` renders the "Actively validated models" table (the
  README's family blocks as one table, a row per model in the README's
  order) and
  `{{validated:OWNER/REPO[,OWNER/REPO]}}` a model card's "Last
  validated" line, both from `docs/model-tiers.json` at build time through
  `scripts/model-tiers.py`'s own functions, so no page source holds a mark.
  A card that covers no roster file carries no line, and naming a
  repository that holds none fails the build. `build.py --roster FILE --out DIR` builds from another
  roster into another directory (the tests do).

Rules the build enforces: no em dashes anywhere in the output (public-prose
rule), no placeholder tokens, no internal link without a page behind it.
The rule it cannot enforce, and the one that matters most: nothing on the
site may claim more than the README and docs do on the day it is built. If
the site and the README disagree, the README is right and the site has a
bug.

Local preview: `python3 site/build.py && python3 -m http.server -d site/_site 8000`.

Layout check, with the preview running: `node site/shot.mjs URL WIDTH
HEIGHT OUT.png [mobile]` drives headless Chrome with device emulation and
prints the page's layout width against the viewport (`scrollW` above
`innerW` is a horizontal overflow) and the first elements that reach past
it, then writes a full-page screenshot; `SHOT_SELECTOR='#id'` captures one
element instead. Check every changed page at 1440 and at 390 with `mobile`.
A plain `--window-size` is clamped to about 500 px and fakes an overflow at
phone width, which is why the emulation is there. Each run uses a Chrome
profile of its own and `CHROME` names the browser binary when it is not the
macOS install.
