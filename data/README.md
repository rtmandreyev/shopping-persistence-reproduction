# The Complete Journey data

dunnhumby's *The Complete Journey* data are not included (see `../NOTICE.md`). `--quick` and
`--check` need no data. `--standard` needs the release, placed as follows.

1. Download *The Complete Journey* from <https://www.dunnhumby.com/source-files/> under
   dunnhumby's terms.
2. Put the release folder at `data/raw/dunnhumby_The-Complete-Journey/` so that it contains
   exactly these nine files (the CSV subfolder name contains spaces):

   ```
   data/raw/dunnhumby_The-Complete-Journey/
   ├── dunnhumby - The Complete Journey User Guide.pdf
   └── dunnhumby_The-Complete-Journey CSV/
       ├── campaign_desc.csv
       ├── campaign_table.csv
       ├── causal_data.csv
       ├── coupon.csv
       ├── coupon_redempt.csv
       ├── hh_demographic.csv
       ├── product.csv
       └── transaction_data.csv
   ```

   Alternatively, keep the folder elsewhere and pass `--data-dir PATH`, where PATH is the
   release folder itself or the folder that contains it.
3. `data/raw/manifest_raw.csv` lists the size and SHA-256 of each file in the version used in
   the paper (about 850 MB in total, of which `causal_data.csv` is 696 MB).
   `--standard` checks every file against it and reports any file that is missing or differs.

Nothing is written into `data/`: `--standard` builds its processed tables in a disposable copy
of the package under `build/workspace/`.
