# Business brief

[`business_brief.pdf`](business_brief.pdf): *Does Repeat Shopping Mean Loyalty? What household
purchase data can and cannot tell a retailer about promotions and return visits.*

An eight-page companion to the paper for readers who need the decision rather than the method:
what the purchase records establish about repeat shopping, what they leave open, the holdout test
that would answer the question a retailer actually faces, and how to judge whether a measured lift
pays for a campaign. Page 1 stands on its own.

Every number from the study that the brief prints comes from the paper and the frozen records in
this repository. The comment block at the top of [`business_brief.tex`](business_brief.tex) lists
each source. The worked example in the break-even section uses invented numbers and says so.

To rebuild the PDF from its source:

```
tectonic business_brief.tex
```

The first local build may download the required TeX fonts (Source Sans Pro and Latin Modern
Math) through Tectonic. Later builds use its local cache. The brief is not part of the
reproduction checks: `reproduce.sh` verifies only that these files match `SHA256SUMS`.
