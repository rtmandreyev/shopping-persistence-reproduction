"""Frozen Phase 5 preparation contracts: no substantive outcome estimation."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

from complete_journey.paths import PROJECT_ROOT

GROUP = "SPAGHETTI DRY"
SCOPE = "DESIGN_PREFLIGHT_NOT_ESTIMATION"
LOCAL_PATHS = {
    "ledger": "data/processed/choice/purchased_composition_ledger.parquet",
    "pure": "data/processed/choice/purchased_composition_event.parquet",
}
OUTPUTS = {
    "contract": "outputs/tables/choice/contracts.json",
    "qa": "outputs/tables/choice/qa_summary.csv",
    "summary": "outputs/tables/choice/phase5_preflight_summary.csv",
    "provenance": "outputs/tables/choice/provenance.json",
}
DESIGN_COLUMNS = (
    "household_key",
    "store_id",
    "week_no",
    "history_category",
    "log1p_prior_pure",
    "calendar_third",
)
EXPECTED = {
    "positive_group_lines": 6841,
    "positive_group_events": 6571,
    "all_recorded_group_events": 6573,
    "pure": 6544,
    "mixed": 27,
    "national_only": 3048,
    "private_only": 3496,
    "excluded_zero_only_events": 2,
}
BASELINE = {
    "rows": 6544,
    "households": 1560,
    "stores": 163,
    "weeks": 102,
    "H0": 1585,
    "HN": 1433,
    "HB": 2064,
    "HP": 1462,
    "mixed_only": 3,
    "national": 3048,
    "private": 3496,
}
LATER = {
    "rows": 4807,
    "households": 1366,
    "stores": 146,
    "weeks": 68,
    "H0": 655,
    "HN": 1088,
    "HB": 1930,
    "HP": 1134,
    "mixed_only": 1,
    "national": 2189,
    "private": 2618,
}
THIRDS = {
    "THIRD_1": {
        "rows": 1737,
        "H0": 930,
        "HN": 345,
        "HB": 134,
        "HP": 328,
        "mixed_only": 2,
        "national": 859,
        "private": 878,
    },
    "THIRD_2": {
        "rows": 2348,
        "H0": 407,
        "HN": 594,
        "HB": 718,
        "HP": 629,
        "mixed_only": 1,
        "national": 1090,
        "private": 1258,
    },
    "THIRD_3": {
        "rows": 2459,
        "H0": 248,
        "HN": 494,
        "HB": 1212,
        "HP": 505,
        "mixed_only": 0,
        "national": 1099,
        "private": 1360,
    },
}
SUPPORT = {"H0": 499.787271, "HB": 854.045885, "HP": 1038.196767}
SINGULAR = (38.615521, 23.985389, 18.043630)
# Fixed before implemented preflight results: accepted six-decimal support anchors.
SUPPORT_ATOL = 1e-6
SINGULAR_ATOL = 1e-6
INDEPENDENT_ATOL = 1e-10
REMAINING_ATOL = 5.1e-5
CONDITION_ATOL = 5.1e-5
SUMMARY_FIELDS = ("sample", "domain", "metric", "value", "scope")


class ChoiceContractError(ValueError):
    """Phase 5 construction/support/privacy contract violation."""


def require(condition, message: str) -> None:
    if not condition:
        raise ChoiceContractError(message)


PROTECTED_INPUTS = {
    "data/processed/basket_panel.parquet": "9ef6a868f80f051b9979763657171c1dc3d320d1c6998e504b6a2e676f6477dd",
    "data/processed/campaign_household_week.parquet": "38f8208410e39b71372bbf6e85f1c488be3c6b0412d809b68fdd0c1d186db096",
    "data/processed/campaign_week_calendar.parquet": "4be48ab605ae0679be41dacf6ad0f5078881c7ae3c0430ab9e2a4990c38043f3",
    "data/processed/choice_household_product_week.parquet": "0b4699d2da2d66103a910a089e141735f71cded8e8dd57040273486336bbc007",
    "data/processed/choice_line_price_candidates.parquet": "746139c535d9e1c4f0bd1eeeef626bf3b08aa5309052be2b8b632e9ef3468b6c",
    "data/processed/choice_product_store_week.parquet": "6f97b8284ce525f16ce62373e748a65157e7b17f73fe79e2eaf1ade141ab9414",
    "data/processed/econometrics_household_week.parquet": "4f30324a71f5648bf70a86a95040ae552194fb776f655755362c200ba1d4330b",
    "data/processed/household_category_week.parquet": "2830b7d43c781291dd54ab86c23f83b30363a941f3737beae0a206064888d0a8",
    "data/processed/household_history.parquet": "723aa744177562df00b8580d9d64fd3c1666143f10fefa24c1737fa1c808902f",
    "data/processed/household_week_base.parquet": "0b4f8e71adb2817abd548b0ee949b959747ec10f2c0d7927461e828682798b75",
    "data/processed/predictive_household_week_features.parquet": "723cbfa9687cd222bb7c212efaf20c39852fe346b19566d433bbf2a46e0e1ce9",
    "data/processed/private/panel_eligibility.parquet": "8013814ac4c3770a211a09a4bbc6c79ca74731cfd4ccdd2080615f31201b3a32",
    "data/raw/dunnhumby_The-Complete-Journey/dunnhumby - The Complete Journey User Guide.pdf": "5a8f89ac4fce70dc69d083f03dadf33e9c5ba3c995de7bf2830b18dcef3ac61f",
    "data/raw/dunnhumby_The-Complete-Journey/dunnhumby_The-Complete-Journey CSV/campaign_desc.csv": "632535c6bcdd97b5681e84c2c0652ddc4eaca8a0f899a3bb37dae8dbdc089684",
    "data/raw/dunnhumby_The-Complete-Journey/dunnhumby_The-Complete-Journey CSV/campaign_table.csv": "a716f88953c9ddf8847e191d862e8d022afd19cf7419f52cbe6dde950a121f21",
    "data/raw/dunnhumby_The-Complete-Journey/dunnhumby_The-Complete-Journey CSV/causal_data.csv": "60ed7021fefc209c0caf36fcf95d1e93693775acfdac01fc9ee38273da88937a",
    "data/raw/dunnhumby_The-Complete-Journey/dunnhumby_The-Complete-Journey CSV/coupon.csv": "718f4d466df8a54bbff45f340e0b601049721b0715503394a601bac3093c727d",
    "data/raw/dunnhumby_The-Complete-Journey/dunnhumby_The-Complete-Journey CSV/coupon_redempt.csv": "70a98f3bfb28be0dd850f19c8d806d2a23e44014546651367b21538db6fba1f8",
    "data/raw/dunnhumby_The-Complete-Journey/dunnhumby_The-Complete-Journey CSV/hh_demographic.csv": "3764c70d5e38b5da1206ab6d60788236d2b2e31ae3168f6382ed5880a0101123",
    "data/raw/dunnhumby_The-Complete-Journey/dunnhumby_The-Complete-Journey CSV/product.csv": "7ecbcec41e0f1e5a51b43a359965cc50dc5586a2b900a378b64032750fedc949",
    "data/raw/dunnhumby_The-Complete-Journey/dunnhumby_The-Complete-Journey CSV/transaction_data.csv": "3a685c0729cef664d634486189f774518b84f53cde7cbf701a5963238692b476",
    "data/raw/manifest_raw.csv": "eafa757c356adc98c429c39606274d98eddb0870e8f1ea76541b38acf7ecae22",
    "docs/DECISIONS.md": "5bda5d126866d2f7ce9829471f5cce61834c70c23f56905da5f1901da2bae7ef",
    "outputs/figures/descriptive/final/f1_weekly_retailer_activity.png": "75c11ea0d3e1c7aafc04afb3cba1dcb51466015b67f5fb0d87ad2e97c8598f3f",
    "outputs/figures/descriptive/final/f2_participation_and_intensity.png": "d9abe325d8b7105b677bf52b0a8de3654de7456a8b4c685298e15d8ea801a5d0",
    "outputs/figures/descriptive/final/f3_household_heterogeneity.png": "cd1571bad1a747d5943786ea99ae9d7a541db7e33a3ef3e6b46afc25a8037849",
    "outputs/figures/descriptive/final/f4_household_activity_concentration.png": "25b76cad77f603b6461670517105ac99c5374352c8989eb32e71771f534da2b9",
    "outputs/figures/descriptive/final/f5_product_structure.png": "eace606e760dfbf0a72979c1a764be4b0858e4bce2e3e6e52223c6aaf2406739",
    "outputs/figures/descriptive/final/f6_commodity_penetration_intensity.png": "38f5750f23dd42dc4ac3ac63b0076368cd081d89bab9e1d6130497956b21d0a8",
    "outputs/figures/descriptive/final/f7_national_private_composition.png": "56bd33f0cc24dd7ef84e8bdb5b25585674d1ab27bf0c60f0245c75b4ea9d199e",
    "outputs/figures/descriptive/final/f8_campaign_landscape.png": "2d6e18ab9150c8745d9ab210e1ce3015adbfbf17e6ea1060235aca831ccf0b28",
    "outputs/figures/descriptive/final/f9_representative_customer_journeys.png": "69adf6dffb43288446a450cdd8ec9510e527fd2784345075a70ad67391e1cf14",
    "outputs/figures/panel/final/f10_panel_adjustment_and_calendar_sensitivity.png": "cbfa391c18f0dea1496dc0660047c9026a8155a4ae4afb6412944a5117d8400d",
    "outputs/figures/panels/exploratory/phase4_run_c_adjustment_ladder.png": "97be8434ed6cb2b1e115e7fa2395c55999992d9ccb2e1b18f5ffb19266dd7dfd",
    "outputs/figures/panels/exploratory/phase4_run_d_covariance_comparison.png": "1b320f5c8ffebb0649d259a21e4ca57d814dab8884d3e03bff6814a06bfe7f5c",
    "outputs/figures/panels/exploratory/phase4_run_d_sensitivity_coefficients.png": "20aef606a17f9ca5e11eeb2782a35433722df24ad2883721d8cb64283184fb78",
    "outputs/tables/descriptive/final/t1_core_kpis.csv": "ad8cb92db9d70edd55004fa16d142a5960e652671008da3177e5dc3d124c12ea",
    "outputs/tables/descriptive/final/t2_commodity_diagnostics.csv": "f030f4383aa60940f7fa6a54a2a73465c3143675eefec0e1b2bf6062e7ca446a",
    "outputs/tables/descriptive/final/t3_campaign_reconnaissance.csv": "8c474a858620484296f4973295b98234b7503cde694dd96a079efec77976d3bb",
    "outputs/tables/panel/final/t4_core_panel_results.csv": "8f9c7bebc93591f3fc8e017c772178ffaf008291f9f9f1390947599ac60b61b1",
    "outputs/tables/panel/final/t5_panel_robustness.csv": "856cf3b3af7abf42f7d27812cd86c331a0334c25b9e4d28fbe06f43a84a46936",
    "outputs/tables/panels/contracts.json": "956a5fe2881baed9d14d8e3688b7b084d41bc6a7cfbbc9225004ee805c7e9616",
    "outputs/tables/panels/exploratory/phase4_run_c_core_results.csv": "cad9e708c607516e3e3b31452d64169c7a9eba429345e09262296fe6afc56ed0",
    "outputs/tables/panels/exploratory/phase4_run_c_diagnostics.csv": "2da35e6fa2e8da9417b474788705dc0e408b62aee0f49c7ed5d9cf276c53ad13",
    "outputs/tables/panels/exploratory/phase4_run_d_diagnostics.csv": "60e49fe0b506adfaf97e6599a567ec863d317b7f4ffb28e7e3ac390f3e17945a",
    "outputs/tables/panels/exploratory/phase4_run_d_sensitivity_matrix.csv": "d503fc5a52f7859547ac2cd6b1e58bdc81a2830b380b02c739d434832ec2e6f5",
    "outputs/tables/panels/phase4_preflight_provenance.json": "25d7003716cc10a45e86f74dcce0c7f14ef904f0691d46e5a57d1015f00c1c37",
    "outputs/tables/panels/phase4_preflight_summary.csv": "0297ead4f887d72ae553b125accddd149f6c8948df6c10069f8f9863191b6625",
    "outputs/tables/panels/phase4_sample_contract.json": "8ba9da8ab9cadb15965186fb4efa90c45d2aaae1b104bff7aa158e57a71a08b1",
}
IGNORE_APPEND = "\n\n# --- Phase 5 Run B: aggregate preparation artifacts only; no final results ---\n!outputs/tables/choice/\noutputs/tables/choice/**\n!outputs/tables/choice/contracts.json\n!outputs/tables/choice/qa_summary.csv\n!outputs/tables/choice/phase5_preflight_summary.csv\n!outputs/tables/choice/provenance.json\n"


UPSTREAM_IGNORE_SHA256 = "91b2e962bd468e90f3b71c002be3163d308625f3a54c462515535d25e07ccd67"


def source_paths(root: Path = PROJECT_ROOT) -> list[Path]:
    return sorted(
        [
            *(root / "src/complete_journey/choice").glob("*.py"),
            *(root / "sql/choice").glob("*.sql"),
            root / "scripts/run_choice.py",
            root / "pyproject.toml",
            root / "uv.lock",
            root / "docs/DECISIONS.md",
            root / ".gitignore",
            root / "src/complete_journey/data/raw_tables.py",
            root / "src/complete_journey/paths.py",
        ]
    )


# Run C authorizes exactly these ignored exploratory targets, not publication.
CORE_OUTPUTS = {
    "results": "outputs/tables/choice/exploratory/phase5_run_c_core_results.csv",
    "diagnostics": "outputs/tables/choice/exploratory/phase5_run_c_diagnostics.csv",
    "report": "outputs/tables/choice/exploratory/phase5_run_c_report.json",
    "covariance": "outputs/models/choice/exploratory/phase5_run_c_native_covariance.npz",
    "figure": "outputs/figures/choice/exploratory/phase5_run_c_history_contrasts.png",
    "provenance": "outputs/tables/choice/exploratory/phase5_run_c_provenance.json",
}
PREPARATION_EVIDENCE_SHA256 = {
    "data/processed/choice/purchased_composition_ledger.parquet": "0764119172f8b5d5a14400e8841b67fbdebbfa2c1827d509bcb1ef509a48667d",
    "data/processed/choice/purchased_composition_event.parquet": "8e28fda0244f86809475dc04f09bbd63dd8bdbb01994c1a568e8855620447832",
    "outputs/tables/choice/contracts.json": "5460b330c6c1f56b6a4c8c26859a03dded87ec2e1b6298b081b39caba1d038bf",
    "outputs/tables/choice/qa_summary.csv": "9884d5427a621a303d01cc97ba5a87d46fe88c82a1ff820ca9babf9e71164f08",
    "outputs/tables/choice/phase5_preflight_summary.csv": "03ce81b2269ec8195e6c0a4d7b2c4f7dfc0d423859e53577387b3d0e294d8f57",
}
REGISTER_PATH = "docs/phases/PHASE_05_RESULT_REGISTER.csv"
REGISTER_FIELDS = (
    "result_id",
    "statement",
    "evidence",
    "diagnostic_status",
    "interpretation_boundary",
    "status",
    "researcher_notes",
)
REGISTER_IDS = ("P5-RC-01", "P5-RC-02", "P5-RC-03", "P5-RC-04")
REGISTER_STATUSES = ("UNREVIEWED", "CHALLENGED", "FAILED")


RUN_B_PROVENANCE_SHA256 = "649022ed91b6edcc1cf86e3db41761c2004a75d01f41efb00eeffdc90f634efa"
PREPARATION_VALIDATION_PATH = (
    "outputs/tables/choice/exploratory/phase5_run_c_preparation_validation.json"
)

# Accepted at b74b6e15b28f3a72f67eb18903b00af0cfead93c, before Run D source evolution.
# Historical generation remains attributed to the exact pinned Run C manifest.
ACCEPTED_RUN_C_SHA256 = MappingProxyType(
    {
        CORE_OUTPUTS["results"]: "229b035f0b2e99fdef8b712ceaf2d4e8dd33646424444a1c8fd89aab869c9226",
        CORE_OUTPUTS[
            "diagnostics"
        ]: "a07af6b3287256ebf29b9469d24e917dbdc5f66c7da6b731e881ecc57e0d695e",
        CORE_OUTPUTS["report"]: "c0417d452d79839f3c00d72112375614298532ef19b6d27b6fcdb49f7493615c",
        CORE_OUTPUTS[
            "covariance"
        ]: "9945b3fc278216e47b0d5b1a10dc53d2da1088f657371ba32cb3d3231cb9126f",
        CORE_OUTPUTS["figure"]: "43d9c1655626d594a75aff29f952d0977d091106765bc634a52c20cdf283d3b3",
        CORE_OUTPUTS[
            "provenance"
        ]: "7e06057c4eb4a0d4226daed6589fc3272d648984fc841b17d04d53b02c07e481",
        PREPARATION_VALIDATION_PATH: "f0b452799b41e70c313c0bf633d96ac5f640c1b6c53454eb115f0490fb624be7",
    }
)

# Run D authorizes only these ignored exploratory targets, never a final allowlist.
RUN_D_OUTPUTS = {
    "report": "outputs/tables/choice/exploratory/phase5_run_d_report.json",
    "provenance": "outputs/tables/choice/exploratory/phase5_run_d_provenance.json",
    "sensitivity": "outputs/tables/choice/exploratory/phase5_run_d_sensitivity_matrix.csv",
    "covariance_table": "outputs/tables/choice/exploratory/phase5_run_d_covariance_comparison.csv",
    "covariance": "outputs/models/choice/exploratory/phase5_run_d_native_covariance.npz",
    "history_figure": "outputs/figures/choice/exploratory/phase5_run_d_history_robustness.png",
    "covariance_figure": "outputs/figures/choice/exploratory/phase5_run_d_covariance_comparison.png",
}

# Accepted Run D generation is historical, never attributed to Run E source.
ACCEPTED_RUN_D_SHA256 = MappingProxyType(
    {
        RUN_D_OUTPUTS["report"]: "431af5d688cac218675bc398df76f3779fc8b228b945af4688e978fc02939242",
        RUN_D_OUTPUTS[
            "provenance"
        ]: "97c33f5f0c8c75e8314feeb8a2180fcf196e21ff3eeceb8001d41b66ddca1fdb",
        RUN_D_OUTPUTS[
            "sensitivity"
        ]: "2212d9ce474c05d5e142f9a083120333d4899f0364c052d0a26708546ab2e59f",
        RUN_D_OUTPUTS[
            "covariance_table"
        ]: "c12d9bbaa752c77b132ef35832b0107976b99e7fb7c329a954836091c8e2317d",
        RUN_D_OUTPUTS[
            "covariance"
        ]: "577932d80e833336a1d871f863d140ec2c55cf50f34cb3660432c74718b85d6f",
        RUN_D_OUTPUTS[
            "history_figure"
        ]: "918c25d1fc5787040f9af757de121966b4f7403c5febf8af48fda3ea0ca41b08",
        RUN_D_OUTPUTS[
            "covariance_figure"
        ]: "21c676f1111dad5f2398b489062fad6c39d8a29b3d323cadf6d83e41016f992e",
    }
)
