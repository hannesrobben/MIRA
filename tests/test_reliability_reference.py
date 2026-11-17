"""compute_icc against a reference implementation (MI-76, CLAUDE.md rule 8).

The reference is `pingouin.intraclass_corr` with rounding switched off.
pingouin names the two-way forms after McGraw & Wong (1996): our ICC(2,1)
is its "ICC(A,1)", our ICC(3,1) its "ICC(C,1)". pingouin is a `dev`
dependency; without it the reference tests are skipped.
"""

import numpy as np
import numpy.ma as ma
import pytest

from mira.analysis.reliability import compute_icc, compute_icc_from_dataframe
from mira.utils.exceptions import ValidationError

PINGOUIN_TYPE = {"ICC(1,1)": "ICC(1,1)", "ICC(2,1)": "ICC(A,1)", "ICC(3,1)": "ICC(C,1)"}
RTOL = 1e-12  # observed: <= 3e-14

# pingouin 0.7 calls a pandas API that pandas 3 deprecates; not our code.
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning:pingouin")


def _random_table(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n, k = int(rng.integers(3, 25)), int(rng.integers(2, 8))
    subjects = rng.normal(size=(n, 1)) * rng.uniform(0.0, 3.0)
    raters = rng.normal(size=(1, k)) * rng.uniform(0.0, 2.0)
    return subjects + raters + rng.normal(size=(n, k))


def _pingouin_table(table: np.ndarray, alpha: float):
    pg = pytest.importorskip("pingouin")
    pd = pytest.importorskip("pandas")
    from pingouin.config import _no_rounding

    n, k = table.shape
    df = pd.DataFrame(
        [(i, j, table[i, j]) for i in range(n) for j in range(k)],
        columns=["subject", "rater", "value"],
    )
    assert alpha == 0.05, "pingouin only reports a 95 % interval"
    with _no_rounding():
        return pg.intraclass_corr(
            df, targets="subject", raters="rater", ratings="value"
        ).set_index("Type")


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("icc_type", sorted(PINGOUIN_TYPE))
def test_matches_pingouin(seed, icc_type):
    table = _random_table(seed)
    reference = _pingouin_table(table, alpha=0.05).loc[PINGOUIN_TYPE[icc_type]]

    result = compute_icc(table, icc_type=icc_type, alpha=0.05)

    assert result.value == pytest.approx(reference["ICC"], rel=RTOL)
    assert result.f_value == pytest.approx(reference["F"], rel=RTOL)
    assert result.df1 == reference["df1"]
    assert result.df2 == reference["df2"]
    assert result.p_value == pytest.approx(reference["pval"], rel=1e-9, abs=1e-15)
    lower, upper = reference["CI95"]
    assert -1.0 <= lower <= 1.0 and -1.0 <= upper <= 1.0  # no clipping involved
    assert result.ci_lower == pytest.approx(lower, rel=RTOL, abs=1e-14)
    assert result.ci_upper == pytest.approx(upper, rel=RTOL, abs=1e-14)


def test_shrout_fleiss_example():
    # Shrout & Fleiss (1979), Table 2: 6 targets x 4 judges; the paper
    # reports ICC(1,1) = .17, ICC(2,1) = .29, ICC(3,1) = .71.
    table = np.array(
        [
            [9, 2, 5, 8],
            [6, 1, 3, 2],
            [8, 4, 6, 8],
            [7, 1, 2, 6],
            [10, 5, 6, 9],
            [6, 2, 4, 7],
        ],
        dtype=float,
    )
    expected = {"ICC(1,1)": 0.17, "ICC(2,1)": 0.29, "ICC(3,1)": 0.71}
    for icc_type, value in expected.items():
        assert round(compute_icc(table, icc_type=icc_type).value, 2) == value


def test_pingouin_wine_dataset():
    pg = pytest.importorskip("pingouin")
    data = pg.read_dataset("icc")  # 8 wines x 4 judges, shipped with pingouin
    table = data.pivot(index="Wine", columns="Judge", values="Scores").to_numpy()
    reference = _pingouin_table(table, alpha=0.05)
    for icc_type, name in PINGOUIN_TYPE.items():
        result = compute_icc(table, icc_type=icc_type)
        assert result.value == pytest.approx(reference.loc[name, "ICC"], rel=RTOL)


@pytest.mark.parametrize(
    "table",
    [
        np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]]),
        np.array([[1.0, 2.0], [3.0, np.inf], [5.0, 6.0]]),
        ma.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], mask=[[0, 0], [0, 1], [0, 0]]),
    ],
    ids=["nan", "inf", "masked"],
)
def test_incomplete_table_is_rejected(table):
    with pytest.raises(ValidationError, match="complete"):
        compute_icc(table)


def test_masked_array_without_masked_cells_is_accepted():
    table = _random_table(0)
    assert compute_icc(ma.array(table)).value == compute_icc(table).value


def test_incomplete_dataframe_is_rejected():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(
        {
            "segment_id": [0, 0, 1, 1, 2],
            "print_id": ["a", "b", "a", "b", "a"],
            "mean_relative": [1.0, 1.1, 2.0, 2.1, 3.0],
        }
    )
    with pytest.raises(ValidationError, match="not complete"):
        compute_icc_from_dataframe(df)
