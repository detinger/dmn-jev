import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from jevdmn.dmn.reader import read_tables, unary_test
from jevdmn.dmn.table import DecisionTable, Domain, InputSpec, Interval, Rule, c_not, to_feel
from jevdmn.dmn.tree2table import nice_threshold, tree_to_table
from jevdmn.dmn.verify import analyse, gaps, merge_rules, overlaps, with_default_rule
from jevdmn.dmn.writer import DecisionNode, InputNode, SourceNode, write_dmn
from jevdmn.features import Column, Encoding

ENUM = ("a", "b", "c", "d")


def _data(n, seed):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "kind": rng.choice(ENUM, n),
        "flag": rng.random(n) < 0.4,
        "level": rng.integers(0, 4, n),
        "p": np.round(rng.random(n), 3),
    })
    y = np.where((df.kind.isin(["a", "b"]) & df.flag) | (df.level >= 3), "yes",
                 np.where(df.p > 0.7, "maybe", "no"))
    flip = rng.random(n) < 0.1
    y = np.where(flip, rng.choice(["yes", "no", "maybe"], n), y)
    return df, y


ENC = Encoding([Column(f"kind={v}", "kind", "onehot", v) for v in ENUM]
               + [Column("flag", "flag", "bool"), Column("level", "level", "ordinal"), Column("p", "p", "real")])
SPECS = [InputSpec("kind", "Kind", Domain("enum", ENUM)), InputSpec("flag", "Flag", Domain("bool", (False, True))),
         InputSpec("level", "Level", Domain("ordinal", (0, 1, 2, 3))), InputSpec("p", "P", Domain("real", lo=0.0, hi=1.0))]


@pytest.fixture(scope="module")
def fitted():
    df, y = _data(3000, 1)
    X = ENC.transform(df)
    tree = DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=0).fit(X, y)
    table = tree_to_table(tree, ENC, SPECS, "out", "Out", X_train=X)
    return tree, table, df, X


def test_table_equals_tree_on_training_and_domain(fitted):
    tree, table, df, X = fitted
    assert (table.evaluate(df) == tree.predict(X)).all()
    probe, _ = _data(5000, 2)  # fresh points; real thresholds were snapped to short decimals
    pred_tree = tree.predict(ENC.transform(probe))
    agree = (table.evaluate(probe) == pred_tree).mean()
    assert agree > 0.995


def test_tree_table_is_unique_and_complete(fitted):
    _, table, _, _ = fitted
    assert overlaps(table) == []
    assert gaps(table) == []


def test_merge_preserves_semantics_and_reduces_rules(fitted):
    _, table, df, _ = fitted
    merged = merge_rules(table)
    assert merged.n_rules() <= table.n_rules()
    assert (merged.evaluate(df) == table.evaluate(df)).all()
    rep = analyse(merged)
    assert rep.overlaps_conflicting == 0 and rep.gap_boxes == 0


def test_default_rule_table(fitted):
    _, table, df, _ = fitted
    first = with_default_rule(merge_rules(table))
    assert first.hit_policy == "FIRST"
    assert (first.evaluate(df) == table.evaluate(df)).all()


def test_xml_roundtrip(tmp_path, fitted):
    _, table, df, _ = fitted
    table = merge_rules(table)
    path = write_dmn(tmp_path / "t.dmn", "defs", "Test",
                     inputs=[InputNode(s.name, s.label, s.type_ref) for s in table.inputs],
                     decisions=[DecisionNode(table, requires_inputs=[s.name for s in table.inputs])],
                     sources=[SourceNode("ks_jev", "Jev", "teacher", authority_for=["out"])])
    parsed = read_tables(path)["out"]
    assert (parsed.evaluate(df) == table.evaluate(df)).all()


def test_gap_detection():
    d = Domain("enum", ENUM)
    t = DecisionTable("o", "O", [InputSpec("k", "K", d)], ("x",), [Rule({"k": frozenset({"a", "b"})}, "x")])
    g = gaps(t)
    assert len(g) == 1 and g[0][0] == frozenset({"c", "d"})


def test_overlap_detection():
    d = Domain("real", lo=0, hi=1)
    r1 = Rule({"p": (Interval(0, 0.6, True, True),)}, "x")
    r2 = Rule({"p": (Interval(0.5, 1, True, True),)}, "y")
    t = DecisionTable("o", "O", [InputSpec("p", "P", d)], ("x", "y"), [r1, r2])
    assert overlaps(t) == [(0, 1, False)]


@pytest.mark.parametrize("cond,expected", [
    (frozenset({"a"}), '"a"'),
    (frozenset({"a", "b", "c"}), 'not("d")'),
])
def test_feel_enum(cond, expected):
    assert to_feel(Domain("enum", ENUM), cond) == expected


def test_feel_real_and_parser_agree():
    d = Domain("real", lo=0, hi=1)
    cond = c_not(d, (Interval(0.25, 0.5, False, True),))
    text = to_feel(d, cond)
    x = np.array([0.0, 0.25, 0.3, 0.5, 0.51, 1.0])
    assert unary_test(text)(x).tolist() == [True, True, False, False, True, True]


def test_nice_threshold():
    assert nice_threshold(0.8712, 0.8899) == 0.88
    assert nice_threshold(0.1, 0.2) == 0.1
