from pathlib import Path


def test_pseudo_gt_variable_names_are_spelled_correctly():
    source_path = Path(__file__).resolve().parents[1] / 'models' / 'CycleHJSuperSloMo.py'
    source = source_path.read_text(encoding='utf-8')

    assert 'pseudo_gt12' in source
    assert 'pseudo_gt23' in source
