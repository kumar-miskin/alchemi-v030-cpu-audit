#!/usr/bin/env python3
"""Audit the actual v0.3.0 offline example, not a substitute implementation."""
import argparse
import ast
import contextlib
import copy
import hashlib
import io
import json
import platform
import subprocess
from pathlib import Path

import torch
import nvalchemi

RELEASE = '77d3c9e930a3eab5f44b61a7ffaf161609e13be2'


def run_once(source):
    # Inject observations immediately before the original training call.
    # All example definitions, labeling, loss, and training are left intact.
    tree = ast.parse(source.read_text())
    hook = ast.parse('audit_before(globals())').body[0]
    for index, node in enumerate(tree.body):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            if ast.unparse(node.value.func) == 'strategy.run':
                tree.body.insert(index, hook)
                break
    else:
        raise RuntimeError('Original training call not found')
    saved = {}

    def audit_before(ns):
        teacher = ns['teacher']
        student = ns['strategy'].models['student']
        saved['teacher'] = copy.deepcopy(teacher.state_dict())
        saved['student'] = copy.deepcopy(student.state_dict())
        saved['calls'] = 0

        def count(*unused):
            saved['calls'] += 1
        saved['handle'] = teacher.register_forward_hook(count)
        saved['holdout'] = ns['build_dataset'](24, ns['NUM_ATOMS'], 133)
        saved['before'] = errors(ns, student, saved['holdout'])
        saved['calls'] = 0  # Exclude explicit holdout scoring, count training only.

    ns = {'__name__': '__main__', '__file__': str(source), 'audit_before': audit_before}
    with contextlib.redirect_stdout(io.StringIO()) as output:
        exec(compile(ast.fix_missing_locations(tree), str(source), 'exec'), ns)
    teacher, student = ns['teacher'], ns['strategy'].models['student']
    training_teacher_calls = saved['calls']
    saved['handle'].remove()
    after = errors(ns, student, saved['holdout'])
    teacher_unchanged = all(torch.equal(v, saved['teacher'][k]) for k, v in teacher.state_dict().items())
    student_changed = any(not torch.equal(v, saved['student'][k]) for k, v in student.state_dict().items())
    losses = ns['trace'].losses
    assert training_teacher_calls == 0
    assert teacher_unchanged and student_changed
    assert len(losses) == 60 and all(torch.isfinite(torch.tensor(losses)))
    assert all(after[k] < saved['before'][k] for k in after)
    return {'loss_trace': losses, 'first_epoch_loss': ns['first'], 'last_epoch_loss': ns['last'],
            'teacher_calls_during_training': training_teacher_calls,
            'teacher_parameters_unchanged': teacher_unchanged, 'student_parameters_changed': student_changed,
            'holdout_systems': 24, 'holdout_seed': 133, 'holdout_mse_before': saved['before'],
            'holdout_mse_after': after, 'stdout': output.getvalue()}


def errors(ns, student, dataset):
    """Pool scalar components per quantity; these are teacher errors, not DFT."""
    batch = ns['Batch'].from_data_list([dataset[i][0] for i in range(len(dataset))])
    with torch.no_grad():
        target = ns['teacher'](batch)
        prediction = student(batch)
    return {key: float(((prediction[key] - target[key]) ** 2).mean())
            for key in ('energy', 'forces', 'atomic_energies')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('results/audit.json'))
    args = parser.parse_args()
    sha = subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip()
    assert sha == RELEASE, (sha, RELEASE)
    assert subprocess.check_output(['git', '-C', str(args.source), 'status', '--porcelain'], text=True).strip() == ''
    source = args.source / 'examples/intermediate/09_offline_distillation.py'
    installed = Path(nvalchemi.__file__).resolve().parent
    source_files = list((args.source / 'nvalchemi').rglob('*.py'))
    assert nvalchemi.__version__ == '0.3.0'
    for item in source_files:
        relative = item.relative_to(args.source / 'nvalchemi')
        assert item.read_bytes() == (installed / relative).read_bytes(), str(relative)
    one, two = run_once(source), run_once(source)
    assert {k: v for k, v in one.items() if k != 'stdout'} == {k: v for k, v in two.items() if k != 'stdout'}, 'The repeated CPU numeric result is not bit-for-bit reproducible'
    result = {'release_commit': sha, 'example_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'python': platform.python_version(), 'torch': torch.__version__,
              'cuda_available': torch.cuda.is_available(), 'cuda_device_count': torch.cuda.device_count(),
              'installed_source_files_matched': len(source_files), 'repeat_identical': True, 'run': one}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
