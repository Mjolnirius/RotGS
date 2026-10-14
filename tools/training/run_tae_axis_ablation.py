"""Launch fresh R078-matched axis-only experiments sequentially."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path('/shared/datasets/3D_Scanning_MT/output_RotGS/session_04')
BASELINE = OUTPUT_ROOT/'4_Quoellfrisch_Top_30d_SAM31AC_1435_TAE_wo_tiny_no_global_30k/cfg_args'
VARIANTS = {
    'legacy': {'axis_constraint_mode':'clamp', 'tae_axis_wiggle_deg':.5, 'freeze_axis':False},
    'fixed': {'axis_constraint_mode':'clamp', 'tae_axis_wiggle_deg':.5, 'freeze_axis':True},
    'projected': {'axis_constraint_mode':'project', 'tae_axis_wiggle_deg':.5, 'freeze_axis':False},
    'wide': {'axis_constraint_mode':'project', 'tae_axis_wiggle_deg':2., 'freeze_axis':False},
}


def baseline_config(path):
    call = ast.parse(path.read_text(),mode='eval').body
    if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Name) or call.func.id!='Namespace' or call.args:
        raise ValueError('Expected cfg_args to contain a Namespace with literal keyword values')
    return {item.arg:ast.literal_eval(item.value) for item in call.keywords}


def arguments_for(config):
    result=[]
    for key,value in config.items():
        if value is None or key.startswith('tae_') and key!='tae_axis_wiggle_deg':
            continue
        if isinstance(value,bool):
            if value:result.append('--'+key)
        elif isinstance(value,list):
            result.extend(['--'+key,*map(str,value)])
        else:
            result.extend(['--'+key,str(value)])
    return result


def configurations(baseline,output_root,variants,wandb_mode):
    for variant in variants:
        cfg=dict(baseline)
        cfg.update(VARIANTS[variant])
        output=output_root/f'4_Quoellfrisch_Top_30d_SAM31AC_1435_TAE_axis_{variant}_30k'
        cfg.update(name=str(output),model_path=str(output),start_checkpoint=None,
                   wandb_mode=wandb_mode,wandb_run_id=None,wandb_artifacts='none',
                   wandb_run_name=f'quoellfrisch-top-s04-r078-axis-{variant}',
                   wandb_group='r078-axis-ablation-20261014',
                   wandb_tags=['axis-ablation','r078-matched',variant],
                   wandb_notes=f'Fresh R078-matched run. Axis variant: {variant}. Measured angles fixed; SH unchanged.')
        yield variant,output,cfg


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=(*VARIANTS,'all'),default='all')
    parser.add_argument('--baseline-config',type=Path,default=BASELINE)
    parser.add_argument('--output-root',type=Path,default=OUTPUT_ROOT)
    parser.add_argument('--wandb-mode',choices=('disabled','offline','online'),default='offline')
    parser.add_argument('--dry-run',action='store_true',help='Print exact training commands without writing or training')
    args=parser.parse_args()
    baseline=baseline_config(args.baseline_config)
    variants=list(VARIANTS) if args.variant=='all' else [args.variant]
    runs=list(configurations(baseline,args.output_root.resolve(),variants,args.wandb_mode))
    for _,output,cfg in runs:
        if output.exists() and any(output.iterdir()):
            parser.error(f'Refusing to overwrite existing run: {output}')
        if not Path(cfg['source_path']).is_dir():
            parser.error(f'Dataset not found: {cfg["source_path"]}')
    source_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                   for name in ['train.py','scene/gaussian_model.py','scene/tae_calibration.py']}
    for variant,output,cfg in runs:
        command=[sys.executable,str(ROOT/'train.py'),*arguments_for(cfg)]
        print(f'\n{variant}: {output}',flush=True)
        print(shlex.join(command),flush=True)
        if args.dry_run:continue
        output.mkdir(parents=True,exist_ok=True)
        (output/'axis_experiment.json').write_text(json.dumps({
            'variant':variant,'baseline_config':str(args.baseline_config.resolve()),
            'baseline_config_sha256':hashlib.sha256(args.baseline_config.read_bytes()).hexdigest(),
            'axis_overrides':VARIANTS[variant],'seed':0,'command':command,
            'source_sha256':source_hashes,
        },indent=2)+'\n')
        subprocess.run(command,cwd=ROOT,check=True)


if __name__=='__main__':main()
