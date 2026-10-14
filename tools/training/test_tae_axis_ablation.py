import unittest
from pathlib import Path

from tools.training.run_tae_axis_ablation import BASELINE, configurations, baseline_config, arguments_for


class AxisAblationConfigTest(unittest.TestCase):
    @unittest.skipUnless(BASELINE.is_file(), "requires the local R078 baseline configuration")
    def test_variants_preserve_all_non_axis_numerical_settings(self):
        baseline=baseline_config(BASELINE)
        mutable={'axis_constraint_mode','tae_axis_wiggle_deg','freeze_axis',
                 'name','model_path','start_checkpoint'}
        runs=list(configurations(baseline,Path('/tmp/axis-ablation'),
                                 ['legacy','fixed','projected','wide'],'offline'))
        self.assertEqual(len({r[1] for r in runs}),4)
        for variant,output,cfg in runs:
            for key,value in baseline.items():
                if key not in mutable and not key.startswith('wandb_'):
                    self.assertEqual(cfg[key],value,(variant,key))
            self.assertIsNone(cfg['start_checkpoint'])
            self.assertEqual(cfg['iterations'],30000)
            self.assertEqual(cfg['densify_until_iter'],13000)
            self.assertEqual(cfg['center_axis_lr_max_steps'],12000)
            self.assertTrue(cfg['wo_tiny'])
            self.assertEqual(cfg['max_sweep_error_deg'],0.)
            self.assertEqual(cfg['center_max_offset'],.05)
            args=arguments_for(cfg)
            self.assertNotIn('--tae_directory',args)
            self.assertNotIn('--start_checkpoint',args)
            self.assertEqual('--freeze_axis' in args,variant=='fixed')


if __name__=='__main__':unittest.main()
