"""Run the current exact-order, Hopf, finite-N and matched-error checks."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figures',action='store_true',help='Also regenerate the four manuscript figures.')
    args=parser.parse_args()
    commands=[
        ['check_exact_nash_orders.py'],
        ['check_exact_four_variants.py'],
        ['check_finite_population_analytic.py'],
        ['verify_hopf.py'],
        ['check_matched_approximation_summary.py'],
    ]
    if args.figures:
        commands.extend([
            ['plot_model_mechanism_v3.py'],
            ['plot_two_strategy_convergence.py'],
            ['plot_manuscript_dynamics.py'],
            ['plot_value_checks_v3.py','--profile','value_full'],
        ])
    for command in commands:
        print('Running: python code/'+' '.join(command),flush=True)
        subprocess.run([sys.executable,str(ROOT/'code'/command[0]),*command[1:]],cwd=ROOT,check=True)

if __name__=='__main__':
    main()
