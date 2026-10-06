"""Numerical regression tests for the current exact-field manuscript results."""
import json
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import check_exact_four_variants as orders
import check_finite_population_analytic as finite
import verify_hopf as hopf


class ExactFourCombinationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows,cls.summaries=orders.calculate()

    def test_orders_and_theoretical_limits(self):
        self.assertEqual(len(self.rows),16)
        expected=[(-1.996,-.997),(-1.993,-.994),(-1.976,-.985),(-1.957,-.972)]
        for summary,(shift,jac) in zip(self.summaries,expected):
            self.assertAlmostEqual(summary['displacement_slope'],shift,delta=.0006)
            self.assertAlmostEqual(summary['jacobian_slope'],jac,delta=.0006)
            self.assertLess(abs(summary['scaled_displacement']/summary['predicted_displacement']-1),.03)
            self.assertLess(abs(summary['scaled_jacobian']/summary['predicted_jacobian']-1),.02)

    def test_roots_derivatives_and_convergence(self):
        for row in self.rows:
            self.assertLess(row['residual'],1e-12)
            self.assertLess(row['derivative_check'],2e-8)
        for summary in self.summaries:
            group=[r for r in self.rows if r['kernel']==summary['kernel'] and r['sampling']==summary['sampling']]
            for key,target in [('scaled_shift',summary['predicted_displacement']),('scaled_jacobian',summary['predicted_jacobian'])]:
                errors=[abs(r[key]-target) for r in group]
                self.assertTrue(all(a>b for a,b in zip(errors,errors[1:])))

    def test_stored_output_matches_calculation(self):
        stored=json.loads((ROOT/'results/exact_four_variants/summary.json').read_text())['variants']
        for actual,reference in zip(self.summaries,stored):
            for key in ['displacement_slope','jacobian_slope','scaled_displacement','scaled_jacobian']:
                self.assertAlmostEqual(actual[key],reference[key],delta=2e-5)


class FinitePopulationAnalyticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows,cls.summary=finite.calculate()

    def test_manuscript_spectrum_and_audits(self):
        target=[.006937,.006731,.006464,.006506,.004490,.005464,.005269,.005267]
        self.assertEqual([r['N'] for r in self.rows],[50,100,200,500,1000,2000,5000,10000])
        for row,value in zip(self.rows,target):
            self.assertAlmostEqual(row['spectral_abscissa'],value,delta=5.1e-7)
            self.assertLess(row['derivative_check'],1e-8)
            self.assertLess(row['generator_implementation_gap'],1e-12)
        self.assertAlmostEqual(self.summary['continuum_spectral_abscissa'],.00528,delta=5.1e-6)
        self.assertAlmostEqual(self.summary['classical_spectral_abscissa'],-.00445,delta=5.1e-6)

    def test_mass_conservation_and_jacobian_off_center(self):
        for n in [50,1000,None]:
            y=np.array([.52,.40]);f,j=finite.drift_jac(y,n)
            self.assertLess(abs(sum(f)),1e-13)
            h=1e-6
            fd=np.column_stack([(finite.drift_jac(y+h*e,n)[0][:2]-finite.drift_jac(y-h*e,n)[0][:2])/(2*h) for e in np.eye(2)])
            np.testing.assert_allclose(j,fd,atol=1e-8,rtol=1e-8)


class HopfClassificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.points=hopf.verify_hopf()

    def test_boundaries_frequencies_and_lyapunov_coefficients(self):
        for point,beta,omega,l1 in zip(self.points,[.761225162,.800304099],[.510735502,.504050267],[.030551706,.037060946]):
            self.assertAlmostEqual(point['beta'],beta,delta=2e-7)
            self.assertAlmostEqual(point['omega'],omega,delta=2e-7)
            self.assertAlmostEqual(point['l1'],l1,delta=2e-6)
            self.assertEqual(point['classification'],'subcritical')
            self.assertGreater(point['l1'],0)
            self.assertLess(point['stationary_residual'],1e-12)
            self.assertAlmostEqual(point['qnorm'],1,delta=1e-12)
            np.testing.assert_allclose(point['pq'],[1,0],atol=1e-12)

    def test_transversality_and_derivative_audits(self):
        for point,sign in zip(self.points,[1,-1]):
            self.assertTrue(all(sign*s>5e-4 for s in point['branch_slopes']))
            self.assertLess(np.ptp(point['branch_slopes']),1e-8)
            self.assertLess(point['independent_field_gap'],1e-12)
            self.assertLess(point['independent_jacobian_gap'],1e-7)
            audits=point['derivative_audits']
            self.assertLess(audits[-1]['l1_difference'],1e-5)
            self.assertTrue(all(a['l1_from_differenced_derivatives']>0 for a in audits))


if __name__=='__main__':
    unittest.main()
