import unittest
import numpy as np

from legolm_p01.statistics import accuracy_interval,ranking_interval,likelihood_interval,generation_interval,ranking_score_diagnostics


class StatisticsContracts(unittest.TestCase):
    def test_score_diagnostics_are_stable_under_large_common_offsets(self):
        rows=[{"scores":[0.,0.],"label":0},{"scores":[0.,2.],"label":1}]
        original=ranking_score_diagnostics(rows)
        shifted=ranking_score_diagnostics([{**r,"scores":[v+1000000 for v in r["scores"]]} for r in rows])
        self.assertEqual(original,shifted)
        self.assertEqual(original["mean_correct_label_margin"],1.)
        tied=ranking_score_diagnostics(rows[:1])
        self.assertAlmostEqual(tied["mean_candidate_score_entropy_nats"],np.log(2))

    def test_cluster_copies_do_not_create_independent_evidence(self):
        values=np.array([[1,0,1,0,1,0],[1,1,0,0,1,0]],dtype=float)
        labels=[0,1,0,1,0,1];groups=["s1","s1","s2","s2","s3","s3"]
        a=accuracy_interval(values,labels,groups,replicates=1000,seed=7)
        b=accuracy_interval(np.repeat(values,5,axis=1),np.repeat(labels,5),np.repeat(groups,5),replicates=1000,seed=7)
        np.testing.assert_allclose(a["micro"]["ci95"],b["micro"]["ci95"])
        np.testing.assert_allclose(a["macro"]["ci95"],b["macro"]["ci95"])
        self.assertEqual(a["clusters"],b["clusters"])

    def test_identical_paired_conditions_have_zero_difference(self):
        rows=[{"id":str(i),"label":i%2,"group":str(i//2),"scores":[1,0]} for i in range(20)]
        r=ranking_interval([rows,rows],comparison=[rows,rows],replicates=500)
        self.assertEqual(r["macro"]["ci95"],[0.,0.])
        self.assertEqual(r["micro"]["mean"],0.)

    def test_likelihood_weights_bytes_and_keeps_pairing(self):
        rows=[{"row":0,"nats":2.,"bytes":1},{"row":1,"nats":4.,"bytes":3}]
        r=likelihood_interval([rows],replicates=500)
        self.assertAlmostEqual(r["mean"],6/(np.log(2)*4))
        delta=likelihood_interval([rows],comparison=[rows],replicates=500)
        self.assertEqual(delta["ci95"],[0.,0.])
        with self.assertRaises(ValueError):
            likelihood_interval([rows],comparison=[[{**rows[0],"bytes":2},rows[1]]],replicates=10)

    def test_repeated_generation_prompts_are_one_sensor_unit(self):
        rows=[{"id":str(i),"label":i%2,"group":str(i),"family":"label","template_split":"test","joint_success":i%2==0}
              for i in range(20) for _ in range(3)]
        r=generation_interval([rows],"label","test",replicates=500)
        self.assertEqual(r["sensors"],20)
        self.assertEqual(r["micro"]["mean"],.5)

    def test_constant_answer_fails_the_two_mapping_rule_diagnostic(self):
        rows=[{"id":str(i),"label":i%2,"group":str(i),"family":"rule","template_split":"test",
               "template":template,"query_label":i%2,"reverse_rule":reverse,"joint_success":not reverse}
              for i in range(20) for template in range(3) for reverse in (False,True)]
        ordinary=generation_interval([rows],"rule","test",replicates=500)
        paired=generation_interval([rows],"rule","test","both_rule_mappings",replicates=500)
        self.assertEqual(ordinary["macro"]["mean"],.5)
        self.assertEqual(paired["macro"]["mean"],0.)
        self.assertEqual(paired["sensors"],20)
        with self.assertRaisesRegex(ValueError,"missing its reversed"):
            generation_interval([rows[:-1]],"rule","test","both_rule_mappings",replicates=10)


if __name__=="__main__":
    unittest.main()
