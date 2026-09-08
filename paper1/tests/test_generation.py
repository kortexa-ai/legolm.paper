import unittest

from legolm_p01.generation import case,score,panel_indices,evaluation_cases,training_rows


class GenerationContracts(unittest.TestCase):
    def test_rule_counterfactual_changes_response_with_same_sensor(self):
        data={"modality":"imu","categories":["walking","walking_upstairs"]}
        a=case(data,1,"rule","test",0,query=1,reverse=False)
        b=case(data,1,"rule","test",0,query=1,reverse=True)
        self.assertNotEqual(a["target"],b["target"])
        self.assertEqual(a["label"],b["label"])
        self.assertTrue(score(a["target"],a,data)["joint_success"])
        self.assertFalse(score(a["target"],b,data)["joint_success"])

    def test_semantics_and_format_fail_independently(self):
        data={"modality":"imu","categories":["walking","walking_upstairs"]}
        item=case(data,1,"label","test",0,query=1)
        result=score("The activity is walking upstairs.",item,data)
        self.assertTrue(result["semantic_correct"])
        self.assertFalse(result["format_compliant"])
        self.assertFalse(result["joint_success"])
        self.assertTrue(score("walking_upstairs",item,data)["joint_success"])

    def test_panel_is_input_only_group_stratified_and_bounded(self):
        data={"modality":"imu","categories":["a","b"],
              "test":{"ids":list(map(str,range(400))),"labels":[i%2 for i in range(400)],"groups":[str(i%10) for i in range(400)]}}
        ids=panel_indices(data)
        self.assertEqual(len(ids),300)
        self.assertEqual(len(set(ids)),300)
        self.assertEqual(ids,panel_indices(data))
        rows=evaluation_cases(data,ids)
        self.assertEqual(len({r["sensor_index"] for r in rows}),300)
        self.assertEqual(len(rows),4800)


if __name__=="__main__":
    unittest.main()
