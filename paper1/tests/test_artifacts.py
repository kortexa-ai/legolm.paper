import copy
import unittest

from legolm_p01.artifacts import milestones, expected_measurements, validate_population, verify_supplements


class EvidenceCoverage(unittest.TestCase):
    def test_complete_training_is_insufficient_for_a_final_release(self):
        with self.assertRaisesRegex(ValueError,"composition cells"):
            verify_supplements(None,{}, {}, {}, {},incomplete=False)
        preview=verify_supplements(None,{}, {}, {}, {},incomplete=True)
        self.assertEqual(preview["missing_composition_cells"],396)
        self.assertEqual(preview["missing_resource_measurements"],36)

    def test_long_imu_requires_intermediate_checkpoints_and_counterfactuals(self):
        r = dict(cohort="P", host="qwen08", modality="imu", kind="bridge", feature_mode="true", diversity=0., steps=6000)
        self.assertEqual(milestones(r), [0,100,300,600,1200,2400,6000])
        main = list(expected_measurements(r,6000,"imu").values())
        rankings = [m["settings"]["intervention"] for m in main if m["measurement"] == "ranking"]
        self.assertIn("different_class", rankings)
        self.assertIn("shuffle-4", rankings)
        self.assertEqual(sum(m["measurement"] == "generation" for m in main), 5)
        self.assertEqual({m["measurement"] for m in expected_measurements(r,1200,"imu").values()},
                         {"ranking", "variation", "functional_sensitivity"})
        frozen={**r,"cohort":"U","kind":"none","steps":0}
        measurements=list(expected_measurements(frozen,0,"imu").values())
        generation=next(m for m in measurements if m["measurement"]=="generation")
        self.assertEqual(generation["settings"],{"objective":"P","intervention":"correct"})

    def test_ranking_rejects_dropped_repeated_or_nonfinite_items(self):
        data = {"categories": ["a","b"], "test": {"ids": ["a","b"], "labels": [0,1], "groups": ["s1","s2"]}}
        evidence = {"identity": {"measurement": "ranking"}, "items": [
            {"id": "a", "label": 0, "group": "s1", "scores": [1.,0.]},
            {"id": "b", "label": 1, "group": "s2", "scores": [0.,1.]}]}
        validate_population(evidence,data)
        for bad in (evidence["items"][:1], [evidence["items"][0]]*2):
            with self.assertRaises(ValueError):
                validate_population({**evidence,"items":bad},data)
        bad=copy.deepcopy(evidence);bad["items"][0]["scores"][0]=float("nan")
        with self.assertRaises(ValueError):
            validate_population(bad,data)

    def test_generation_diagnostic_and_instruction_panels_have_distinct_sizes(self):
        data={"modality":"imu","test":{"ids":[str(i) for i in range(300)],"labels":[0]*300,"groups":["s1"]*300}}
        rows=[{"id":str(i),"label":0,"group":"s1","semantic_correct":False,"format_compliant":True,"joint_success":False} for i in range(300)]
        evidence={"identity":{"measurement":"generation","objective":"P"},"items":rows}
        validate_population(evidence,data)
        evidence["identity"]["objective"]="G"
        with self.assertRaises(ValueError):
            validate_population(evidence,data)
        evidence["items"]=[r for r in rows for _ in range(16)]
        validate_population(evidence,data)


if __name__ == "__main__":
    unittest.main()
