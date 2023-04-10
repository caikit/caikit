# Copyright The Caikit Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Local
from caikit.core.toolkit import F1Metrics, F1MetricsContainer, QualityEvaluator

# Unit Test Infrastructure
from tests.base import TestCaseBase
import caikit.core


class TestQualityEvaluation(TestCaseBase):
    def test___init__(self):
        q_er = caikit.core.toolkit.QualityEvaluator("abc", "def")
        self.assertEqual(q_er.gold, "abc")
        self.assertEqual(q_er.pred, "def")

    def test_multilabel_multiclass_evaluation(self):
        gold = [["a", "b", "c"], ["a"]]
        pred = [["b", "c"], ["b"]]

        def find_label_func(ele):
            return ele

        def find_label_data_func(data, label):
            return [ele for ele in data if ele == label]

        evaluation_type = caikit.core.toolkit.EvalTypes.MULTILABEL_MULTICLASS
        evaluator = caikit.core.toolkit.QualityEvaluator(gold, pred)
        output = evaluator.run(evaluation_type, find_label_func, find_label_data_func)

        self.assertEqual(output["per_class_confusion_matrix"]["b"]["precision"], 0.5)
        self.assertEqual(output["per_class_confusion_matrix"]["b"]["recall"], 1)
        self.assertEqual(output["per_class_confusion_matrix"]["a"]["precision"], 0)
        self.assertEqual(round(output["macro_precision"], 3), 0.5)
        self.assertEqual(round(output["macro_recall"], 3), 0.667)
        self.assertEqual(round(output["macro_f1"], 3), 0.556)
        self.assertEqual(round(output["micro_precision"], 3), 0.667)
        self.assertEqual(round(output["micro_recall"], 3), 0.5)
        self.assertEqual(round(output["micro_f1"], 3), 0.571)

    def test_hierarchical_multilabel_multiclass_evaluation_no_hierarchy(self):
        """Test that hierarchical eval is consistent with normal eval with no label hierarchy"""
        gold = [["a", "b", "c"], ["a"]]
        pred = [["b", "c"], ["b"]]

        def label_func_builder(level):
            def find_label_func(ele):
                return ele

            return find_label_func

        def label_data_func_builder(level):
            def find_label_data_func(data, label):
                return [ele for ele in data if ele == label]

            return find_label_data_func

        evaluation_type = (
            caikit.core.toolkit.EvalTypes.MULTILABEL_MULTICLASS_HIERARCHICAL
        )
        evaluator = caikit.core.toolkit.QualityEvaluator(gold, pred)
        full_output = evaluator.run(
            evaluation_type, label_func_builder, label_data_func_builder
        )
        for output in full_output.values():
            self.assertEqual(
                output["per_class_confusion_matrix"]["b"]["precision"], 0.5
            )
            self.assertEqual(output["per_class_confusion_matrix"]["b"]["recall"], 1)
            self.assertEqual(output["per_class_confusion_matrix"]["a"]["precision"], 0)
            self.assertEqual(round(output["macro_precision"], 3), 0.5)
            self.assertEqual(round(output["macro_recall"], 3), 0.667)
            self.assertEqual(round(output["macro_f1"], 3), 0.556)
            self.assertEqual(round(output["micro_precision"], 3), 0.667)
            self.assertEqual(round(output["micro_recall"], 3), 0.5)
            self.assertEqual(round(output["micro_f1"], 3), 0.571)

    def test_hierarchical_multilabel_multiclass_evaluation_with_hierarchy(self):
        """Test that hierarchical eval is consistent with normal eval with a label hierarchy"""
        gold = [
            [("A sentence about basketball players", "/sports/basketball/players")],
            [("A sentence about basketball", "/sports/basketball")],
            [("A sentence about sports", "/sports")],
        ]
        pred = [
            [("A sentence about basketball players", "/sports/basketball")],
            [("A sentence about basketball", "/sports/baseball")],
            [("A sentence about sports", "/sports")],
        ]
        expected_metrics = {
            # Matching up to L1 - Matches everything
            "level_1": {"overall_tp": 3, "overall_fp": 0, "overall_fn": 0},
            # Matching up to L2 - Matches first and last example, misses middle one
            "level_2": {"overall_tp": 2, "overall_fp": 1, "overall_fn": 1},
            # Full matching - only one total match
            "level_3": {"overall_tp": 1, "overall_fp": 2, "overall_fn": 2},
            "level_all": {"overall_tp": 1, "overall_fp": 2, "overall_fn": 2},
        }

        # Helper for slicing up to subhierarchy
        def slice_up_to_level(level, label):
            if level is None:
                return label
            return "/".join(label.split("/")[: level + 1])

        # Define the hierarchical builder functions
        def label_func_builder(level):
            def find_label_func(ele):
                return slice_up_to_level(level, ele[1])

            return find_label_func

        def label_data_func_builder(level):
            def find_label_data_func(data, gold_label):
                def matches_up_to_level(label1, label2):
                    return slice_up_to_level(level, label1) == slice_up_to_level(
                        level, label2
                    )

                text, pred_label = data[0]
                if matches_up_to_level(gold_label, pred_label):
                    return [text]
                return []

            return find_label_data_func

        # Run the evaluator
        evaluation_type = (
            caikit.core.toolkit.EvalTypes.MULTILABEL_MULTICLASS_HIERARCHICAL
        )
        evaluator = caikit.core.toolkit.QualityEvaluator(gold, pred)
        full_output = evaluator.run(
            evaluation_type,
            label_func_builder,
            label_data_func_builder,
            max_hierarchy_levels=3,
        )

        full_output_2_levels = evaluator.run(
            evaluation_type,
            label_func_builder,
            label_data_func_builder,
            max_hierarchy_levels=2,
        )

        # Full output has N levels + a "level_all"
        self.assertEqual(len(full_output), 4)  # 3 levels + level_all
        self.assertEqual(len(full_output_2_levels), 3)  # 2 levels + level_all

        # For each hierarchy level, ensure that generated metrics match
        for level_key, actual_level_metrics in full_output.items():
            exp_level_metrics = expected_metrics[level_key]
            for metric_key in exp_level_metrics:
                self.assertEqual(
                    exp_level_metrics[metric_key], actual_level_metrics[metric_key]
                )

    def test_singlelabel_multiclass_evaluation_no_labels(self):
        gold = ["a", "b", "a", "b"]
        pred = ["b", "b", "b", "b"]
        evaluator = caikit.core.toolkit.QualityEvaluator(gold, pred)
        output = evaluator.run(
            evaluation_type=caikit.core.toolkit.EvalTypes.SINGLELABEL_MULTICLASS
        )

        self.assertEqual(output["micro_f1"], 0.5)
        self.assertEqual(output["micro_precision"], 0.5)
        self.assertEqual(output["micro_recall"], 0.5)
        self.assertEqual(output["macro_precision"], 0.25)
        self.assertEqual(output["macro_recall"], 0.5)
        self.assertEqual(round(output["macro_f1"], 3), 0.333)
        self.assertEqual(output["per_class_confusion_matrix"]["a"]["precision"], 0.0)
        self.assertEqual(
            round(output["per_class_confusion_matrix"]["b"]["f1"], 3), 0.667
        )

    def test_bad_evaluation_type(self):
        with self.assertRaises(ValueError):
            gold = []
            pred = []
            evaluator = caikit.core.toolkit.QualityEvaluator(gold, pred)
            evaluator.run(evaluation_type="This is a bad evaluation type! :)")

    def test_multilabel_multiclass_evaluation_detailed_metrics(self):
        def find_label_func(*args):
            raise NotImplementedError("func not implemented")

        def find_label_data_func(*args):
            raise NotImplementedError("func not implemented")

        gold_annos = [["Tom Anniston", "Acme", "USA"]]
        pred_annos = [["Tom", "Acme", "USA since"]]
        evaluator = caikit.core.toolkit.QualityEvaluator(gold_annos, pred_annos)
        output = evaluator.run(
            caikit.core.toolkit.EvalTypes.MULTILABEL_MULTICLASS,
            find_label_func,
            find_label_data_func,
            detailed_metrics=True,
        )
        output = output["detailed_metrics"]
        self.assertEqual(round(output[0]["exact_match_precision"], 3), 0.333)
        self.assertEqual(round(output[0]["exact_match_recall"], 3), 0.333)
        self.assertEqual(round(output[0]["exact_match_f1"], 3), 0.333)
        self.assertEqual(output[0]["partial_match_precision"], 1)
        self.assertEqual(output[0]["partial_match_recall"], 1)
        self.assertEqual(output[0]["partial_match_f1"], 1)

    def test_find_partial_matches_basic_functionality(self):
        # When gold annotations are part of a prediction or vice versa, both annotations are counted as matches
        gold_annos = ["Tom Anniston", "Acme", "USA", "works"]
        pred_annos = ["Tom", "Acme", "USA since", "works at", "works at Acme"]

        expected_gold_matches = ["Tom Anniston", "Acme", "USA", "works"]
        expected_partial_pred_matches = [
            "Acme",
            "Tom",
            "USA since",
            "works at",
            "works at Acme",
        ]

        (
            actual_gold_matches,
            actual_pred_matches,
        ) = QualityEvaluator.find_partial_matches(gold_annos, pred_annos)
        self.assertEqual(
            sorted(actual_pred_matches), sorted(expected_partial_pred_matches)
        )
        self.assertEqual(sorted(actual_gold_matches), sorted(expected_gold_matches))

    def test_find_partial_matches_no_duplicates(self):
        # When a gold annotation is part of a prediction and prediction is also part of another gold annotation,
        # the same prediction should be counted only once, but both gold annotations are counted
        gold_annos = ["gian franco kasper", "ioc representative gian franco kasper"]
        pred_annos = ["representative gian franco kasper"]

        expected_gold_matches = [
            "gian franco kasper",
            "ioc representative gian franco kasper",
        ]
        expected_partial_pred_matches = ["representative gian franco kasper"]

        (
            actual_gold_matches,
            actual_pred_matches,
        ) = QualityEvaluator.find_partial_matches(gold_annos, pred_annos)
        self.assertEqual(
            sorted(actual_pred_matches), sorted(expected_partial_pred_matches)
        )
        self.assertEqual(sorted(actual_gold_matches), sorted(expected_gold_matches))

    def test_find_partial_matches_no_overlaps(self):
        # Overlaps are not matched, only subsets
        gold_annos = ["ioc representative gian franco kasper"]
        pred_annos = ["FIS president gian franco kasper"]

        expected_gold_matches = []
        expected_partial_pred_matches = []

        (
            actual_gold_matches,
            actual_pred_matches,
        ) = QualityEvaluator.find_partial_matches(gold_annos, pred_annos)
        self.assertEqual(
            sorted(actual_pred_matches), sorted(expected_partial_pred_matches)
        )
        self.assertEqual(sorted(actual_gold_matches), sorted(expected_gold_matches))

    def test_find_partial_matches_with_exact_match(self):
        # When an exact match is also part of another annotation, the other annotation should also be counted
        gold_annos = ["Acme", "Acme marketplace", "Bar"]
        pred_annos = ["Acme", "Foo", "Bar Giant"]

        expected_gold_matches = ["Acme", "Acme marketplace", "Bar"]
        expected_partial_pred_matches = ["Acme", "Bar Giant"]

        (
            actual_gold_matches,
            actual_pred_matches,
        ) = QualityEvaluator.find_partial_matches(gold_annos, pred_annos)
        self.assertEqual(
            sorted(actual_pred_matches), sorted(expected_partial_pred_matches)
        )
        self.assertEqual(sorted(actual_gold_matches), sorted(expected_gold_matches))

    def test_calc_f1_score(self):
        gold = ["a", "b", "c", "d"]
        pred = ["a", "b", "d", "e", "f"]
        precision, recall, f1_score = QualityEvaluator.calc_f1_score(gold, pred)
        expected_precision = 3 / 5
        expected_recall = 3 / 4

        self.assertEqual(precision, expected_precision)
        self.assertEqual(recall, expected_recall)
        self.assertEqual(round(f1_score, 3), 0.667)

    def test_calc_f1_score_error_partial(self):
        gold = ["a", "b", "c", "d"]
        pred = ["a", "b", "d", "e", "f"]

        def partial_match(gold, pred):
            return None

        with self.assertRaises(ValueError):
            precision, recall, f1_score = QualityEvaluator.calc_f1_score(
                gold, pred, partial_match
            )

    def test_calc_f1_score_partial(self):
        gold = ["a", "b", "c", "d"]
        pred = ["a", "b", "d", "e", "f"]

        def partial_match(gold, pred):
            # Returns all gold and predictions as matched
            return gold, pred

        precision, recall, f1_score = QualityEvaluator.calc_f1_score(
            gold, pred, partial_match
        )
        expected_precision = 5 / 5
        expected_recall = 4 / 4

        self.assertEqual(precision, expected_precision)
        self.assertEqual(recall, expected_recall)
        self.assertEqual(round(f1_score, 3), 1.0)

    def test_multilabel_multiclass_evaluation(self):
        gold = [["a", "b", "c"], ["a"]]
        pred = [["b", "c"], ["b"]]

        def find_label_func(ele):
            return ele

        def find_label_data_func(data, label):
            return [ele for ele in data if ele == label]

        evaluation_type = caikit.core.toolkit.EvalTypes.MULTILABEL_MULTICLASS
        evaluator = caikit.core.toolkit.QualityEvaluator(gold, pred)
        test_labels = ["a", "b"]
        output = evaluator.run(
            evaluation_type, find_label_func, find_label_data_func, labels=test_labels
        )

        # checking if only the labels we passed in are in the result
        check = [
            True
            for label in output["per_class_confusion_matrix"]
            if label in test_labels
        ]
        if False in check:
            check = False
        else:
            check = True

        self.assertTrue(check)

        self.assertEqual(output["per_class_confusion_matrix"]["b"]["precision"], 0.5)
        self.assertEqual(output["per_class_confusion_matrix"]["b"]["recall"], 1)
        self.assertEqual(output["per_class_confusion_matrix"]["a"]["precision"], 0)
        self.assertEqual(round(output["macro_precision"], 3), 0.25)
        self.assertEqual(round(output["macro_recall"], 3), 0.5)
        self.assertEqual(round(output["macro_f1"], 3), 0.333)
        self.assertEqual(round(output["micro_precision"], 3), 0.5)
        self.assertEqual(round(output["micro_recall"], 3), 0.333)
        self.assertEqual(round(output["micro_f1"], 3), 0.4)
        self.assertEqual(output["micro_precision_partial_match"], 0)
        self.assertEqual(output["micro_recall_partial_match"], 0)
        self.assertEqual(output["micro_f1_partial_match"], 0)

    def test_multilabel_multiclass_evaluation_partial_match(self):
        gold = [["United States", "California", "Colorado"], ["Liberty"]]
        pred = [["States", "California"], ["Statue Liberty"]]

        def find_label_func(ele):
            return ele

        def find_label_data_func(data, label):
            return [ele for ele in data if ele == label]

        evaluation_type = caikit.core.toolkit.EvalTypes.MULTILABEL_MULTICLASS
        evaluator = caikit.core.toolkit.QualityEvaluator(gold, pred)
        output = evaluator.run(
            evaluation_type,
            find_label_func,
            find_label_data_func,
            partial_match_metrics=True,
        )
        self.assertEqual(round(output["micro_precision_partial_match"], 3), 1)
        self.assertEqual(round(output["micro_recall_partial_match"], 3), 0.75)
        self.assertEqual(round(output["micro_f1_partial_match"], 3), 0.857)

    def test_singlelabel_multiclass_evaluation(self):
        gold = ["a", "b", "a", "b"]
        pred = ["b", "b", "b", "b"]
        evaluator = caikit.core.toolkit.QualityEvaluator(gold, pred)
        test_labels = ["b"]
        output = evaluator.run(
            evaluation_type=caikit.core.toolkit.EvalTypes.SINGLELABEL_MULTICLASS,
            labels=test_labels,
        )

        # checking if only the labels we passed in are in the result
        check = [
            True
            for label in output["per_class_confusion_matrix"]
            if label in test_labels
        ]
        if False in check:
            check = False
        else:
            check = True

        self.assertTrue(check)

        for label in test_labels:
            self.assertIn("true_positive", output["per_class_confusion_matrix"][label])
            self.assertIn("false_positive", output["per_class_confusion_matrix"][label])
            self.assertIn("false_negative", output["per_class_confusion_matrix"][label])
            self.assertIn("precision", output["per_class_confusion_matrix"][label])
            self.assertIn("recall", output["per_class_confusion_matrix"][label])
            self.assertIn("f1", output["per_class_confusion_matrix"][label])

        self.assertEqual(output["macro_precision"], 0.5)
        self.assertEqual(output["macro_recall"], 1.0)
        self.assertEqual(round(output["macro_f1"], 3), 0.667)
        self.assertEqual(output["micro_precision"], 0.5)
        self.assertEqual(output["micro_recall"], 1.0)
        self.assertEqual(round(output["micro_f1"], 3), 0.667)

    def test_multilabel_multiclass_evaluation_with_empty_labels(self):
        gold = [["a", "b"], ["b"], [], ["a"]]
        pred = [["a"], ["b"], ["a"], []]

        def find_label_func(ele):
            return ele

        def find_label_data_func(data, label):
            return [ele for ele in data if ele == label]

        evaluation_type = caikit.core.toolkit.EvalTypes.MULTILABEL_MULTICLASS
        evaluator = caikit.core.toolkit.QualityEvaluator(gold, pred)
        output = evaluator.run(evaluation_type, find_label_func, find_label_data_func)
        self.assertEqual(output["per_class_confusion_matrix"]["b"]["precision"], 1)
        self.assertEqual(output["per_class_confusion_matrix"]["b"]["recall"], 0.5)
        self.assertEqual(output["per_class_confusion_matrix"]["a"]["precision"], 0.5)
        self.assertEqual(output["per_class_confusion_matrix"]["a"]["recall"], 0.5)
        self.assertEqual(output["per_class_confusion_matrix"]["a"]["true_positive"], 1)
        self.assertEqual(output["per_class_confusion_matrix"]["a"]["false_positive"], 1)
        self.assertEqual(output["per_class_confusion_matrix"]["a"]["false_negative"], 1)
        self.assertEqual(output["per_class_confusion_matrix"]["b"]["true_positive"], 1)
        self.assertEqual(output["per_class_confusion_matrix"]["b"]["false_positive"], 0)
        self.assertEqual(output["per_class_confusion_matrix"]["b"]["false_negative"], 1)
        self.assertEqual(round(output["macro_precision"], 3), 0.75)
        self.assertEqual(round(output["macro_recall"], 3), 0.5)
        self.assertEqual(round(output["macro_f1"], 3), 0.583)
        self.assertEqual(round(output["micro_precision"], 3), 0.667)
        self.assertEqual(round(output["micro_recall"], 3), 0.5)
        self.assertEqual(round(output["micro_f1"], 3), 0.571)
        self.assertEqual(round(output["micro_precision_partial_match"], 3), 0)
        self.assertEqual(round(output["micro_recall_partial_match"], 3), 0)
        self.assertEqual(round(output["micro_f1_partial_match"], 3), 0)

    def test_multilabel_multiclass_evaluation_with_no_find_label_func(self):
        gold = [["a", "b"], ["b"], [], ["a"]]
        pred = [["a"], ["b"], ["a"], []]

        def find_label_func(*_args, **_kwargs):
            raise NotImplementedError("Func not implemented")

        def find_label_data_func(*_args, **_kwargs):
            raise NotImplementedError("Func not implemented")

        evaluation_type = caikit.core.toolkit.EvalTypes.MULTILABEL_MULTICLASS
        evaluator = caikit.core.toolkit.QualityEvaluator(gold, pred)
        output = evaluator.run(evaluation_type, find_label_func, find_label_data_func)
        # should calculate only micro metrics
        self.assertFalse(output["per_class_confusion_matrix"])
        self.assertEqual(output["macro_precision"], 0.0)
        self.assertEqual(output["macro_recall"], 0.0)
        self.assertEqual(output["macro_f1"], 0.0)
        self.assertEqual(round(output["micro_precision"], 3), 0.667)
        self.assertEqual(round(output["micro_recall"], 3), 0.5)
        self.assertEqual(round(output["micro_f1"], 3), 0.571)
        self.assertEqual(round(output["micro_precision_partial_match"], 3), 0)
        self.assertEqual(round(output["micro_recall_partial_match"], 3), 0)
        self.assertEqual(round(output["micro_f1_partial_match"], 3), 0)

    def test_multilabel_multiclass_evaluation_with_edge_case_zero_values(self):
        # true positives would be 0, resulting in 0 in denominator of F1 calculation
        gold = [["a", "b"], ["b"], [], ["a"]]
        pred = [[], ["a"], [], ["b"]]

        def find_label_func(*_args, **_kwargs):
            raise NotImplementedError("Func not implemented")

        def find_label_data_func(*_args, **_kwargs):
            raise NotImplementedError("Func not implemented")

        evaluation_type = caikit.core.toolkit.EvalTypes.MULTILABEL_MULTICLASS
        evaluator = caikit.core.toolkit.QualityEvaluator(gold, pred)
        output = evaluator.run(evaluation_type, find_label_func, find_label_data_func)
        # should calculate only micro metrics
        self.assertFalse(output["per_class_confusion_matrix"])
        self.assertEqual(output["macro_precision"], 0.0)
        self.assertEqual(output["macro_recall"], 0.0)
        self.assertEqual(output["macro_f1"], 0.0)
        self.assertEqual(round(output["micro_precision"], 3), 0.0)
        self.assertEqual(round(output["micro_recall"], 3), 0.0)
        self.assertEqual(round(output["micro_f1"], 3), 0.0)
        self.assertEqual(round(output["micro_precision_partial_match"], 3), 0)
        self.assertEqual(round(output["micro_recall_partial_match"], 3), 0)
        self.assertEqual(round(output["micro_f1_partial_match"], 3), 0)

    def test_calc_metrics_from_confusion_matrix(self):
        # gold = ['a', 'b', 'a', 'a']
        # pred = ['a', 'b', 'b' , 'b']

        confusion_matrix = {
            "a": F1Metrics(true_positive=1, false_positive=0, false_negative=2),
            "b": F1Metrics(true_positive=1, false_positive=2, false_negative=0),
        }

        metrics_summary = QualityEvaluator.calc_metrics_from_confusion_matrix(
            confusion_matrix
        )
        self.assertEqual(metrics_summary.per_class_confusion_matrix["a"].precision, 1)
        self.assertEqual(
            round(metrics_summary.per_class_confusion_matrix["a"].recall, 3), 0.333
        )
        self.assertEqual(
            round(metrics_summary.per_class_confusion_matrix["a"].f1, 3), 0.5
        )
        self.assertEqual(
            round(metrics_summary.per_class_confusion_matrix["b"].precision, 3), 0.333
        )
        self.assertEqual(round(metrics_summary.micro_metrics.precision, 3), 0.5)
        self.assertEqual(round(metrics_summary.micro_metrics.recall, 3), 0.5)
        self.assertEqual(round(metrics_summary.macro_metrics.precision, 3), 0.667)
        self.assertEqual(round(metrics_summary.macro_metrics.recall, 3), 0.667)
        self.assertEqual(round(metrics_summary.macro_metrics.f1, 3), 0.5)
        self.assertEqual(metrics_summary.micro_metrics.true_positive, 2)

    def test_convert_F1MetricsContainer_to_dict(self):
        confusion_matrix = {
            "a": F1Metrics(precision=0.5, recall=0.5, f1=0.5),
            "b": F1Metrics(precision=0.667, recall=0.667, f1=0.667),
        }

        micro_metrics = F1Metrics(
            precision=0.8,
            recall=0.5,
            f1=0.75,
            true_positive=5,
            false_negative=6,
            false_positive=7,
        )
        macro_metrics = F1Metrics(precision=0.5, recall=0.3, f1=0.4)
        metrics_summary = F1MetricsContainer(
            per_class_confusion_matrix=confusion_matrix,
            micro_metrics=micro_metrics,
            macro_metrics=macro_metrics,
        )
        metrics_out = QualityEvaluator.convert_F1MetricsContainer_to_dict(
            metrics_summary
        )
        self.assertEqual(
            metrics_out["per_class_confusion_matrix"]["a"]["precision"], 0.5
        )
        self.assertEqual(
            metrics_out["per_class_confusion_matrix"]["b"]["precision"], 0.667
        )
        self.assertEqual(round(metrics_out["micro_precision"], 3), 0.8)
        self.assertEqual(round(metrics_out["micro_f1"], 3), 0.75)
        self.assertEqual(round(metrics_out["macro_f1"], 3), 0.4)
        self.assertEqual(metrics_out["overall_tp"], 5)
        self.assertEqual(metrics_out["overall_fp"], 7)
        self.assertEqual(metrics_out["overall_fn"], 6)
