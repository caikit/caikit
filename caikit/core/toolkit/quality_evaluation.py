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

"""Evaluate quality of models.
"""

# Standard
from dataclasses import dataclass
from typing import Dict, Optional
import enum
import math
import re

# First Party
import alog

# Local
from .errors import error_handler

log = alog.use_channel("TLKIT")
error = error_handler.get(log)


class EvalTypes(enum.Enum):
    """Enum that contains set of all possible evaluation types."""

    SINGLELABEL_MULTICLASS = 1
    MULTILABEL_MULTICLASS = 2
    MULTILABEL_MULTICLASS_HIERARCHICAL = 3


@dataclass
class F1Metrics:
    true_positive: Optional[int] = None
    false_positive: Optional[int] = None
    false_negative: Optional[int] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None


@dataclass
class F1MetricsContainer:
    per_class_confusion_matrix: Dict[str, F1Metrics]
    macro_metrics: F1Metrics
    micro_metrics: F1Metrics


class QualityEvaluator:
    """Class that holds all evaluation logic for now. May eventually be broken up into
    subclasses."""

    def __init__(self, gold, pred):
        self.gold = gold
        self.pred = pred

    def run(
        self,
        evaluation_type,
        find_label_func=None,
        find_label_data_func=None,
        detailed_metrics=False,
        labels=None,
        partial_match_metrics=False,
        max_hierarchy_levels=3,
    ):
        """Main entry point for evaluation.

        Args:
            evaluation_type (str): Which type of evaluation to run. Only a few
                are currently supported.
            find_label_func: function to fetch labels from any one prediction, used in
                multiclass multilabel evaluation.
                eg: if a prediction is of form (token, label), this function should be
                able to tell us how to extract the class labels from the prediction, in
                this case return the second element of the tuple.
            find_label_data_func: function to fetch predictions that belongs to a certain label,
                used only in multiclass multilabel eval type, e.g., if predictions for a data
                example looks like [(tok1, labX), (tok2, labY), (tok3, labX)], then
                the function should be able to return all predictions with a given label - labX
                return should look like [(tok1, labX), (tok3, labX)]
            detailed_metrics: flag to indicate whether or not you want detailed metrics
                              (currently only for multiclass multilabel eval type)
                              Detailed metrics give us metrics for every example, and
                              metrics using a custom partial match function
            labels: list (Optional, defaults to None)
                Optional list of class labels to evaluate quality on. By default evaluation is done
                over all class labels. Using this, you can explicitly mention only a subset of
                labels to include in the quality evaluation.
            partial_match_metrics: flag to indicate whether or not you want partial match
                                   micro avg metrics.
                                   (currently only for multiclass multilabel eval type)
            max_hierarchy_levels (int): Used in hierarchical multilabel
                multiclass evaluation only. The number of levels in the
                hierarchy to run model evaluation on, in addition to complete
                matches.

        Returns:
            dict: Full results from evaluation on dataset and model.
        """
        if evaluation_type == EvalTypes.MULTILABEL_MULTICLASS:
            return self.multilabel_multiclass_evaluation(
                find_label_func,
                find_label_data_func,
                labels,
                detailed_metrics,
                partial_match_metrics,
            )
        if evaluation_type == EvalTypes.SINGLELABEL_MULTICLASS:
            return self.singlelabel_multiclass_evaluation(labels)
        if evaluation_type == EvalTypes.MULTILABEL_MULTICLASS_HIERARCHICAL:
            return self.multilabel_multiclass_hierarchical_evaluation(
                find_label_func, find_label_data_func, max_hierarchy_levels
            )

        error(
            "<COR81451123E>",
            ValueError("Unknown evaluation_type: {0}".format(evaluation_type)),
        )

    def singlelabel_multiclass_evaluation(self, labels=None) -> dict:
        """Obtain results of evaluation for a single-label, multi-class model.

        Args:
            Note: here class should be initialized with gold and pred in the following format
            self.gold (list): list of gold set labels for every example, where each example
                can have only one label eg: ['label1','label2', 'label3','label4']
            self.pred (list): Predicted-by-the-model set labels for every example.
            labels: list (Optional, defaults to None)
                Optional list of class labels to evaluate quality on. By default evaluation is done
                over all class labels. Using this, you can explicitly mention only a subset of
                labels to include in the quality evaluation.

        Returns:
            dict: Dictionary looks like: { 'per_class_confusion_matrix':
                {'entity_type': {'true_positive': int ...}} 'macro_precision': 0
                <= float <= 1, 'macro_recall': 0 <= float <= 1, 'macro_f1': 0 <=
                float <= 1, 'micro_precision': 0 <= float <= 1,, 'micro_recall':
                0 <= float <= 1,, 'micro_f1': 0 <= float <= 1, 'overall_tp':
                int, 'overall_fp': int, 'overall_fn': int


                }
        """

        gold, pred = self.gold, self.pred
        assert len(gold) == len(
            pred
        ), "Length of gold and predicted datasets does not match"

        per_class_confusion_matrix = {}

        for gold_label, pred_label in zip(gold, pred):
            if gold_label not in per_class_confusion_matrix and (
                labels is None or gold_label in labels
            ):
                per_class_confusion_matrix[gold_label] = F1Metrics(
                    true_positive=0,
                    false_positive=0,
                    false_negative=0,
                    precision=0.0,
                    recall=0.0,
                    f1=0.0,
                )
            if pred_label not in per_class_confusion_matrix and (
                labels is None or pred_label in labels
            ):
                per_class_confusion_matrix[pred_label] = F1Metrics(
                    true_positive=0,
                    false_positive=0,
                    false_negative=0,
                    precision=0.0,
                    recall=0.0,
                    f1=0.0,
                )
            # true positive
            if gold_label == pred_label and (labels is None or gold_label in labels):
                per_class_confusion_matrix[gold_label].true_positive += 1
            else:
                if labels is None or pred_label in labels:
                    per_class_confusion_matrix[pred_label].false_positive += 1
                if labels is None or gold_label in labels:
                    per_class_confusion_matrix[gold_label].false_negative += 1

        calc_metrics = QualityEvaluator.calc_metrics_from_confusion_matrix(
            per_class_confusion_matrix
        )
        metrics_out = QualityEvaluator.convert_F1MetricsContainer_to_dict(calc_metrics)
        return metrics_out

    def multilabel_multiclass_evaluation(
        self,
        find_label_func,
        find_label_data_func,
        labels=None,
        detailed_metrics=False,
        partial_match_metrics=False,
        use_labels_for_matching=False,
    ) -> dict:
        """Obtain results of evaluation for a multi-label, multi-class model.

        Args:
            Note: here class should be initialized with gold and pred in the following format
            self.gold (list(list)): list of gold set labels for every example eg:
                [['label1','label2'], ['label1', 'label4']]
            self.pred (list(list)): Predicted-by-the-model set labels for every example.
            find_label_func: function to fetch labels from any one prediction
            find_label_data_func: function to fetch data that belongs to a certain class
            labels: list (Optional, defaults to None)
                Optional list of class labels to evaluate quality on. By default evaluation is done
                over all class labels. Using this, you can explicitly mention only a subset of
                labels to include in the quality evaluation.
            detailed_metrics: flag to indicate whether or not you want detailed metrics
                              Detailed metrics give us metrics for every example, and
                              metrics using a custom partial match function
            partial_match_metrics: flag to indicate whether or not you want partial match
                                   micro avg metrics.
            use_labels_for_matching (bool): Indicates whether or not we should
                use the output of find_label_func for metric computations, or
                the raw data tuples.

        Returns:
            dict: Dictionary looks like: { 'per_class_confusion_matrix':
                {'entity_type': {'true_positive': int ...}} 'macro_precision': 0
                <= float <= 1, 'macro_recall': 0 <= float <= 1, 'macro_f1': 0 <=
                float <= 1, 'micro_precision': micro_precision, 'micro_recall':
                micro_recall, 'micro_f1': micro_f1, 'detailed_metrics' :
                {'exact_match_precision'..,'partial_match_precision'}
                'micro_precision_partial_match': 0 <= float <= 1,
                'micro_recall_partial_match': 0 <= float <= 1,
                'micro_f1_partial_match': 0 <= float <= 1 }
        """
        gold, pred = self.gold, self.pred
        assert len(gold) == len(
            pred
        ), "Length of gold and predicted datasets does not match"
        detailed_output = []
        per_class_confusion_matrix = {}
        all_labels = set()
        num_preds_partial_matched = 0
        num_gold_partial_matched = 0
        total_pred = 0
        total_gold = 0
        overall_tp = 0
        overall_fp = 0
        overall_fn = 0
        micro_precision = 0.0
        micro_recall = 0.0
        micro_f1 = 0.0

        micro_metrics_partial = F1Metrics(precision=0.0, recall=0.0, f1=0.0)
        if labels:
            try:
                gold = [
                    [label for label in gold_ex if find_label_func(label) in labels]
                    for gold_ex in gold
                ]
                pred = [
                    [label for label in pred_ex if find_label_func(label) in labels]
                    for pred_ex in pred
                ]
            except NotImplementedError:
                error(
                    "<COR19114599E>",
                    NotImplementedError(
                        "find_label_func must be implemented to use [labels]"
                    ),
                )

        for gold_ex, pred_ex in zip(gold, pred):
            if detailed_metrics:
                precision, recall, f1 = QualityEvaluator.calc_f1_score(gold_ex, pred_ex)
                (
                    partial_precision,
                    partial_recall,
                    partial_f1,
                ) = QualityEvaluator.calc_f1_score(
                    gold_ex, pred_ex, QualityEvaluator.find_partial_matches
                )
                instances = {
                    "exact_match_precision": precision,
                    "exact_match_recall": recall,
                    "exact_match_f1": f1,
                    "partial_match_precision": partial_precision,
                    "partial_match_recall": partial_recall,
                    "partial_match_f1": partial_f1,
                }
                detailed_output.append(instances)
            try:
                gold_labels = set(map(find_label_func, gold_ex))
                pred_labels = set(map(find_label_func, pred_ex))
                # get per-class information if possible
                all_labels = gold_labels.union(pred_labels)

            except NotImplementedError:
                # If find_label_func raises NotImplementedError, we can't do label-based matching.
                # In this case we need to fall back to set operations on the raw data tuples.
                log.info(
                    "INFO: find_label_func not implemented for this module type - falling back "
                    "to tuple match!!"
                )
                use_labels_for_matching = False

            for label in all_labels:
                # dictionary initizalization
                if label not in per_class_confusion_matrix:
                    per_class_confusion_matrix[label] = F1Metrics(
                        true_positive=0,
                        false_positive=0,
                        false_negative=0,
                        precision=0.0,
                        recall=0.0,
                        f1=0.0,
                    )
                # build confusion matrix
                pred_label_data = set(find_label_data_func(pred_ex, label))
                gold_label_data = set(find_label_data_func(gold_ex, label))

                # true positive
                per_class_confusion_matrix[label].true_positive += len(
                    gold_label_data.intersection(pred_label_data)
                )

                # false positive
                per_class_confusion_matrix[label].false_positive += len(
                    pred_label_data - gold_label_data
                )

                # false negative
                per_class_confusion_matrix[label].false_negative += len(
                    gold_label_data - pred_label_data
                )
            if use_labels_for_matching:
                gold_ex_set = gold_labels
                pred_ex_set = pred_labels
            else:
                # In case the user did not specify how to obtain class labels,
                # we can still calculate micro avg using sum of true positives,
                # false positives etc over all examples (over all classes)
                # We should deprecate this section in next major release
                gold_ex_set = set(gold_ex)
                pred_ex_set = set(pred_ex)
                overall_tp += len(gold_ex_set.intersection(pred_ex_set))
                overall_fp += len(pred_ex_set - gold_ex_set)
                overall_fn += len(gold_ex_set - pred_ex_set)
                # Calculate micro average metrics
                # Micro precision =
                # no. of correct precisions over all classes / no. of total predictions
                if not math.isclose(overall_tp + overall_fp, 0):
                    micro_precision = overall_tp / (overall_tp + overall_fp)
                # Micro recall = no. of correct precisions over all classes / no. of true samples
                if not math.isclose(overall_tp + overall_fn, 0):
                    micro_recall = overall_tp / (overall_tp + overall_fn)
                # Micro avg F1 = harmonic mean of precision and recall
                if not math.isclose(micro_precision + micro_recall, 0):
                    micro_f1 = (2.0 * micro_precision * micro_recall) / (
                        micro_precision + micro_recall
                    )

            if partial_match_metrics:
                gold_matched, preds_matched = QualityEvaluator.find_partial_matches(
                    gold_ex_set, pred_ex_set
                )
                num_preds_partial_matched += len(preds_matched)
                num_gold_partial_matched += len(gold_matched)
                total_pred += len(pred_ex_set)
                total_gold += len(gold_ex_set)

        calc_metrics = QualityEvaluator.calc_metrics_from_confusion_matrix(
            per_class_confusion_matrix
        )

        # This section should be deprecated with future refactors
        if not use_labels_for_matching:
            log.warning(
                "WARNING: Only Micro_avg metrics could be calculated based on the information "
                "available for this module type."
            )
            calc_metrics.micro_metrics.precision = micro_precision
            calc_metrics.micro_metrics.recall = micro_recall
            calc_metrics.micro_metrics.f1 = micro_f1
            calc_metrics.micro_metrics.true_positive = overall_tp
            calc_metrics.micro_metrics.false_positive = overall_fp
            calc_metrics.micro_metrics.false_negative = overall_fn

        metrics_out = QualityEvaluator.convert_F1MetricsContainer_to_dict(calc_metrics)
        # This flag only controls calculation of micro average partial match metrics
        # Detailed metrics flag calculates partial match metrics per data row
        if partial_match_metrics:
            # Calculate micro average partial match metrics
            # Micro precision = Number of matched predictions / Number predicted
            # Micro precision = Fraction of retrieved instances that are relevant
            if total_pred > 0:
                micro_metrics_partial.precision = num_preds_partial_matched / total_pred
            # Micro recall = Number of matched gold / Number in gold set
            # Micro recall = Fraction of relevant instances that are retrieved
            if total_gold > 0:
                micro_metrics_partial.recall = num_gold_partial_matched / total_gold
            # Micro avg F1 = harmonic mean of precision and recall
            if not math.isclose(
                micro_metrics_partial.precision + micro_metrics_partial.recall, 0
            ):
                micro_metrics_partial.f1 = (
                    2.0 * micro_metrics_partial.precision * micro_metrics_partial.recall
                ) / (micro_metrics_partial.precision + micro_metrics_partial.recall)

        metrics_out["detailed_metrics"] = detailed_output
        metrics_out["micro_precision_partial_match"] = micro_metrics_partial.precision
        metrics_out["micro_recall_partial_match"] = micro_metrics_partial.recall
        metrics_out["micro_f1_partial_match"] = micro_metrics_partial.f1

        return metrics_out

    def multilabel_multiclass_hierarchical_evaluation(
        self,
        find_label_func_builder,
        find_label_data_func_builder,
        max_hierarchy_levels=3,
    ) -> dict:
        """Evaluate multilabel/multiclass over a hierarchy, e.g., for ESA categories. This method
        Evaluates over a set number of hierarchy levels.

        Because each level in the hierarchy needs to be able to compare and extract differently,
        we use builder funcs that create the appropriate functions for a given level of the
        hierarchy.

        Args:
            find_label_func_builder (function): A function that takes in a level
                number (or None if full hierarchy) and returns a find_label_func
                for this level that can be passed to the multilabel multiclass
                evaluator.
            find_label_data_func_builder (function): A function that takes in a
                level number (or None if full hierarchy) and returns a
                find_label_data_func for this level that can be passed to the
                multilabel multiclass evaluator.
            max_hierarchy_levels (int): The number of levels to run in the
                hierarchy, in addition to complete match.
        Returns:
            dict: Dictionary, where each key is a level number, or 'FULL', and
                maps to the dict returned by multilabel_multiclass_evaluation
                for that level of the hierarchy.
        """
        metrics = {}
        # Levels are None [FULL], and 1...n, where n is the deepest level in the hierarchy (for now,
        # needs to be manually determined by the user).
        # pylint: disable=unnecessary-comprehension
        levels = [None] + [level for level in range(1, max_hierarchy_levels + 1)]
        for level in levels:
            # Get the find_label_func/find_label_data_func for this level in the hierarchy
            find_label_func = find_label_func_builder(level)
            find_label_data_func = find_label_data_func_builder(level)
            # Get the appropriate dictionary key - Fall is a bit more descriptive than None
            dict_key = "level_{}".format(level) if level is not None else "level_all"
            # NOTE: We use label matching for computing our metrics here. This means that we
            # compare the outputs of find_label_func on gold/pred examples to get our metrics
            # instead of the gold/pred example tuples themselves. The reason that we generally
            # want to do this is that the examples have the full labels, but we need to slice out
            # just part of the hierarchical label to consider each level.
            metrics[dict_key] = self.multilabel_multiclass_evaluation(
                find_label_func, find_label_data_func, use_labels_for_matching=True
            )
        return metrics

    @staticmethod
    def calc_f1_score(gold, pred, match_fun=None):
        """Calculates F1 score
        Args:
            gold (list): List of gold annotations
            pred (list): List of predictions
            match_fun: Function that finds the matches and returns tuple of matched gold, preds
        Returns:
            tuple: Precision, Recall, F1 score
        """
        if match_fun:
            # In case of partial match, matched predictions need not equal matched gold
            # Two predictions can match one gold, while another gold may not be matched
            try:
                matched_gold, matched_preds = match_fun(gold, pred)
            except (ValueError, TypeError):
                error(
                    "<COR19474599E>",
                    ValueError("Match function not returning expected tuple format"),
                )
        else:
            matched_preds = set(gold).intersection(set(pred))
            matched_gold = matched_preds

        num_correct_preds, num_pred, num_gold, num_correct_gold = (
            len(matched_preds),
            len(pred),
            len(gold),
            len(matched_gold),
        )

        # precision == Fraction of relevant instances among retrieved instances
        # If we could match 3 predictions with gold out of 4 predictions, precision = 3/4
        precision = num_correct_preds / num_pred if num_pred > 0 else 0.0
        # recall == Fraction of retrieved instances among relevant instances
        # If we could match/retrieve only one gold out of 3, recall = 1/3
        recall = num_correct_gold / num_gold if num_gold > 0 else 0.0
        # f1 == harmonic_mean(precision, recall)
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if precision != 0 and recall != 0
            else 0.0
        )
        return precision, recall, f1

    @staticmethod
    def find_partial_matches(groundtruth, prediction):
        """Function to do find partial match between predicted phrases and the ground truth.
           partial match means a complete predicted phrase is a part of any ground truth phrase or
           a complete ground truth phrase is a part of any predicted phrase.
           Overlaps are not considered.

        Args:
            groundtruth (list): Groundtruth data
            prediction (list): Predictions returned by the model

        Returns:
            tuple: gold_matched: set, pred_matched: set gold annotations that
                were matched Predictions that partially or fully matched with
                groundtruth
        """

        gold_matched = set()
        pred_matched = set()

        for ground_truth_phrase in groundtruth:
            for predicted_phrase in prediction:
                pd_compiler = re.compile(r"\b%s\b" % re.escape(predicted_phrase), re.I)
                # Checks if prediction is part of groundtruth
                if pd_compiler.search(ground_truth_phrase):
                    pred_matched.add(predicted_phrase)
                    gold_matched.add(ground_truth_phrase)
                else:
                    # Checks if groundtruth is part of prediction
                    gt_compiler = re.compile(
                        r"\b%s\b" % re.escape(ground_truth_phrase), re.I
                    )
                    if gt_compiler.search(predicted_phrase):
                        pred_matched.add(predicted_phrase)
                        gold_matched.add(ground_truth_phrase)

        return gold_matched, pred_matched

    @staticmethod
    def calc_metrics_from_confusion_matrix(
        per_class_confusion_matrix: Dict[str, F1Metrics]
    ) -> F1MetricsContainer:
        """Function to calculate precision, recall, F1 metrics using a confusion matrix containing
           statistics per class label.

        Args:
            per_class_confusion_matrix (Dict[str, F1Metrics]): Dictionary of
                 statistics per class label. Class labels are keys for the
                 dictionary. For each class label, there should be a F1Metrics
                 class object with values true positive, false_positive ,
                 false_negative representating the count of these per class. The
                 dictionary looks like: per_class_confusion_matrix[label] =
                 F1Metrics(true_positive = val 1, false_positive = val 2,
                 false_negative = val 3)

        Returns:
            Returns:
            metrics_summary: F1MetricsContainer
            An instance of F1MetricsContainer dataclass containing summary of F1 metrics
        """
        macro_metrics = F1Metrics(precision=0.0, recall=0.0, f1=0.0)
        micro_metrics = F1Metrics(
            true_positive=0,
            false_positive=0,
            false_negative=0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
        )
        num_classes = len(per_class_confusion_matrix)
        # Compute metrics per label
        for label in per_class_confusion_matrix:
            tp = per_class_confusion_matrix[label].true_positive
            fp = per_class_confusion_matrix[label].false_positive
            fn = per_class_confusion_matrix[label].false_negative
            # Calculate precision of a label X = \
            # no. of correct predictions of X / no. of predictions of X
            if not math.isclose(tp + fp, 0):
                per_class_confusion_matrix[label].precision = tp / (tp + fp)
            # Calculate recall of label X = \
            # no. of correct predictions of X / no. of true samples of X
            if not math.isclose(tp + fn, 0):
                per_class_confusion_matrix[label].recall = tp / (tp + fn)
            prec = per_class_confusion_matrix[label].precision
            recall = per_class_confusion_matrix[label].recall
            # Calculate F1 score of label X = harmonic mean of precision and recall of X
            if not math.isclose(prec + recall, 0):
                per_class_confusion_matrix[label].f1 = (2.0 * prec * recall) / (
                    prec + recall
                )

            micro_metrics.true_positive += tp
            micro_metrics.false_positive += fp
            micro_metrics.false_negative += fn

            macro_metrics.precision += per_class_confusion_matrix[label].precision
            macro_metrics.recall += per_class_confusion_matrix[label].recall
            macro_metrics.f1 += per_class_confusion_matrix[label].f1

        # Macro average metrics = average of metrics over all classes
        if num_classes > 0:
            macro_metrics.precision = macro_metrics.precision / num_classes
            macro_metrics.recall = macro_metrics.recall / num_classes
            macro_metrics.f1 = macro_metrics.f1 / num_classes

        # Calculate micro average metrics
        # Micro precision = no. of correct precisions over all classes / no. of total predictions
        if not math.isclose(
            micro_metrics.true_positive + micro_metrics.false_positive, 0
        ):
            micro_metrics.precision = micro_metrics.true_positive / (
                micro_metrics.true_positive + micro_metrics.false_positive
            )
        # Micro recall = no. of correct precisions over all classes / no. of true samples
        if not math.isclose(
            micro_metrics.true_positive + micro_metrics.false_negative, 0
        ):
            micro_metrics.recall = micro_metrics.true_positive / (
                micro_metrics.true_positive + micro_metrics.false_negative
            )
        # Micro avg F1 = harmonic mean of precision and recall
        if not math.isclose(micro_metrics.precision + micro_metrics.recall, 0):
            micro_metrics.f1 = (
                2.0 * micro_metrics.precision * micro_metrics.recall
            ) / (micro_metrics.precision + micro_metrics.recall)

        metrics_summary = F1MetricsContainer(
            per_class_confusion_matrix, macro_metrics, micro_metrics
        )
        return metrics_summary

    # pylint: disable=no-self-argument
    def convert_F1MetricsContainer_to_dict(metrics_summary: F1MetricsContainer) -> dict:
        """
        Args:
            metrics_summary (F1MetricsContainer): An object of dataclass
                 F1MetricsContainer

        Returns:
            Returns:
            dict
                Dictionary looks like: {
                    'per_class_confusion_matrix': {'entity_type': {'true_positive': int ...}}
                    'macro_precision': 0 <= float <= 1,
                    'macro_recall': 0 <= float <= 1,
                    'macro_f1': 0 <= float <= 1,
                    'micro_precision': 0 <= float <= 1,,
                    'micro_recall': 0 <= float <= 1,,
                    'micro_f1': 0 <= float <= 1,
                    'overall_tp': int,
                    'overall_fp': int,
                    'overall_fn': int
                }
        """

        for label, obj in metrics_summary.per_class_confusion_matrix.items():
            # Converts the object to dictionary
            metrics_summary.per_class_confusion_matrix[label] = vars(obj)

        out = {"per_class_confusion_matrix": metrics_summary.per_class_confusion_matrix}

        for k, v in vars(metrics_summary.macro_metrics).items():
            out[f"macro_{k}"] = v

        out["micro_precision"] = metrics_summary.micro_metrics.precision
        out["micro_recall"] = metrics_summary.micro_metrics.recall
        out["micro_f1"] = metrics_summary.micro_metrics.f1
        out["overall_tp"] = metrics_summary.micro_metrics.true_positive
        out["overall_fp"] = metrics_summary.micro_metrics.false_positive
        out["overall_fn"] = metrics_summary.micro_metrics.false_negative

        return out
