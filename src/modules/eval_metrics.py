import torch
import time
import copy
import torch.multiprocessing as mp
from src.utils.hidden_layer_selection import get_hidden_layer_eval
from src.modules.registry.eval_metrics_registry import eval_metric_registry

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def eval_single_metric_process(args,device):
    Eval_obj, metric, model, x_batch, y_batch, a_batch, custom_batch = args
    return Eval_obj._evaluate_single_metric(
                metric,
                model.to(device),
                x_batch.to(device),
                y_batch.to(device),
                a_batch.to(device),
                custom_batch
            )

class EvalMetricsModule:
    def __init__(self, cfg, model):
        self.modality = cfg.data.modality
        self.eval_cfg = cfg.eval_metric
        # self.layer = get_hidden_layer_eval(model)
        self.eval_metrics = self._initialize_eval_metrics()

    def _initialize_eval_metrics(self):
        """Initialize evaluation metrics based on configuration."""
        eval_metrics = []

        for metric_name in self.eval_cfg.keys():
            metric_func = eval_metric_registry.get_metric(metric_name)
            hparams = self.eval_cfg.get(metric_name, {})
            # metric = metric_func(hparams.copy(), self.modality, layer=self.layer)
            metric = metric_func(hparams.copy(), self.modality)
            eval_metrics.append(metric)

        return eval_metrics

    def evaluate(
        self, 
        model, 
        x_batch, 
        y_batch, 
        a_batch, 
        # xai_methods, 
        # count_xai, 
        custom_batch
    ):
        """Evaluate the metrics using the provided model and data."""
        # from pympler import muppy, summary
        
        eval_scores = []
        for metric in self.eval_metrics:
            # start = time.time()
            eval_scores.append(self._evaluate_single_metric(
                metric,
                model,
                x_batch,
                y_batch,
                a_batch,
                custom_batch,
                # xai_methods,
                # count_xai,
            ))
        return eval_scores
    def _evaluate_single_metric(
        self,
        metric,
        model,
        x_batch,
        y_batch,
        a_batch,
        # xai_methods,
        # count_xai,
        custom_batch,
    ):
        """Evaluate a single metric with the specified parameters."""
        return metric(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            device=device,
            custom_batch=custom_batch,
        )

