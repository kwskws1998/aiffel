import numpy as np
import robust_loss_pytorch
import torch
from transformers import Trainer

from va_gaze.eval.oof_reports import pearsonr


def _pop_labels(inputs):
    labels = inputs["labels"]
    model_inputs = dict(inputs)
    model_inputs.pop("labels", None)
    return labels, model_inputs


def _add_model_auxiliary_loss(task_loss, outputs):
    auxiliary_loss = outputs.get("loss")
    if auxiliary_loss is None:
        return task_loss
    if auxiliary_loss.ndim > 0:
        auxiliary_loss = auxiliary_loss.mean()
    return task_loss + auxiliary_loss


def _ccc_loss(logits, labels):
    logits_v = logits[:, 0]
    logits_a = logits[:, 1]
    labels_v = labels[:, 0]
    labels_a = labels[:, 1]

    num_v = 2 * pearsonr(logits_v, labels_v) * torch.std(logits_v) * torch.std(labels_v)
    den_v = torch.var(logits_v) + torch.var(labels_v) + torch.square(torch.mean(logits_v) - torch.mean(labels_v))
    ccc_v = num_v / den_v
    loss_v = 1 - ccc_v

    num_a = 2 * pearsonr(logits_a, labels_a) * torch.std(logits_a) * torch.std(labels_a)
    den_a = torch.var(logits_a) + torch.var(labels_a) + torch.square(torch.mean(logits_a) - torch.mean(labels_a))
    ccc_a = num_a / den_a
    loss_a = 1 - ccc_a

    return 0.5 * loss_v + 0.5 * loss_a


def _build_adaptive_loss(num_dims=2):
    adaptive_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return robust_loss_pytorch.adaptive.AdaptiveLossFunction(
        num_dims=num_dims,
        float_dtype=np.float32,
        device=adaptive_device,
    )


def _attach_adaptive_params(optimizer, adaptive):
    adaptive_params = [p for p in adaptive.parameters() if p.requires_grad]
    if not adaptive_params:
        return optimizer

    existing_param_ids = {
        id(param)
        for group in optimizer.param_groups
        for param in group["params"]
    }
    new_params = [param for param in adaptive_params if id(param) not in existing_param_ids]
    if new_params:
        optimizer.add_param_group({"params": new_params, "weight_decay": 0.0})
    return optimizer


def _robust_loss(adaptive, logits, labels):
    if next(adaptive.parameters()).device != logits.device:
        adaptive.to(logits.device)
    residual = labels - logits
    return torch.mean(adaptive.lossfun(residual))


def _split_heteroscedastic_logits(logits):
    if logits.shape[-1] < 4:
        raise ValueError(
            "Heteroscedastic loss requires model logits with at least 4 columns: "
            "valence_mu, arousal_mu, valence_logvar, arousal_logvar."
        )
    return logits[:, :2], logits[:, 2:4]


class VARegressionTrainer(Trainer):
    """Shared trainer base that keeps custom gaze models reloadable at every save."""

    def __init__(self, *args, gaze_learning_rate=None, **kwargs):
        self.gaze_learning_rate = gaze_learning_rate
        self._gaze_learning_rate_applied = False
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        """Place an explicitly exposed gaze residual in its own learning-rate group."""

        optimizer = super().create_optimizer()
        if self._gaze_learning_rate_applied or self.gaze_learning_rate is None:
            return optimizer

        model = self.accelerator.unwrap_model(self.model)
        parameter_getter = getattr(model, "gaze_residual_parameters", None)
        residual_parameters = list(parameter_getter()) if callable(parameter_getter) else []
        residual_parameter_ids = {
            id(parameter) for parameter in residual_parameters if parameter.requires_grad
        }
        if not residual_parameter_ids:
            self._gaze_learning_rate_applied = True
            return optimizer

        moved_parameters = []
        retained_groups = []
        for group in optimizer.param_groups:
            retained_parameters = []
            for parameter in group["params"]:
                if id(parameter) in residual_parameter_ids:
                    moved_parameters.append(parameter)
                else:
                    retained_parameters.append(parameter)
            if retained_parameters:
                group["params"] = retained_parameters
                retained_groups.append(group)
        optimizer.param_groups[:] = retained_groups
        optimizer.add_param_group(
            {
                "params": moved_parameters,
                "lr": float(self.gaze_learning_rate),
                "weight_decay": 0.0,
            }
        )
        self._gaze_learning_rate_applied = True
        return optimizer

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir=output_dir, state_dict=state_dict)
        target_dir = output_dir or self.args.output_dir
        model = self.accelerator.unwrap_model(self.model)
        save_manifest = getattr(model, "save_architecture_manifest", None)
        if callable(save_manifest):
            save_manifest(target_dir)


def _heteroscedastic_loss(logits, labels, mse_weight=0.1, logvar_min=-5.0, logvar_max=3.0):
    mu, logvar = _split_heteroscedastic_logits(logits)
    logvar = torch.clamp(logvar, min=logvar_min, max=logvar_max)
    squared_error = torch.square(labels - mu)
    nll = 0.5 * torch.exp(-logvar) * squared_error + 0.5 * logvar
    mse_anchor = torch.nn.functional.mse_loss(mu, labels)
    return nll.mean() + float(mse_weight) * mse_anchor


class CustomTrainerMSE(VARegressionTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels, model_inputs = _pop_labels(inputs)
        outputs = model(**model_inputs)
        logits = outputs.get("logits")
        loss = torch.nn.functional.mse_loss(logits.view(-1), labels.view(-1))
        loss = _add_model_auxiliary_loss(loss, outputs)
        return (loss, outputs) if return_outputs else loss


class CustomTrainerCCC(VARegressionTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels, model_inputs = _pop_labels(inputs)
        outputs = model(**model_inputs)
        logits = outputs.get("logits")
        loss = _ccc_loss(logits, labels)
        loss = _add_model_auxiliary_loss(loss, outputs)
        return (loss, outputs) if return_outputs else loss


class CustomTrainerMSE_CCC(VARegressionTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels, model_inputs = _pop_labels(inputs)
        outputs = model(**model_inputs)
        logits = outputs.get("logits")
        mse_loss = torch.nn.functional.mse_loss(logits.view(-1), labels.view(-1))
        ccc_loss = _ccc_loss(logits, labels)
        loss = 0.5 * (mse_loss + ccc_loss)
        loss = _add_model_auxiliary_loss(loss, outputs)
        return (loss, outputs) if return_outputs else loss


class CustomTrainerRobust(VARegressionTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adaptive = _build_adaptive_loss(num_dims=2)

    def create_optimizer(self):
        optimizer = super().create_optimizer()
        return _attach_adaptive_params(optimizer, self.adaptive)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels, model_inputs = _pop_labels(inputs)
        outputs = model(**model_inputs)
        logits = outputs.get("logits")
        loss = _robust_loss(self.adaptive, logits, labels)
        loss = _add_model_auxiliary_loss(loss, outputs)
        return (loss, outputs) if return_outputs else loss


class CustomTrainerRobustCCC(VARegressionTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adaptive = _build_adaptive_loss(num_dims=2)

    def create_optimizer(self):
        optimizer = super().create_optimizer()
        return _attach_adaptive_params(optimizer, self.adaptive)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels, model_inputs = _pop_labels(inputs)
        outputs = model(**model_inputs)
        logits = outputs.get("logits")
        robust = _robust_loss(self.adaptive, logits, labels)
        ccc = _ccc_loss(logits, labels)
        loss = 0.5 * (robust + ccc)
        loss = _add_model_auxiliary_loss(loss, outputs)
        return (loss, outputs) if return_outputs else loss


class CustomTrainerHeteroscedastic(VARegressionTrainer):
    def __init__(
        self,
        *args,
        hetero_mse_weight=0.1,
        hetero_logvar_min=-5.0,
        hetero_logvar_max=3.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.hetero_mse_weight = hetero_mse_weight
        self.hetero_logvar_min = hetero_logvar_min
        self.hetero_logvar_max = hetero_logvar_max

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels, model_inputs = _pop_labels(inputs)
        outputs = model(**model_inputs)
        logits = outputs.get("logits")
        loss = _heteroscedastic_loss(
            logits,
            labels,
            mse_weight=self.hetero_mse_weight,
            logvar_min=self.hetero_logvar_min,
            logvar_max=self.hetero_logvar_max,
        )
        loss = _add_model_auxiliary_loss(loss, outputs)
        return (loss, outputs) if return_outputs else loss
