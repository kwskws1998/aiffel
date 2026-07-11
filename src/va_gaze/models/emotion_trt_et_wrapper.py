"""TRT-only emotion gaze predictor adapter."""

from va_gaze.models.emotion_et_wrapper import EmotionEtFixationsPredictor


DEFAULT_REPO_ID = "skboy/emotion_trt_roberta_lr2e5_preval10"
DEFAULT_WEIGHT_NAME = "emotion_trt_predictor_lr2e5_preval10_seed42.safetensors"


class EmotionTrtFixationsPredictor(EmotionEtFixationsPredictor):
    """Load the emotion-specific RoBERTa predictor that exposes TRT only."""

    def __init__(
        self,
        modelTokenizer,
        model_id=None,
        weight_name=None,
        max_length=512,
        device=None,
    ):
        super().__init__(
            modelTokenizer=modelTokenizer,
            model_id=model_id or DEFAULT_REPO_ID,
            weight_name=weight_name or DEFAULT_WEIGHT_NAME,
            max_length=max_length,
            device=device,
            local_files_only_env="EMOTION_TRT_ET_LOCAL_FILES_ONLY",
        )
