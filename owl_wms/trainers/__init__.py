def get_trainer_cls(trainer_id):
    if trainer_id == "causvid_vid":
        from .causvid_vid_only import CausVidTrainer
        return CausVidTrainer
    if trainer_id == "sforce_vid":
        from .sf_vid_only import SelfForceTrainer
        return SelfForceTrainer
    if trainer_id == "av":
        """
        Most basic trainer. Does audio + video training.
        """
        from .av_trainer import AVRFTTrainer
        return AVRFTTrainer
    if trainer_id == "rft":
        """
        Most basic trainer. Does audio + video training.
        """
        from .rft_trainer import RFTTrainer
        return RFTTrainer
    if trainer_id == "mixed_av":
        """
        Allows for datasets that are a mix of unlabelled (wrt controls) and labelled
        """
        from .mixed_av_trainer import MixedAVRFTTrainer
        return MixedAVRFTTrainer
    if trainer_id == "ode_distill_vid":
        """
        Prune video only trainer
        """
        from .prune_vid_only import DistillODETrainer
        return DistillODETrainer
    if trainer_id == "audio_rft":
        """
        Audio RFT trainer for unconditional audio generation
        """
        from .audio_rft_trainer import AudioRFTTrainer
        return AudioRFTTrainer
