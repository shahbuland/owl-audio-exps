def get_model_cls(model_id):
    if model_id == "game_rft_audio":
        """
        GameRFTAudio is Rectified Flow Transformer for video + audio
        """
        from .gamerft_audio import GameRFTAudio
        return GameRFTAudio




