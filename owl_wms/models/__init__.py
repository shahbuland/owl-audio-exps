def get_model_cls(model_id):
    if model_id == "game_rft":
        """
        GameRFT is Rectified Flow Transformer for video only
        """
        from .gamerft import GameRFT
        return GameRFT
    if model_id == "game_rft_audio":
        """
        GameRFTAudio is Rectified Flow Transformer for video + audio
        """
        from .gamerft_audio import GameRFTAudio
        return GameRFTAudio
    if model_id == "game_mft_audio":
        """
        GameMFTAudio is Mean Flow Transformer for video + audio
        """
        from .gamemft_audio import GameMFTAudio
        return GameMFTAudio
    if model_id == "audio_rft":
        """
        AudioRFT is Rectified Flow Transformer for audio only
        """
        from .audiorft import AudioRFT
        return AudioRFT
