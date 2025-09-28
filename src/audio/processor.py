import torch
import torchaudio
import torchaudio.transforms as T

class AudioProcessor:
    def __init__(self, sample_rate=16000, n_fft=512, hop_length=256, n_mels=64):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        # Создаем преобразование для MEL-спектрограммы
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0  # используем power спектрограмму
        )
        
        #? Do we actually need it?
        # Преобразование для amplitude to dB
        self.amplitude_to_db = T.AmplitudeToDB()

    def load_audio(self, file_path: str) -> torch.Tensor:
        """Загрузка аудиофайла с помощью torchaudio"""
        waveform, sr = torchaudio.load(file_path)
        
        # Ресемплинг если необходимо
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Конвертируем в моно если нужно
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        return waveform

    def audio_to_melspectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Преобразование аудио в MEL-спектрограмму"""
        # Нормализация аудио
        waveform = waveform / torch.max(torch.abs(waveform))
        
        # Получение MEL-спектрограммы
        mel_spec = self.mel_transform(waveform)
        
        #? Do we actually need it?
        # Конвертация в dB
        mel_spec_db = self.amplitude_to_db(mel_spec)
        
        # return mel_spec_db
        return mel_spec

    def preprocess_audio(self, file_path: str) -> torch.Tensor:
        """Полный пайплайн предобработки аудио"""
        waveform = self.load_audio(file_path)
        mel_spec = self.audio_to_melspectrogram(waveform)
        return mel_spec