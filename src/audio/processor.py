import torch
import torchaudio
import torchaudio.transforms as T
import random

class SpecScaler(torch.nn.Module):
    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        return torch.log(spectrogram.clamp_(1e-9, 1e9))

class AudioProcessor:
    def __init__(self, sample_rate=16000, n_fft=512, hop_length=256, win_length=400, n_mels=64):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.win_length = win_length
        
        # Создаем преобразование для MEL-спектрограммы
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            power=2.0  #? what does it mean?
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
        
        scaler = SpecScaler()
        mel_spec= scaler(mel_spec)
        
        # #? Do we actually need it?
        # # Конвертация в dB
        # mel_spec_db = self.amplitude_to_db(mel_spec)
        
        # return mel_spec_db
        return mel_spec

    def preprocess_audio(self, file_path: str) -> torch.Tensor:
        """Полный пайплайн предобработки аудио"""
        waveform = self.load_audio(file_path)
        mel_spec = self.audio_to_melspectrogram(waveform)
        return mel_spec
    
class AudioAugmentation:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        
    def cyclic_shift(self, waveform, shift_ratio=None):
        """
        Циклический сдвиг аудиосигнала
        shift_ratio: отношение сдвига (0.0-1.0), если None - случайный
        """
        if shift_ratio is None:
            shift_ratio = random.uniform(0.0, 1.0)
            
        num_samples = waveform.shape[1]
        shift_samples = int(num_samples * shift_ratio)
        
        # Циклический сдвиг
        shifted_waveform = torch.roll(waveform, shifts=shift_samples, dims=1)
        
        return shifted_waveform
    
    def time_mask(self, waveform, max_mask_time=0.1, num_masks=1):
        """
        Наложение временных масок
        max_mask_time: максимальная длительность маски в секундах
        num_masks: количество масок
        """
        masked_waveform = waveform.clone()
        num_samples = waveform.shape[1]
        max_mask_samples = int(max_mask_time * self.sample_rate)
        
        for _ in range(num_masks):
            mask_samples = random.randint(1, max_mask_samples)
            mask_start = random.randint(0, num_samples - mask_samples)
            masked_waveform[:, mask_start:mask_start + mask_samples] = 0
            
        return masked_waveform
    
    def frequency_mask(self, waveform, n_fft=400, max_freq_mask=10, num_masks=1):
        """
        Наложение частотных масок (чеспектрограмму)
        n_fft: размер окна FFT
        max_freq_mask: максимальное количество маскируемых частотных бинов
        num_masks: количество масок
        """
        # Конвертируем в спектрограмму
        spectrogram = T.Spectrogram(n_fft=n_fft)(waveform)
        
        for _ in range(num_masks):
            freq_mask_size = random.randint(1, max_freq_mask)
            freq_start = random.randint(0, spectrogram.shape[1] - freq_mask_size)
            spectrogram[:, freq_start:freq_start + freq_mask_size, :] = 0
            
        # Обратно в waveform
        inverse_transform = T.InverseSpectrogram(n_fft=n_fft)
        masked_waveform = inverse_transform(spectrogram)
        
        return masked_waveform
    
    def add_noise(self, waveform, noise_level=1e-6):
        """Добавление гауссового шума"""
        noise = torch.randn_like(waveform) * noise_level
        return waveform + noise
    
    def change_speed(self, waveform, speed_factor=None):
        """
        Изменение скорости воспроизведения
        speed_factor: коэффициент скорости (0.5-2.0), если None - случайный
        """
        if speed_factor is None:
            speed_factor = random.uniform(0.8, 1.2)
            
        effects = [
            ["speed", str(speed_factor)],  # Изменение скорости
            ["rate", str(self.sample_rate)]  # Возврат к исходной sample rate
        ]
        
        transformed_waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
            waveform, self.sample_rate, effects)
        
        return transformed_waveform
    
    def pitch_shift(self, waveform, n_steps=None):
        """
        Сдвиг высоты тона
        n_steps: количество полутонов, если None - случайный
        """
        if n_steps is None:
            n_steps = random.uniform(-2, 2)
            
        pitch_shift = T.PitchShift(
            sample_rate=self.sample_rate,
            n_steps=n_steps
        )
        return pitch_shift(waveform)
    
    def time_stretch(self, waveform, stretch_factor=None):
        """
        Растяжение/сжатие по времени
        stretch_factor: коэффициент растяжения (0.5-2.0), если None - случайный
        """
        if stretch_factor is None:
            stretch_factor = random.uniform(0.8, 1.2)
            
        time_stretch = T.TimeStretch(
            hop_length=512,
            n_freq=201
        )
        
        # Конвертируем в спектрограмму для растяжения
        spectrogram = T.Spectrogram(n_fft=400)(waveform)
        stretched_spec = time_stretch(spectrogram, stretch_factor)
        
        # Обратно в waveform
        inverse_transform = T.InverseSpectrogram(n_fft=400)
        return inverse_transform(stretched_spec)