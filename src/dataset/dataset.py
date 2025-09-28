import torch
from torch.utils.data import Dataset
from src.audio.processor import AudioProcessor
import pandas as pd
from pathlib import Path
from .utils import label2id

class AudioDataset(Dataset):
    def __init__(self, manifest_path, sample_rate=16000, n_fft=512, hop_length=256, n_mels=64, test: bool = False):
        self.manifest_path = Path(manifest_path)
        self.parent_data_path = self.manifest_path.parent
        assert(self.manifest_path.exists())
        assert(self.parent_data_path.exists())
        
        self.test = test
        
        self.audio_processor = AudioProcessor(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        self.file_list = self._load_files()

    def _load_files(self):
        """Загрузка списка файлов и меток"""
        file_list = []
        manifest_df = pd.read_csv(self.manifest_path.absolute())
        
        # shafle values in dataframe before train the model (only for train and validation)
        if not self.test:
            manifest_df = manifest_df.sample(frac=1).reset_index(drop=True)
        
        if not self.test:
            for label, path in zip(manifest_df.label, manifest_df.path):
                file_list.append((self.parent_data_path / Path(path), label2id[label]))
        else:
            for path in manifest_df.path:
                file_list.append(self.parent_data_path / Path(path))
                        
        return file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if not self.test:
            path, label = self.file_list[idx]
        else:
            path = self.file_list[idx]
        
        try:
            # Предобработка аудио с помощью torchaudio
            mel_spec = self.audio_processor.preprocess_audio(str(path.absolute()))
            
            if not self.test:
                # Добавляем размерность канала (1 канал для grayscale спектрограммы)
                # mel_spec shape: [1, n_mels, time_steps]
                return mel_spec, torch.tensor(label, dtype=torch.long)
            else:
                return mel_spec 
        
        except Exception as e:
            print(f"Error processing {path}: {e}")
            # Возвращаем пустой тензор в случае ошибки
            return torch.zeros(1, 64, 100), torch.tensor(-1, dtype=torch.long)