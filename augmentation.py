from src.audio.processor import AudioAugmentation

import torchaudio
import random
import pandas as pd
import argparse
from pathlib import Path
import warnings
from tqdm import tqdm

# turn off the UserWarnings because lots of them are talking about
# library function refactoring, last or future deprecations
warnings.filterwarnings('ignore', category=UserWarning)

def augment_dataset(data_path: Path, audio_files: list, labels: list, num_augmentations=5):
    """
    Функция для аугментации всего датасета
    """
    augmenter = AudioAugmentation()
    
    augmented_files = []
    augmented_labels = []

    # Если нет директории для аугментированных файлов, создаем ее
    if not (data_path / 'augmented').exists():
        (data_path / 'augmented').mkdir()
        output_dir = data_path / 'augmented'
    else:
        output_dir = data_path / 'augmented'
    
    for i, (audio_file, label) in tqdm(enumerate(zip(audio_files, labels)), desc="Data augmentation"):
        audio_file_path = data_path / Path(audio_file)
        
        # Загрузка аудиофайла
        waveform, sample_rate = torchaudio.load(audio_file_path.absolute())
        
        # Создание аугментированных версий
        for aug_num in range(num_augmentations):
            aug_waveform = waveform.clone()
            
            # Случайный выбор методов аугментации
            augmentations = random.sample([
                # 'cyclic_shift', #! it can break the system
                'time_mask', 
                # 'frequency_mask', #! bad option
                'add_noise',
                # 'change_speed', #! bad option
                # 'pitch_shift', #! bad option
                'time_stretch'
            ], random.randint(1, 2))  # Применяем 1-2 случайную аугментацию
            
            for aug_method in augmentations:
                if aug_method == 'cyclic_shift':
                    aug_waveform = augmenter.cyclic_shift(aug_waveform)
                elif aug_method == 'time_mask':
                    aug_waveform = augmenter.time_mask(aug_waveform)
                elif aug_method == 'frequency_mask':
                    aug_waveform = augmenter.frequency_mask(aug_waveform)
                elif aug_method == 'add_noise':
                    aug_waveform = augmenter.add_noise(aug_waveform)
                elif aug_method == 'change_speed':
                    aug_waveform = augmenter.change_speed(aug_waveform)
                elif aug_method == 'pitch_shift':
                    aug_waveform = augmenter.pitch_shift(aug_waveform)
                elif aug_method == 'time_stretch':
                    aug_waveform = augmenter.time_stretch(aug_waveform)
            
            # Сохранение аугментированного файла
            aug_path = output_dir / f"aug_{i}_{aug_num}.wav"
            torchaudio.save(aug_path, aug_waveform, sample_rate)
            
            augmented_files.append(aug_path.relative_to(data_path))
            augmented_labels.append(label)
            
    return augmented_files, augmented_labels

# Пример использования
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Аугментация датасета")
    parser.add_argument('-m', '--manifest_path', type=str, default='./data/train/manifest_train.csv')
    args = parser.parse_args()

    manifest_path = Path(args.manifest_path)
    assert(manifest_path.exists())

    manifest_df = pd.read_csv(manifest_path.absolute())

    audio_files = manifest_df.path.to_list()
    labels = manifest_df.label.to_list()

    # Аугментация датасета
    augmented_files, augmented_labels = augment_dataset(
        manifest_path.parent, audio_files, labels, num_augmentations=1
    )
    
    print(f"Создано {len(augmented_labels)} аугментированных файлов")

    df = pd.DataFrame({
        'label': augmented_labels,
        'path': augmented_files
    })

    new_manifest = pd.concat((df, manifest_df), axis=0)
    new_manifest_path = manifest_path.parent / "manifest_train_augmented.csv"
    new_manifest.to_csv(new_manifest_path, index=False)
