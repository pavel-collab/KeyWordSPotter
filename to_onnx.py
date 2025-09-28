import torch
import hydra
from omegaconf import DictConfig
from pathlib import Path
import onnx
import os

from src.models.module import KeywordSpotter
from src.dataset.module import AudioDataModule

@hydra.main(version_base=None, config_path="./configs", config_name="kws.yaml")
def main(cfg: DictConfig):
    data_module = AudioDataModule(cfg.data)
    data_module.setup()
    dataloader = data_module.train_dataloader()
    
    for batch in dataloader:
        # we're getting batch from train dataloader, that returns a batch of data and labels
        # the first element of batch (batch[0]) is a batch of data (3 dimention for audio)
        # the second one is a batch of labels (batch[1]) -- one dimention tensor
        # So, here to obtain a shape of input data, we have to take the batch of data batch[0]
        # and next take the first element in this batch -- batch[0][0]
        input_dim = batch[0][0].shape
        break
    output_dim = cfg.model.num_classes

    # Загрузка модели из чекпоинта
    checkpoint_path = Path(cfg.onnx.checkpoint_path)
    assert(checkpoint_path.exists())

    model = KeywordSpotter.load_from_checkpoint(
        checkpoint_path, 
        input_dim=input_dim, 
        output_dim=output_dim
    )

    # Установка модели в режим оценки
    model.eval()

    # Создание примера входных данных
    # Actually, model is waiting for a 4d data input:
    # 3d audio tensor data and the batch of such 3d examples,
    # the result is a 4d data
    example_input = torch.randn(input_dim)[None, :, :, :]

    # Экспорт в ONNX
    onnx_path = Path(cfg.onnx.save_path)
    if not onnx_path.exists():
        os.mkdir(onnx_path)
    onnx_path = onnx_path / cfg.onnx.onnx_file_name    
    
    with torch.no_grad():
        torch.onnx.export(
            model,                     # Модель для экспорта
            example_input,            # Пример входных данных
            onnx_path,                # Путь для сохранения ONNX файла
            input_names=["input"],    # Имя входного тензора
            output_names=["output"],  # Имя выходного тензора
            dynamic_axes={            # Настройка динамических осей
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            },
            opset_version=11,         # Версия ONNX opset:cite[1]:cite[9]
            export_params=True,       # Сохранять веса параметров в файле:cite[2]
            do_constant_folding=True  # Выполнять свертку констант для оптимизации:cite[2]
        )

    print(f"Модель успешно экспортирована в {onnx_path}")

    if cfg.onnx.verify_onnx:
        # (Опционально) Проверка корректности ONNX модели
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX модель прошла проверку корректности")
    
if __name__ == '__main__':
    main()