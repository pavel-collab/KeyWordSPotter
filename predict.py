import torch
import onnxruntime as ort
import numpy as np
import hydra
from omegaconf import DictConfig
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd
from pathlib import Path

from src.dataset.module import AudioDataModule
from src.dataset.utils import label2id

def get_predictions_from_logits(logits, method='numpy'):
    if method == 'pytorch':
        logits_tensor = torch.tensor(logits)
        probabilities = F.softmax(logits_tensor, dim=1)
        predicted_labels = torch.argmax(probabilities, dim=1)
        return predicted_labels.numpy(), probabilities.numpy()
    
    elif method == 'numpy':
        # Стабильная реализация softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        predicted_labels = np.argmax(probabilities, axis=1)
        return predicted_labels, probabilities

def to_numpy(tensor):
    """Конвертирует тензор в numpy array"""
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def onnx_inference(test_dataloader, onnx_model_path):
    """
    Выполняет инференс на ONNX модели
    
    Args:
        test_dataloader: DataLoader с тестовыми данными
        onnx_model_path: путь к ONNX модели
    """
    
    # Настройка детерминированного выполнения
    sess_options = ort.SessionOptions()
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Создаем сессию ONNX Runtime
    # providers=['CPUExecutionProvider'] для CPU
    # providers=['CUDAExecutionProvider'] для GPU
    ort_session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
    
    # Получаем имена входов и выходов модели
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    
    all_predictions = []
    
    # Переключаемся в режим инференса
    with torch.no_grad():
        for data_batch in tqdm(test_dataloader, desc="Run the onnx model inference to predict a lables on the test data"):
            # Конвертируем входные данные в numpy array
            input_data = to_numpy(data_batch)
            
            # Выполняем инференс
            ort_outputs = ort_session.run(
                [output_name], 
                {input_name: input_data}
            )
            
            # Извлекаем результаты (ort_outputs возвращает список)
            predictions = ort_outputs[0]
            
            # Сохраняем результаты
            all_predictions.append(predictions)
    
    # Объединяем все результаты
    all_predictions = np.concatenate(all_predictions, axis=0)
    
    return all_predictions

@hydra.main(version_base=None, config_path="./configs", config_name="kws.yaml")
def main(cfg: DictConfig):
    # Укажите путь к вашей ONNX модели
    onnx_model_path = Path(cfg.onnx.save_path)
    assert(onnx_model_path.exists())
    onnx_model_path = onnx_model_path / cfg.onnx.onnx_file_name
    assert(onnx_model_path.exists())
        
    data_module = AudioDataModule(cfg.data)
    data_module.setup()
    test_dataloader = data_module.test_dataloader()
    
    # Выполняем инференс
    logits = onnx_inference(test_dataloader, onnx_model_path)
    predicted_labels, probabilities = get_predictions_from_logits(logits, method='numpy')
    
    id2label = {label: key for key, label in label2id.items()}
    
    predicted_names = [id2label[label] for label in predicted_labels]
        
    test_manifest_path = Path(cfg.data.dataset_path)
    assert(test_manifest_path.exists())
    test_manifest_path = test_manifest_path / "test" / "manifest.csv"
    assert(test_manifest_path.exists())
    
    test_data_df = pd.read_csv(test_manifest_path)
    test_data_df['label'] = predicted_names
    test_data_df[["index", "label"]].to_csv("submission.csv", index=False)
    
# Пример использования
if __name__ == "__main__":
    main()