import hydra
import omegaconf
from datetime import datetime
from functools import wraps
import torch
import onnxruntime as ort
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

def omegaconf_extension(func):
    @wraps(func)
    def wrapper(*args, **kwargs):        
        # Регистрируем кастомные резолверы для даты
        omegaconf.OmegaConf.register_new_resolver(
            "now", 
            lambda format_str="%Y-%m-%d_%H-%M-%S": datetime.now().strftime(format_str)
        )
        
        omegaconf.OmegaConf.register_new_resolver(
            "today",
            lambda format_str="%Y-%m-%d": datetime.now().strftime(format_str)
        )
        
        omegaconf.OmegaConf.register_new_resolver(
            "timestamp",
            lambda: str(int(datetime.now().timestamp()))
        )
        
        # Резолвер для генерации уникальных ID
        omegaconf.OmegaConf.register_new_resolver(
            "unique_id",
            lambda length=8: datetime.now().strftime(f"%f%S%M%H{length}")
        )
        
        # Вызываем оригинальную функцию
        return func(*args, **kwargs)
    return wrapper

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

def onnx_inference(dataloader, onnx_model_path, validation=False):
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
    all_labels = []
    
    # Переключаемся в режим инференса
    with torch.no_grad():
        for data_batch in tqdm(dataloader, desc="Run the onnx model inference to predict a lables on the test data"):
            if not validation:
                # Конвертируем входные данные в numpy array
                input_data = to_numpy(data_batch)
            else:
                data, labels = data_batch
                input_data = to_numpy(data)
                all_labels.extend(labels)
            
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
    
    if not validation:
        return all_predictions
    else:
        return all_predictions, all_labels

# Скрипт проверяет корректное сочетание параметров для цифровой обработки аудио
def check_params(sample_rate, n_fft, win_length, hop_length, n_mels, f_min=0, f_max=None): 
    errors = []

    if win_length > n_fft:
        errors.append(f"win_length={win_length} can't be grater n_fft={n_fft}")

    if hop_length <= 0:
        errors.append("hop_length have to be > 0")

    if n_mels > 1 + n_fft // 2:
        errors.append(f"n_mels={n_mels} too high (max value {1 + n_fft // 2})")

    nyquist = sample_rate / 2
    if f_max is None:
        f_max = nyquist
    if f_max > nyquist:
        errors.append(f"f_max={f_max} is grater than Nyquist freq {nyquist}")

    # win_ms = win_length / sample_rate * 1000
    # hop_ms = hop_length / sample_rate * 1000

    return errors