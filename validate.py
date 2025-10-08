import hydra
from omegaconf import DictConfig
from pathlib import Path
from pathlib import Path
from sklearn.metrics import classification_report 

from src.dataset.module import AudioDataModule
from utils import onnx_inference, get_predictions_from_logits
from src.dataset.utils import label2id

@hydra.main(version_base=None, config_path="./configs", config_name="kws.yaml")
def main(cfg: DictConfig):
    # Укажите путь к вашей ONNX модели
    onnx_model_path = Path(cfg.onnx.save_path)
    assert(onnx_model_path.exists())
    onnx_model_path = onnx_model_path / cfg.onnx.onnx_file_name
    assert(onnx_model_path.exists())
        
    data_module = AudioDataModule(cfg.data)
    data_module.setup()
    val_dataloader = data_module.val_dataloader()
    
    # Выполняем инференс
    logits, true_labels = onnx_inference(val_dataloader, onnx_model_path, validation=True)
    predicted_labels, probabilities = get_predictions_from_logits(logits, method='numpy')
    
    class_report = classification_report(true_labels, predicted_labels, target_names=list(label2id.keys()))  
    print(class_report)  
    
# Пример использования
if __name__ == "__main__":
    main()