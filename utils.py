import hydra
import omegaconf
from datetime import datetime
from functools import wraps

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