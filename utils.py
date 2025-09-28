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