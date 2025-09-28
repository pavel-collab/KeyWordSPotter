import logging

# Создание отдельного логгера для ваших сообщений
local_logger = logging.getLogger("kws_local_logger")
local_logger.setLevel(logging.INFO)

# Создание файлового обработчика
file_handler = logging.FileHandler("local_log.log")
file_handler.setLevel(logging.INFO)

# Форматирование
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Добавление обработчика к логгеру
local_logger.addHandler(file_handler)