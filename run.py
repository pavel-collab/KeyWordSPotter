import subprocess
import sys
from typing import List, Dict, Any
from utils import check_params
from sklearn.model_selection import ParameterGrid


def check_model_parameters(params: dict):
    sample_rate = params['sample_rate']
    n_fft = params['n_fft']
    win_length = params['win_length']
    hop_length = params['hop_length']
    n_mels = params['n_mels']

    errors = check_params(
        sample_rate,
        n_fft,
        win_length,
        hop_length,
        n_mels
    )
    return len(errors) == 0

def generate_parameter_combinations() -> List[Dict[str, Any]]:
    grid = {
        "sample_rate": [2000 * (2 ** rate_pow) for rate_pow in range(0, 4)],
        "n_fft": [n for n in range(200, 800 + 1, 200)],
        "hop_length": [2 ** p for p in range(7, 10 + 1)],
        "win_length": [n for n in range(200, 800 + 1, 200)],
        "n_mels": [2 ** p for p in range(4, 7 + 1)]
    }

    valid_param_combinations = [p for p in ParameterGrid(grid) if check_model_parameters(p)]
    return valid_param_combinations


def parameters_to_hydra_args(parameters: Dict[str, Any]) -> List[str]:
    args = [f"data.sample_rate={parameters['sample_rate']}", f"data.n_fft={parameters['n_fft']}",
            f"data.hop_length={parameters['hop_length']}", f"data.win_length={parameters['win_length']}",
            f"data.n_mels={parameters['n_mels']}"]

    return args


def run_experiment(script_path: str, parameters: Dict[str, Any],
                   base_args: List[str] = None) -> bool:
    if base_args is None:
        base_args = []

    hydra_args = parameters_to_hydra_args(parameters)

    # Формируем полную команду
    command = [sys.executable, script_path] + base_args + hydra_args

    print(f"Run experiment with parameters: {parameters}")
    print(f"Command: {' '.join(command)}")
    print("-" * 80)

    try:
        # Запускаем процесс
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True
        )

        print(f"Successful experiment result!")
        if result.stdout:
            print(f"STDOUT:\n{result.stdout}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Ошибка при выполнении эксперимента!")
        print(f"STDERR:\n{e.stderr}")
        if e.stdout:
            print(f"STDOUT:\n{e.stdout}")
        return False


def main():
    """
    Основная функция для запуска всех экспериментов.
    """
    # Конфигурация
    SCRIPT_TO_RUN = "train.py"
    BASE_ARGUMENTS = ["data.num_workers=4", "model.backbone=MicroCNN"]

    # Генерируем комбинации параметров
    parameter_combinations = generate_parameter_combinations()

    print(f"Всего экспериментов для запуска: {len(parameter_combinations)}")

    # Запускаем эксперименты последовательно
    successful_runs = 0
    failed_runs = 0

    for i, parameters in enumerate(parameter_combinations, 1):
        print(f"\nЗапуск эксперимента {i}/{len(parameter_combinations)}")

        try:
            success = run_experiment(SCRIPT_TO_RUN, parameters, BASE_ARGUMENTS)
        except KeyboardInterrupt as e:
            print(f"[DEBUG] the experiment execution was interupted by use keyboard signal")
            break

        if success:
            successful_runs += 1
        else:
            failed_runs += 1

        print(f"Прогресс: {successful_runs} успешных, {failed_runs} неудачных")

    # Итоговый отчет
    print("\n" + "=" * 80)
    print("ИТОГИ:")
    print(f"Успешных экспериментов: {successful_runs}")
    print(f"Неудачных экспериментов: {failed_runs}")
    print(f"Всего запущено: {len(parameter_combinations)}")


if __name__ == "__main__":
    main()