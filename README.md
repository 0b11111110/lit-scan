usage:
```
lit-scan.py [-x X0] [-y Y0] [-w WIDTH] [-h HEIGH] [-c] [-r] [-n] [-m] [-q] [-s] [-d] [-?]
                   input_folder output_base

Программа для автоматической сортировки сканов по распознанным печатным шифрам в рамке

positional arguments:
  input_folder          Папка с исходными изображениями
  output_base           Базовая папка для сортировки результатов

options:
  -x X0, --x0 X0        X координата области с шифром
  -y Y0, --y0 Y0        Y координата области с шифром
  -w WIDTH, --width WIDTH
                        Ширина области с шифром
  -h HEIGH, --heigh HEIGH
                        Высота области с шифром
  -c, --copy            Не перемещать, а копировать файлы
  -r, --recursive       Рекурсивный обход поддиректорий, в output_base будет аналогичная иерархия
  -n, --not_fix_wrong   Не предлагать исправлять все неверно распознанные пока не разложатся все файлы
  -m, --manual_only     Не распознавать - только ручная проверка
  -q, --quiet           Выводить только процент выполнения
  -s, --silent          Не выводить ничего
  -u UNPROCESSED_FILE, --unprocessed UNPROCESSED_FILE
                        Файл для сохранения списка необработанных изображений (по умолчанию: output_base/0-unprocessed.txt)
  -d, --debug           Сохранять промежуточные изображения для отладки
  -?, --help            Показать это сообщение и выйти
```
if you choose `--recursive` in `output_base`, the same hierarchical structure as in `input_folder` will NOT be created
`-m`, `--manual_only` - just manual recognising, has more priority than `-n`