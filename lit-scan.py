"""recognizing codes on scanned sheets"""

import argparse
import os
import re
import sys
import shutil
import time
import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np

import easyocr
from PIL import Image, ImageTk

# Настройки по умолчанию
DEFAULT_COORDS = (1100, 0, 1200, 700)  # x, y, w, h
needed_manual_recognizing = []  # для не распознанных


def find_text_contour(image):
    """Находит прямоугольный контур вокруг текста с учетом рамки и заданного соотношения сторон."""
    # Улучшенная бинаризация
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    # Увеличение контуров
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edged, kernel, iterations=2)

    contours, _ = cv2.findContours(
        dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Целевое соотношение сторон (939:383 ≈ 2.452)
    target_ratio = 939 / 383  # ≈ 2.452
    tolerance = 0.15  # Допуск ±15% (можно настроить)
    min_ratio = target_ratio * (1 - tolerance)  # ~2.08
    max_ratio = target_ratio * (1 + tolerance)  # ~2.82

    # Фильтрация контуров по площади, форме и соотношению сторон
    min_area = 500  # Минимальная площадь контура
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # Проверяем, что контур прямоугольный и достаточно большой
        if len(approx) == 4 and cv2.contourArea(contour) > min_area:
            # Получаем ширину и высоту ограничивающего прямоугольника
            x, y, w, h = cv2.boundingRect(approx)
            current_ratio = w / h

            # Проверяем, что соотношение сторон близко к целевому
            if min_ratio <= current_ratio <= max_ratio:
                return approx

    return None  # Если подходящий контур не найден


def crop_to_contour(image, contour, padding=10):
    """
    Обрезает изображение по контуру, убирая рамку.

    Args:
        image: Исходное изображение.
        contour: Найденный контур рамки.
        padding: Отступ внутрь от границ рамки (в пикселях).

    Returns:
        Обрезанное изображение без рамки.
    """
    # Получаем координаты рамки
    x, y, w, h = cv2.boundingRect(contour)

    # Уменьшаем область обрезки (сдвигаем внутрь)
    x += padding
    y += padding
    w -= 2 * padding  # Уменьшаем ширину с двух сторон
    h -= 2 * padding  # Уменьшаем высоту с двух сторон

    # Проверяем, чтобы новые координаты не вышли за границы
    x = max(x, 0)
    y = max(y, 0)
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)

    # Обрезаем изображение
    cropped = image[y : y + h, x : x + w]
    return cropped


def preprocess_image(image):
    """Альтернативный вариант с выделением толстых линий"""
    # 1. Конвертация в grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Размытие для уменьшения шума
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Детекция толстых линий (используем threshold вместо adaptiveThreshold)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 4. Находим только толстые линии (удаляем мелкие элементы)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 5. Инвертируем обратно
    result = cv2.bitwise_not(closed)

    return result


def deskew_image_by_frame(
    cropped_img,
    max_angle=20,
    frame_thickness=15,
    quiet=False,
    silent=False,
    debug=False,
):
    """
    Универсальная функция выравнивания, работающая с UMat и numpy.ndarray
    Возвращает (выровненное изображение, угол поворота) или (None, 0) при ошибке
    """
    # 1. Проверка и нормализация входных данных
    try:
        # Конвертируем в numpy array если это UMat
        if isinstance(cropped_img, cv2.UMat):
            img_np = cropped_img.get()
        else:
            img_np = np.asarray(cropped_img)

        # Проверка на пустое изображение
        if img_np.size == 0:
            if not quiet and not silent:
                print("Ошибка: пустое изображение")
            return None, 0

        # Приведение к uint8 если нужно
        if img_np.dtype != np.uint8:
            img_np = img_np.astype(np.uint8)

    except Exception as e:
        if debug:
            print(f"Ошибка подготовки изображения: {str(e)}")
        return None, 0

    # 2. Подготовка grayscale изображения
    try:
        if len(img_np.shape) == 2:
            gray = img_np
        elif len(img_np.shape) == 3 and img_np.shape[2] == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        else:
            if not quiet and not silent:
                print("Ошибка: неподдерживаемый формат изображения")
            return None, 0
    except Exception as e:
        if debug:
            print(f"Ошибка конвертации в grayscale: {str(e)}")
        return None, 0

    # 3. Основная обработка (работаем только с numpy)
    try:
        # Размытие
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        # blurred = cv2.medianBlur(gray, 5)  # Лучше для шумных изображений

        # Детекция границ
        edges = cv2.Canny(blurred, 70, 180)

        # Усиление контуров
        kernel = np.ones((frame_thickness, frame_thickness), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)

        # Поиск контуров
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            if debug:
                print("Контуры не найдены")
            return img_np.copy(), 0.0

        # Расчет угла поворота
        angles = []
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            angle = rect[-1]
            angles.append(angle if angle < -45 else angle - 90)

        if not angles:
            return img_np.copy(), 0.0

        median_angle = np.median(angles)
        final_angle = max(-max_angle, min(max_angle, median_angle))

        if abs(final_angle) < 0.5:
            if debug:
                print(f"Угол слишком мал: {final_angle:.2f}°")
            return img_np.copy(), 0.0

        # Поворот изображения
        h, w = img_np.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), final_angle, 1.0)
        rotated = cv2.warpAffine(
            img_np, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )

        return rotated, final_angle

    except Exception as e:
        if debug:
            print(f"Ошибка обработки изображения: {str(e)}")
        return None, 0


def recognize_with_easyocr(image, verbose=False):
    reader = easyocr.Reader(
        ["ru"],
        gpu=False,
        verbose=verbose,
    )  # gpu=True для использования видеокарты
    res = reader.readtext(
        image,
        allowlist="АГЕН0123456789",
        # detail=0
    )
    # return res[0] if res else ""
    if res and res[0][-1] > 0.6:
        return res[0][-2]
    return ""


def manual_check(img, predicted_text, filename="", cur_num=1, total=1):
    "Ручная проверка в графическом интерфейсе Tkinter"

    class ResultContainer:
        def __init__(self):
            self.value = predicted_text
            self.should_exit = False

    manual_check.result = ResultContainer()

    # Функции подтверждения и выхода
    def confirm(_=None):
        manual_check.result.value = str(manual_check.entry_var.get())
        manual_check.root.quit()  # Выходим из mainloop

    def escape(_=None):
        manual_check.result.should_exit = True
        manual_check.root.quit()
        manual_check.root.destroy()
        sys.exit(0)

    def skip(_=None):
        manual_check.result.value = ""
        manual_check.root.quit()

    # Функция автозамены символов (как в старом варианте)
    def on_text_change(*_):
        # fmt: off
        translation_table = str.maketrans({
            'q': 'Й', 'w': 'Ц', 'e': 'У', 'r': 'К', 't': 'Е', 'y': 'Н',
            'u': 'Г', 'i': 'Ш', 'o': 'Щ', 'p': 'З', '[': 'Х', ']': 'Ъ',
            'a': 'Ф', 's': 'Ы', 'd': 'В', 'f': 'А', 'g': 'П', 'h': 'Р',
            'j': 'О', 'k': 'Л', 'l': 'Д', ';': 'Ж', "'": 'Э',
            'z': 'Я', 'x': 'Ч', 'c': 'С', 'v': 'М', 'b': 'И', 'n': 'Т',
            'm': 'Ь', ',': 'Б', '.': 'Ю',
            'Q': 'Й', 'W': 'Ц', 'E': 'У', 'R': 'К', 'T': 'Е', 'Y': 'Н',
            'U': 'Г', 'I': 'Ш', 'O': 'Щ', 'P': 'З',
            'A': 'Ф', 'S': 'Ы', 'D': 'В', 'F': 'А', 'G': 'П', 'H': 'Р',
            'J': 'О', 'K': 'Л', 'L': 'Д',
            'Z': 'Я', 'X': 'Ч', 'C': 'С', 'V': 'М', 'B': 'И', 'N': 'Т',
            'M': 'Ь'
        })
        # fmt: on
        current_text = manual_check.entry_var.get()
        cursor_pos = manual_check.entry.index(tk.INSERT)
        new_text = current_text.translate(translation_table).upper()
        if new_text != current_text:
            manual_check.entry_var.set(new_text)
            manual_check.entry.icursor(cursor_pos)

    # Создаём главное окно один раз при первом вызове
    if not hasattr(manual_check, "window_initialized"):
        # Инициализация главного окна
        manual_check.root = tk.Tk()
        manual_check.root.title("Ручная проверка кодов")

        manual_check.counter_label = ttk.Label(
            manual_check.root,
            text=f"Прогресс: {cur_num}/{total} ({100*cur_num/total:.1f}%)",
            font=("Arial", 14, "bold"),
        )
        manual_check.counter_label.pack(pady=5)

        # Label для отображения информации о файле
        manual_check.file_label = ttk.Label(manual_check.root, text=f"Файл: {filename}")
        manual_check.file_label.pack()

        # Label для отображения распознанного кода
        manual_check.code_label = ttk.Label(
            manual_check.root, text=predicted_text, font=("Arial", 12, "bold")
        )
        manual_check.code_label.pack()

        # Label с инструкциями
        ttk.Label(manual_check.root, text="ESC - пропустить").pack()
        ttk.Label(manual_check.root, text="Enter - подтвердить").pack()

        # Label для изображения
        manual_check.img_label = ttk.Label(manual_check.root)
        manual_check.img_label.pack(pady=10)

        # Поле для ввода (как в старом варианте)
        manual_check.entry_var = tk.StringVar(value=predicted_text)
        manual_check.entry = ttk.Entry(
            manual_check.root,
            textvariable=manual_check.entry_var,
            width=30,
            font=("Arial", 12),
        )
        manual_check.entry.pack(pady=10)

        # Кнопки (для тех, кто предпочитает мышку)
        btn_frame = ttk.Frame(manual_check.root)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Подтвердить (Enter)", command=confirm).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(btn_frame, text="Пропустить (ESC)", command=skip).pack(
            side=tk.LEFT, padx=5
        )

        # Горячие клавиши (как в старом варианте)
        manual_check.root.bind("<Return>", confirm)
        manual_check.root.bind("<Escape>", skip)
        manual_check.root.protocol("WM_DELETE_WINDOW", escape)

        manual_check.window_initialized = True

    # Обновляем содержимое окна для нового изображения
    manual_check.counter_label.config(
        text=f"Прогресс: {cur_num}/{total} ({100*cur_num/total:.1f}%)"
    )
    manual_check.file_label.config(text=f"Файл: {filename}")
    manual_check.code_label.config(text=predicted_text)
    manual_check.entry_var.set(predicted_text)

    # Обновляем изображение
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(img_rgb)

    # Масштабирование (как в старом варианте)
    max_width = 600
    if pil_image.width > max_width:
        ratio = max_width / pil_image.width
        new_height = int(pil_image.height * ratio)
        pil_image = pil_image.resize((max_width, new_height), Image.Resampling.LANCZOS)

    tk_image = ImageTk.PhotoImage(pil_image)
    manual_check.img_label.config(image=tk_image)
    manual_check.img_label.image = tk_image  # Сохраняем ссылку

    if not hasattr(manual_check, "centred"):
        # Принудительно обновляем окно, чтобы получить его реальные размеры
        manual_check.root.update_idletasks()

        # Вычисляем координаты для центрирования
        x = (
            manual_check.root.winfo_screenwidth() - manual_check.root.winfo_width()
        ) // 2
        y = (
            manual_check.root.winfo_screenheight() - manual_check.root.winfo_height()
        ) // 2

        # Устанавливаем позицию окна
        manual_check.root.geometry(f"+{x}+{y}")
        manual_check.centred = True

    # Фокус и выделение текста (как в старом варианте)
    manual_check.entry.focus_force()
    manual_check.entry.select_range(0, tk.END)

    # Подключаем обработчик изменений
    manual_check.entry_var.trace_add("write", on_text_change)

    # Запускаем главный цикл
    manual_check.root.mainloop()

    if manual_check.result.should_exit:
        manual_check.root.destroy()
        manual_check.root.quit()
        sys.exit(0)

    return manual_check.result.value


def find_image_files(
    input_folder, recursive=False, extensions=("png", "jpg", "jpeg", "tif", "bmp")
):
    """Находит все изображения в директории (рекурсивно при необходимости)"""
    files = []
    if recursive:
        for root, _, filenames in os.walk(input_folder):
            for filename in filenames:
                if filename.lower().endswith(extensions):
                    files.append(os.path.join(root, filename))
    else:
        files = [
            os.path.join(input_folder, f)
            for f in os.listdir(input_folder)
            if f.lower().endswith(extensions)
        ]
    return files


def print_progress(current, total, prefix="", suffix="", silent=False):
    """
    Выводит прогресс в процентах с возможностью обновления строки
    :param current: текущее количество обработанных элементов
    :param total: общее количество элементов
    :param prefix: текст перед процентом
    :param suffix: текст после процента
    """
    if silent:
        return
    percent = 100 * (current / float(total))
    # \r возвращает каретку в начало строки
    # end='' предотвращает перенос строки
    sys.stdout.write(f"\r{prefix}{percent:.1f}% ({current}/{total}){suffix}\r")
    sys.stdout.flush()
    if current == total:
        print()  # перенос строки после завершения


def process_image_from_memory(
    img,
    coords,
    orig_name,
    file_path,
    need_manual_check=False,
    debug=False,
    number=1,
    total=1,
):
    """Обработка изображения из памяти с тройной проверкой кода"""
    x, y, w, h = coords

    # Вырезаем область интереса
    roi = img[y : y + h, x : x + w]

    # Выравниваем наклон
    roi, _ = deskew_image_by_frame(roi, max_angle=5, frame_thickness=8)

    if debug:
        debug_dir = "ocr_debug"
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(f"{debug_dir}/{orig_name}_roi.png", roi)

    # Предварительная обработка
    processed = preprocess_image(roi)
    if debug:
        cv2.imwrite(f"{debug_dir}/{orig_name}_preprocessed.png", processed)

    # Находим контур текста
    contour = find_text_contour(processed)
    cropped = crop_to_contour(processed, contour) if contour is not None else roi
    if cropped.size == 0 or np.sum(cropped < 127) / (cropped.size) < 0.03: # если пусто - белый лист
        return "А0000"      # папка для всех ведомостей и листов без кодов

    # Проверка кода с помощью EasyOCR
    corrected_text = easyocr_and_correct(cropped, debug)

    if not corrected_text and need_manual_check:
        corrected_text = manual_check(roi, corrected_text, file_path, number, total)
        if file_path not in needed_manual_recognizing:
            needed_manual_recognizing.append(file_path)
    # else:
    #     if debug:
    #         print(f"{(b:=np.sum(resized < 127))=}, {(b / (resized.size))=:.5f}")

    # return (corrected_text, resized.shape[0]) if corrected_text else (None, None)
    return corrected_text if corrected_text else None


def code_is_correct(code):
    return re.match(r"^[АГЕН]\d{4}$", code)


def easyocr_and_correct(image, verbose):
    code = recognize_with_easyocr(image, verbose)
    return code if code_is_correct(code) else None


def organize_files(
    input_folder,
    output_base,
    coords,
    copy=False,
    quiet=False,
    silent=False,
    debug=False,
    not_fix_wrong=False,
    manual_only=False,
    recursive=False,
):
    """Организация файлов по папкам с поддержкой рекурсивного обхода"""
    os.makedirs(output_base, exist_ok=True)

    files = find_image_files(input_folder, recursive)
    total_files = len(files)
    if not files:
        if not silent:
            print("Нет файлов для обработки!")
        return (0, 0)

    def process2(files, need_manual_check=False, debug=False):
        moved_files = 0
        pending_file = None  # Для хранения предыдущего (нечётного) файла
        n_prev_file = 3 # т.к. первые - ведомости

        for i, file_path in enumerate(files):
            print_progress(i, total_files, prefix="Обработка файлов: ", silent=silent)

            # Получаем оригинальное имя файла
            orig_name = os.path.basename(file_path)

            try:
                with open(file_path, "rb") as f:
                    img_data = np.frombuffer(
                        f.read(), np.uint8
                    )  # cv2 не дружит с кириллицей
                    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                    del img_data  # не уверен, что он уйдёт вместе с f
                if img is None:
                    if debug:
                        print(f"Ошибка загрузки: {file_path}")
                    continue

                # Обработка изображения
                code = process_image_from_memory(
                    img,
                    coords,
                    orig_name,
                    file_path,
                    need_manual_check,
                    debug,
                    number=i,
                    total=total_files,
                )

                n_file = int(orig_name[-7:-4]) # .../М_302_003.jpg -> 003
                if n_file - n_prev_file > 1:
                    if pending_file:
                        move(pending_file[1], os.path.join(output_base, pending_file[3], pending_file[2]))
                        moved_files += 1
                    if n_file % 2 == 0: # значит перед ним не было первой страницы
                        if code:
                            move(file_path, os.path.join(output_base, code, orig_name))
                            moved_files += 1
                        else:
                            needed_manual_recognizing.append(file_path)
                            if not silent and not quiet:
                                print(f"Не распознан код в {file_path}")
                        pending_file = None
                        n_prev_file = n_file
                        continue
                if n_file % 2 == 1:  # Нечётный файл - первая страница
                    pending_file = (n_file, file_path, orig_name, code)
                else:  # Чётный файл
                    if pending_file:
                        n_prev_file, prev_file_path, prev_orig_name, prev_code = pending_file

                        def code_of_pair(prev_code, code):
                            # Определяем конечный код
                            if code and (not prev_code or prev_code == "А0000"):  # Если текущий распознан, а предыдущий нет
                                return code
                            elif prev_code and (not code or code == "А0000"):  # Если предыдущий распознан, а текущий нет
                                return prev_code
                            elif code and prev_code and code == prev_code:  # Оба распознаны и совпадают
                                return code
                            else:  # Не распознаны или разные коды
                                return None
                        final_code = code_of_pair(prev_code, code)

                        if final_code:
                            folder_name = final_code
                            target_folder = os.path.join(output_base, folder_name)
                            os.makedirs(target_folder, exist_ok=True)

                            # Перемещаем оба файла
                            for path, name in [(prev_file_path, prev_orig_name), (file_path, orig_name)]:
                                target_path = os.path.join(target_folder, name)
                                move(path, target_path)
                            moved_files += 2
                        else:
                            needed_manual_recognizing.extend([prev_file_path, file_path])
                            if not silent and not quiet:
                                print(f"Не распознаны или разные коды: {prev_file_path} и {file_path}")

                        pending_file = None  # Сбрасываем ожидание пары
                n_prev_file = n_file
            except Exception as e:
                if debug:
                    print(f"Ошибка обработки {file_path}: {str(e)}")
                if pending_file:
                    needed_manual_recognizing.append(pending_file[1])
                    pending_file = None
                needed_manual_recognizing.append(file_path)

        # Обработка оставшегося непарного файла в конце
        if pending_file:
            if pending_file[3]:
                move(pending_file[1], os.path.join(output_base, pending_file[3], pending_file[2]))
            else:
                needed_manual_recognizing.append(pending_file[0])
                if not silent and not quiet:
                    print(f"Оставшийся непарный файл: {pending_file[0]}")

        return moved_files

    def move(path, target_path):
        """Перемещает/копирует файл, создавая все необходимые директории"""
        try:
            # Создаем целевую директорию, если её нет
            target_dir = os.path.dirname(target_path)
            os.makedirs(target_dir, exist_ok=True)
            if copy:
                shutil.copy2(path, target_path)
            else:
                shutil.move(path, target_path)
            if not quiet and not silent:
                print(f"{path} → {target_path}")

        except OSError as e:
            if not silent:
                print(f"Ошибка при обработке {path}: {str(e)}")
            needed_manual_recognizing.append(path)

    # Первый проход - автоматическое распознавание
    global needed_manual_recognizing
    if not manual_only:
        rec = process2(files, debug=debug)
    else:
        rec = 0
        needed_manual_recognizing = files
    unrec = len(needed_manual_recognizing)

    # Второй проход - ручная проверка нераспознанных
    if manual_only or not not_fix_wrong and needed_manual_recognizing:
        if not silent:
            print(f"\nНачинаем ручную проверку {unrec} нераспознанных файлов")
        rec += process2(needed_manual_recognizing, need_manual_check=True, debug=debug)
        needed_manual_recognizing.clear()

    if args.unprocessed_file:
        try:
            if os.path.basename(args.unprocessed_file) == args.unprocessed_file:
                file = os.path.join(args.output_base, args.unprocessed_file)
            else:
                file = args.unprocessed_file
            with open(file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(needed_manual_recognizing))
            if not silent:
                print(f"\nСписок необработанных файлов сохранен в: {args.unprocessed_file}")
        except IOError as e:
            if not silent:
                print(f"\nОшибка сохранения списка необработанных файлов: {str(e)}")

    return (rec, unrec)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Программа для автоматической сортировки сканов по распознанным печатным шифрам в рамке",
        add_help=False,
    )
    parser.add_argument("input_folder", help="Папка с исходными изображениями")
    parser.add_argument("output_base", help="Базовая папка для сортировки результатов")
    parser.add_argument(
        "-x",
        "--x0",
        type=int,
        default=DEFAULT_COORDS[0],
        help="X координата области с шифром",
    )
    parser.add_argument(
        "-y",
        "--y0",
        type=int,
        default=DEFAULT_COORDS[1],
        help="Y координата области с шифром",
    )
    parser.add_argument(
        "-w",
        "--width",
        type=int,
        default=DEFAULT_COORDS[2],
        help="Ширина области с шифром",
    )
    parser.add_argument(
        "-h",
        "--heigh",
        type=int,
        default=DEFAULT_COORDS[3],
        help="Высота области с шифром",
    )
    parser.add_argument(
        "-c",
        "--copy",
        action="store_true",
        help="Не перемещать, а копировать файлы",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Рекурсивный обход поддиректорий, в output_base будет аналогичная иерархия",
    )
    parser.add_argument(
        "-n",
        "--not_fix_wrong",
        action="store_true",
        help="Не предлагать исправлять все неверно распознанные пока не разложатся все файлы",
    )
    parser.add_argument(
        "-m",
        "--manual_only",
        action="store_true",
        help="Не распознавать - только ручная проверка",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Выводить только процент выполнения",
    )
    parser.add_argument(
        "-s",
        "--silent",
        action="store_true",
        help="Не выводить ничего",
    )
    parser.add_argument(
    '-u', '--unprocessed',
    dest='unprocessed_file',
    type=str,
    default='0-unprocessed.txt',
    help='Файл для сохранения списка необработанных изображений (по умолчанию: output_base/0-unprocessed.txt)'
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Сохранять промежуточные изображения для отладки",
    )
    parser.add_argument(
        "-?",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="Показать это сообщение и выйти",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.input_folder):
        if args.debug:
            print(f"Ошибка: папка {args.input_folder} не существует!")
        sys.exit(1)

    start_time = time.time()
    recognized, unrecognized = organize_files(
        input_folder=args.input_folder,
        output_base=args.output_base,
        coords=(args.x0, args.y0, args.width, args.heigh),
        copy=args.copy,
        quiet=args.quiet,
        silent=args.silent,
        recursive=args.recursive,
        debug=args.debug,
        not_fix_wrong=args.not_fix_wrong,
        manual_only=args.manual_only,
    )
    duration = time.time() - start_time
    if not args.silent:
        if total := recognized + unrecognized:
            print(
                f"Распознано {recognized} кодов, не распознано - {unrecognized} ({100 * recognized / (total):.2f}%)"
                f", это заняло {duration // (60 * 60):02.0f}:{duration // 60 % 60:02.0f}:{duration % 60:02.0f}"
            )
        else:
            print(
                f"Распознано {recognized} кодов, не распознано - {unrecognized}"
                f", это заняло {duration // (60 * 60):02.0f}:{duration // 60 % 60:02.0f}:{duration % 60:02.0f}"
            )
