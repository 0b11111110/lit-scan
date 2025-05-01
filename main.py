import argparse
import os
import re
import sys
import shutil

# import sys
import cv2
import numpy as np

# import matplotlib.pyplot as plt
import pytesseract

# import easyocr
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# from matplotlib.widgets import TextBox

## you have to have installed tesseract [in $PATH] with russian language https://github.com/tesseract-ocr/tesseract/releases
# Установите путь к tesseract.exe, если он не в PATH
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Настройки по умолчанию
DEFAULT_COORDS = (1100, 0, 1200, 700)  # x, y, w, h
needed_manual_recognizing = []  # для не распознанных

# def enable_unicode_windows():
#     if sys.platform == 'win32':
#         import ctypes
#         kernel32 = ctypes.windll.kernel32
#         kernel32.SetConsoleCP(65001)
#         kernel32.SetConsoleOutputCP(65001)

# enable_unicode_windows()


def translit(name):
    """
    Транслитерация кириллицы в латиницу (упрощенная схема)
    с использованием str.translate() для максимальной производительности

    :param name: Исходный текст на кириллице
    :return: Транслитерированный текст
    """
    # Словарь замены (кириллица -> латиница)
    trans_dict = {
        "а": "a",
        "б": "b",
        "в": "v",
        "г": "g",
        "д": "d",
        "е": "e",
        "ё": "yo",
        "ж": "zh",
        "з": "z",
        "и": "i",
        "й": "y",
        "к": "k",
        "л": "l",
        "м": "m",
        "н": "n",
        "о": "o",
        "п": "p",
        "р": "r",
        "с": "s",
        "т": "t",
        "у": "u",
        "ф": "f",
        "х": "h",
        "ц": "ts",
        "ч": "ch",
        "ш": "sh",
        "щ": "shch",
        "ъ": "",
        "ы": "y",
        "ь": "",
        "э": "e",
        "ю": "yu",
        "я": "ya",
        "А": "A",
        "Б": "B",
        "В": "V",
        "Г": "G",
        "Д": "D",
        "Е": "E",
        "Ё": "Yo",
        "Ж": "Zh",
        "З": "Z",
        "И": "I",
        "Й": "Y",
        "К": "K",
        "Л": "L",
        "М": "M",
        "Н": "N",
        "О": "O",
        "П": "P",
        "Р": "R",
        "С": "S",
        "Т": "T",
        "У": "U",
        "Ф": "F",
        "Х": "H",
        "Ц": "Ts",
        "Ч": "Ch",
        "Ш": "Sh",
        "Щ": "Shch",
        "Ъ": "",
        "Ы": "Y",
        "Ь": "",
        "Э": "E",
        "Ю": "Yu",
        "Я": "Ya",
    }

    # Создаем таблицу перевода (для метода translate)
    trans_table = str.maketrans(trans_dict)

    # Применяем транслитерацию
    return name.translate(trans_table)

def ensure_valid_folder_name(name):
    """Заменяет запрещённые символы в именах папок"""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, "_")
    return name.strip()


def old_find_text_contour(image):
    """Находит прямоугольный контур вокруг текста"""
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edged = cv2.Canny(blurred, 150, 200)

    contours, _ = cv2.findContours(
        edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:  # Если найден прямоугольник
            return approx

    return None


def find_text_contour(image):
    """Находит прямоугольный контур вокруг текста с учетом рамки"""
    # Улучшенная бинаризация
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    
    # Увеличение контуров
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edged, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilated.copy(), 
                                 cv2.RETR_EXTERNAL, 
                                 cv2.CHAIN_APPROX_SIMPLE)
    
    # Фильтрация контуров по площади и форме
    min_area = 500  # Минимальная площадь контура
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        # Проверяем, что контур прямоугольный и достаточно большой
        if len(approx) == 4 and cv2.contourArea(contour) > min_area:
            return approx
    
    return None


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
    cropped = image[y:y+h, x:x+w]
    return cropped


def old_preprocess_image(image):
    """Улучшение изображения перед распознаванием"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, h=30)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(thresh, -1, kernel)
    clean = cv2.medianBlur(sharpened, 7)
    return clean

def ext_preprocess_image(image):
    """Финальная обработка с сохранением толстых рамок и текста"""
    # 1. Подготовка изображения
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Улучшение контраста (CLAHE лучше работает для текста)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 3. Адаптивная бинаризация (сохраняет толстые линии)
    binary = cv2.adaptiveThreshold(enhanced, 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY,
                                 21, 5)  # Увеличиваем размер блока для сохранения толстых линий
    
    # 4. Морфологические операции для сохранения толстых элементов
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    
    # 5. Удаление мелкого шума без удаления толстых линий
    denoised = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_clean, iterations=1)
    
    # 6. Усиление толстых линий
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    enhanced_lines = cv2.dilate(denoised, kernel_dilate, iterations=1)
    
    return enhanced_lines


def preprocess_image(image):
    """Альтернативный вариант с выделением толстых линий"""
    # 1. Конвертация в grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Размытие для уменьшения шума
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Детекция толстых линий (используем threshold вместо adaptiveThreshold)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 4. Находим только толстые линии (удаляем мелкие элементы)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # 5. Инвертируем обратно
    result = cv2.bitwise_not(closed)
    
    return result


def deskew_image(image, max_angle=20):
    """Выравнивает наклоненный текст, не превышая максимальный угол поворота"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    
    # Определяем угол наклона
    coords = np.column_stack(np.where(gray > 0))
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    
    # Корректируем угол в зависимости от ориентации
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    # Если угол превышает максимальный - оставляем максимальный с сохранением направления
    if abs(angle) > max_angle:
        angle = max_angle if angle > 0 else -max_angle
    
    # Не поворачиваем если угол меньше 1 градуса
    if abs(angle) < 1.0:
        return image.copy()
    
    # Поворачиваем изображение
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)
    return rotated


def deskew_image_by_frame(cropped_img, max_angle=20, frame_thickness=15, debug=False):
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
            print("Ошибка: пустое изображение")
            return None, 0
            
        # Приведение к uint8 если нужно
        if img_np.dtype != np.uint8:
            img_np = img_np.astype(np.uint8)
            
    except Exception as e:
        print(f"Ошибка подготовки изображения: {str(e)}")
        return None, 0

    # 2. Подготовка grayscale изображения
    try:
        if len(img_np.shape) == 2:
            gray = img_np
        elif len(img_np.shape) == 3 and img_np.shape[2] == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        else:
            print("Ошибка: неподдерживаемый формат изображения")
            return None, 0
    except Exception as e:
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
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            if debug: print("Контуры не найдены")
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
        
        if abs(final_angle) < 0.1:
            if debug: print(f"Угол слишком мал: {final_angle:.2f}°")
            return img_np.copy(), 0.0
        
        # Поворот изображения
        h, w = img_np.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), final_angle, 1.0)
        rotated = cv2.warpAffine(img_np, M, (w, h),
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)
        
        return rotated, final_angle
        
    except Exception as e:
        print(f"Ошибка обработки изображения: {str(e)}")
        return None, 0


# def recognize_with_easyocr(image):
#     """Распознавание с помощью EasyOCR"""
#     reader = easyocr.Reader(["ru"])
#     result = reader.readtext(image, detail=0)
#     return "".join(result).upper()


def resize_and_invert(image, target_height=200):
    """
    Уменьшает изображение до 200px в высоту (с пропорциями) и инвертирует цвета.
    
    Args:
        image (numpy.ndarray): Входное изображение (BGR или Grayscale).
        target_height (int): Желаемая высота (по умолчанию 200px).
    
    Returns:
        numpy.ndarray: Уменьшенное и инвертированное изображение.
    """
    # Получаем текущие размеры
    h, w = image.shape[:2]
    
    # Рассчитываем новый размер с сохранением пропорций
    ratio = target_height / h
    new_width = int(w * ratio)
    resized = cv2.resize(image, (new_width, target_height), interpolation=cv2.INTER_AREA)
    
    # Инвертируем цвета (чтобы текст был черным на белом фоне)
    inverted = cv2.bitwise_not(resized)
    
    return inverted


def recognize_with_tesseract(image):
    """Распознавание с помощью Tesseract"""
    custom_config = r"--oem 3 --psm 6"
    # " -c tessedit_char_whitelist=АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ0123456789"  # не нужно, тк проблема с кодировкой и он их не так воспринимает
    text = pytesseract.image_to_string(image, config=custom_config, lang="rus")
    return "".join(c for c in text if c.isalnum())


def correct_text(text):
    """Коррекция распознанного текста"""
    correction_map_letters = {
        "0": "О",
        "4": "А",
        "A": "А",
        "B": "В",
        "C": "С",
        "D": "О",
        "E": "Е",
        "H": "Н",
        "K": "К",
        "M": "М",
        "O": "О",
        "P": "Р",
        "T": "Т",
        "X": "Х",
        "Y": "У",
    }
    correction_map_digits = {
        "А": "4",
        "O": "0",  # латинская
        "О": "0",
        "З": "3",
        "Ч": "4",
        "Б": "6",
        "T": "7",  # латинская
        "Т": "7",
        " ": "",
        "D": "0",
        "S": "5",
    }
    text = text.upper()
    if text:
        corrected = [
            (
                correction_map_letters[text[0]]
                if text[0] in correction_map_letters
                else text[0]
            )
        ]
    else:
        return ""
    for c in text[1:]:
        if c in correction_map_digits:
            corrected.append(correction_map_digits[c])
        elif c in "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ0123456789":
            corrected.append(c)
    return "".join(corrected)


def manual_check(img, predicted_text, filename=""):
    troot = tk.Tk() # для использования Toplevel окон
    troot.withdraw()  # Скрываем главное окно
    root = tk.Toplevel()  # Используем Toplevel вместо Tk для дочерних окон
    root.title("Проверка распознавания")
    
    # Делаем окно модальным (но не блокирующим полностью)
    root.grab_set()
    root.focus_force()
    
    # Конвертируем изображение
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(img_rgb)
    
    # Масштабирование
    max_width = 800
    if pil_image.width > max_width:
        ratio = max_width / pil_image.width
        new_height = int(pil_image.height * ratio)
        pil_image = pil_image.resize((max_width, new_height), Image.LANCZOS)
    
    tk_image = ImageTk.PhotoImage(pil_image)
    
    # Отображение
    img_label = ttk.Label(root, image=tk_image)
    img_label.image = tk_image  # Сохраняем ссылку!
    img_label.pack(pady=10)
    
    # Текст и поле ввода
    ttk.Label(root, text=f"Файл: {filename}\nРаспознанный код:").pack()
    ttk.Label(root, text=predicted_text, font=("Arial", 12, "bold")).pack()
    
    ttk.Label(root, text="ESC - Выход из редактирования всех оставшихся").pack()
    ttk.Label(root, text="Enter - подтвердить. Пустая строка для пропуска").pack()
    
    entry_var = tk.StringVar(value=predicted_text)
    entry = ttk.Entry(root, textvariable=entry_var, width=30, font=("Arial", 12))
    entry.pack(pady=10)
    
    # Переменная для результата
    result = [predicted_text]
    
    def confirm():
        result[0] = entry.get()
        troot.quit()
        root.quit()
    
    def escape():
        root.destroy()
        troot.destroy()
        sys.exit(0)
    
    # Кнопки
    btn_frame = ttk.Frame(root)
    btn_frame.pack(pady=10)
    
    ttk.Button(btn_frame, text="Подтвердить (Enter)", command=confirm).pack(side=tk.LEFT, padx=5)
    ttk.Button(btn_frame, text="Выход (ESC)", command=escape).pack(side=tk.LEFT, padx=5)
    
    # Настройка горячих клавиш
    entry.bind("<Return>", lambda e: confirm())
    root.bind("<Return>", lambda e: confirm())
    root.bind("<Escape>", lambda e: escape())
    
    # Фокус и выделение текста
    entry.focus_force()
    entry.select_range(0, tk.END)
    
    # Центрирование окна
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'+{x}+{y}')
    
    # Ждем закрытия окна
    root.mainloop()
    
    return result[0]


def find_image_files(input_folder, recursive=False, extensions=('png', 'jpg', 'jpeg', 'tif', 'bmp')):
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


def print_progress(current, total, prefix='', suffix=''):
    """
    Выводит прогресс в процентах с возможностью обновления строки
    :param current: текущее количество обработанных элементов
    :param total: общее количество элементов
    :param prefix: текст перед процентом
    :param suffix: текст после процента
    """
    percent = 100 * (current / float(total))
    # \r возвращает каретку в начало строки
    # end='' предотвращает перенос строки
    sys.stdout.write(f'\r{prefix}{percent:.1f}%{suffix}\r')
    sys.stdout.flush()
    if current == total:
        print()  # перенос строки после завершения


def process_image_from_memory(img, coords, orig_name, file_path, need_manual_check=False, debug=False):
    """Обработка изображения из памяти"""
    x, y, w, h = coords
    
    # Вырезаем область интереса
    roi = img[y : y + h, x : x + w]

    # Выравниваем наклон при необходимости
    # roi = deskew_image(roi, max_angle=5)
    roi, _ = deskew_image_by_frame(roi, max_angle=5, frame_thickness=8)

    if debug:
        debug_dir = "ocr_debug"
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(f"{debug_dir}/{orig_name}_roi.png", roi)

    # Предварительная обработка
    processed = preprocess_image(roi)
    if debug:
        debug_dir = "ocr_debug"
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(f"{debug_dir}/{orig_name}_preprocessed.png", processed)

    # Находим контур текста
    contour = find_text_contour(processed)

    if contour is not None:
        # Обрезаем по контуру
        cropped = crop_to_contour(processed, contour)
        if debug:
            cv2.imwrite(f"{debug_dir}/{orig_name}_cropped.png", cropped)
    else:
        cropped = roi  # Если контур не найден, используем всю область

    code, size = "", 70
    while not code and size > 20:
        size -= 20
        resized = resize_and_invert(cropped, target_height=size)
        if debug:
            debug_dir = "ocr_debug"
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(f"{debug_dir}/{orig_name}_resized.png", resized)
        recognized_text = recognize_with_tesseract(resized)
        corrected_text = correct_text(recognized_text)
        code = code[0] if (code := re.match(r"[А-Я]\d{4}", corrected_text)) else ""

    if not code:
        if need_manual_check:
            corrected_text = manual_check(roi, corrected_text, orig_name)
        else:
            if debug:
                print(corrected_text, file=open(f"{debug_dir}/{orig_name}.txt", "wt", encoding='utf-8'))
            corrected_text = ""
        if file_path not in needed_manual_recognizing:
            needed_manual_recognizing.append(file_path)

    return corrected_text if corrected_text else None


def organize_files(
    input_folder, output_base, coords, copy=False, debug=False, not_fix_wrong=False, recursive=False
):
    """Организация файлов по папкам с поддержкой рекурсивного обхода"""
    os.makedirs(output_base, exist_ok=True)
    
    files = find_image_files(input_folder, recursive)
    total_files = len(files)
    if not files:
        print("Нет файлов для обработки!")
        return (0, 0)

    def process(files, need_manual_check=False, debug=False):
        moved_files = 0
        for i, file_path in enumerate(files):
            print_progress(i, total_files, prefix='Обработка файлов: ')
            
            # Получаем оригинальное имя файла
            orig_name = os.path.basename(file_path)
            
            try:
                base_dir = os.path.dirname(file_path)
                tmp_name = os.path.join(base_dir, translit(orig_name))
                os.rename(file_path, tmp_name) # cv2 плохо с виндовой кирилицей
                # Загружаем изображение
                img = cv2.imread(tmp_name)
                os.rename(tmp_name, file_path)
                if img is None:
                    print(f"Ошибка загрузки: {file_path}")
                    continue

                # Обработка изображения
                code = process_image_from_memory(img, coords, orig_name, file_path, need_manual_check, debug)

                if code:
                    folder_name = ensure_valid_folder_name(code)
                    target_folder = os.path.join(output_base, folder_name)
                    os.makedirs(target_folder, exist_ok=True)
                    target_path = os.path.join(target_folder, orig_name)
                    
                    if copy:
                        shutil.copy2(file_path, target_path)
                    else:
                        shutil.move(file_path, target_path)
                    
                    print(f"{orig_name} → {folder_name}\\          ")
                    moved_files += 1
                    
            except Exception as e:
                if debug:
                    print(f"Ошибка обработки {file_path}: {str(e)}")
                    
        return moved_files
    
    # Первый проход - автоматическое распознавание
    rec = process(files, debug=debug)
    unrec = len(needed_manual_recognizing)
    
    # Второй проход - ручная проверка нераспознанных
    if not not_fix_wrong and needed_manual_recognizing:
        print(f"\nНачинаем ручную проверку {unrec} нераспознанных файлов")
        manual_files = needed_manual_recognizing.copy()
        needed_manual_recognizing.clear()
        rec += process(manual_files, need_manual_check=True, debug=debug)
    
    return (rec, unrec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Программа для автоматической сортировки сканов по распознанным печатным шифрам в рамке",
        add_help=False
    )
    parser.add_argument("input_folder", help="Папка с исходными изображениями")
    parser.add_argument("output_base", help="Базовая папка для сортировки результатов")
    parser.add_argument(
        "-x", "--x0", type=int, default=DEFAULT_COORDS[0], help="X координата области с шифром"
    )
    parser.add_argument(
        "-y", "--y0", type=int, default=DEFAULT_COORDS[1], help="Y координата области с шифром"
    )
    parser.add_argument(
        "-w", "--width", type=int, default=DEFAULT_COORDS[2], help="Ширина области с шифром"
    )
    parser.add_argument(
        "-h" , "--heigh", type=int, default=DEFAULT_COORDS[3], help="Высота области с шифром"
    )
    parser.add_argument(
        "-c", "--copy",
        action="store_true",
        help="Не перемещать, а копировать файлы",
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Рекурсивный обход поддиректорий, в output_base будет аналогичная иерархия",
    )
    parser.add_argument(
        "-n", "--not_fix_wrong",
        action="store_true",
        help="Не предлагать исправлять все неверно распознанные пока не разложатся все файлы",
    )
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Сохранять промежуточные изображения для отладки",
    )
    parser.add_argument(
        "-?", "--help", 
        action="help",
        default=argparse.SUPPRESS,
        help="Показать это сообщение и выйти"
    )

    args = parser.parse_args()

    if not os.path.isdir(args.input_folder):
        print(f"Ошибка: папка {args.input_folder} не существует!")
        exit(1)

    rec, unrec = organize_files(
        input_folder=args.input_folder,
        output_base=args.output_base,
        coords=(args.x0, args.y0, args.width, args.heigh),
        copy=args.copy,
        recursive=args.recursive,
        debug=args.debug,
        not_fix_wrong=args.not_fix_wrong,
    )
    print(f"Распознано {rec} кодов, не распознано - {unrec}")
