from __future__ import annotations

import cv2
import numpy as np
from scipy.fft import fft2, fftshift
from scipy.stats import kurtosis, skew
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import gabor


def safe_divide(numerator, denominator, eps: float = 1e-8):
    return numerator / (denominator + eps)


def _clean_features(features: dict[str, float]) -> dict[str, float]:
    cleaned = {}
    for key, value in features.items():
        value = float(value)
        if np.isinf(value) or np.isnan(value):
            value = 0.0
        elif abs(value) > 1e6:
            value = float(np.clip(value, -1e6, 1e6))
        cleaned[key] = value
    return cleaned


def _prepare_crop(image_bgr: np.ndarray, is_already_cropped: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_h, target_w = 128, 64
    image_bgr = cv2.resize(image_bgr, (target_w, target_h))
    if not is_already_cropped:
        crop_start = int(target_w * 0.20)
        crop_end = int(target_w * 0.80)
        image_bgr = image_bgr[:, crop_start:crop_end]

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float64)
    return image_bgr, gray, hsv


def extract_light_features(image_bgr: np.ndarray, is_already_cropped: bool = True) -> dict[str, float] | None:
    if image_bgr is None or image_bgr.size == 0:
        return None

    _image_bgr, gray, hsv = _prepare_crop(image_bgr, is_already_cropped)
    h, w = gray.shape
    gray_uint8 = np.clip(gray, 0, 255).astype(np.uint8)
    features: dict[str, float] = {}

    blurred = cv2.GaussianBlur(gray, (5, 9), 0)
    sobel_y = cv2.convertScaleAbs(cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3))
    edge_profile = np.mean(sobel_y, axis=1)
    features["waterline_y_ratio"] = np.argmax(edge_profile) / h if edge_profile.size else 0.5

    mean_val = np.mean(gray)
    y_coords, _ = np.where(gray < mean_val)
    features["darkness_center_y"] = np.mean(y_coords) / h if len(y_coords) else 0.5

    top_gray = gray[: h // 2, :]
    bottom_gray = gray[h // 2 :, :]
    features["tb_gray_std_ratio"] = safe_divide(np.std(top_gray), np.std(bottom_gray))
    features["tb_gray_mean_diff"] = np.mean(top_gray) - np.mean(bottom_gray)

    for channel, name in enumerate(("h", "s", "v")):
        hist_range = [0, 180] if channel == 0 else [0, 256]
        hist = cv2.calcHist([hsv.astype(np.uint8)], [channel], None, [8], hist_range).flatten() / (h * w)
        for idx, value in enumerate(hist):
            features[f"{name}_hist_{idx}"] = value
        values = hsv[:, :, channel]
        features[f"color_moment_{name}_mean"] = np.mean(values)
        features[f"color_moment_{name}_std"] = np.std(values)

    lbp = local_binary_pattern(gray_uint8, 8, 1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(11), density=True)
    for idx in range(10):
        features[f"lbp_{idx}"] = lbp_hist[idx]

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx) * 180 / np.pi
    angle[angle < 0] += 180
    grad_hist = np.histogram(angle.flatten(), bins=8, range=(0, 180), weights=magnitude.flatten())[0]
    grad_hist = safe_divide(grad_hist, np.sum(grad_hist))
    for idx, value in enumerate(grad_hist):
        features[f"grad_hist_{idx}"] = value

    _, threshold = cv2.threshold(gray_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    features["otsu_white_ratio"] = np.sum(threshold > 0) / (h * w)
    return _clean_features(features)


def extract_full_features(image_bgr: np.ndarray, is_already_cropped: bool = True) -> dict[str, float] | None:
    if image_bgr is None or image_bgr.size == 0:
        return None

    _image_bgr, gray, hsv = _prepare_crop(image_bgr, is_already_cropped)
    h, w = gray.shape
    gray_uint8 = np.clip(gray, 0, 255).astype(np.uint8)
    features: dict[str, float] = {}

    blurred = cv2.GaussianBlur(gray, (5, 9), 0)
    sobel_y = cv2.convertScaleAbs(cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3))
    edge_profile = np.mean(sobel_y, axis=1)
    features["waterline_y_ratio"] = np.argmax(edge_profile) / h if edge_profile.size else 0.5

    mean_val = np.mean(gray)
    y_coords, _ = np.where(gray < mean_val)
    features["darkness_center_y"] = np.mean(y_coords) / h if len(y_coords) else 0.5

    top_gray = gray[: h // 2, :]
    bottom_gray = gray[h // 2 :, :]
    features["tb_gray_std_ratio"] = safe_divide(np.std(top_gray), np.std(bottom_gray))
    features["tb_gray_mean_diff"] = np.mean(top_gray) - np.mean(bottom_gray)

    bin_h = h // 4
    bin_means = [np.mean(gray[idx * bin_h : (idx + 1) * bin_h, :]) for idx in range(4)]
    features["drop_0_1"] = bin_means[0] - bin_means[1]
    features["drop_1_2"] = bin_means[1] - bin_means[2]
    features["drop_2_3"] = bin_means[2] - bin_means[3]

    top_hsv = hsv[: h // 2, :, :]
    bottom_hsv = hsv[h // 2 :, :, :]
    features["bottom_saturation"] = np.mean(bottom_hsv[:, :, 1])
    features["diff_saturation"] = np.mean(top_hsv[:, :, 1]) - features["bottom_saturation"]

    for channel, name in enumerate(("h", "s", "v")):
        hist_range = [0, 180] if channel == 0 else [0, 256]
        hist = cv2.calcHist([hsv.astype(np.uint8)], [channel], None, [16], hist_range).flatten() / (h * w)
        for idx, value in enumerate(hist):
            features[f"{name}_hist_{idx}"] = value
        values = hsv[:, :, channel]
        features[f"color_moment_{name}_mean"] = np.mean(values)
        features[f"color_moment_{name}_std"] = np.std(values)
        features[f"color_moment_{name}_skew"] = skew(values.flatten())
        features[f"color_moment_{name}_kurtosis"] = kurtosis(values.flatten())

    lbp_r1 = local_binary_pattern(gray_uint8, 8, 1, method="uniform")
    lbp_hist_r1, _ = np.histogram(lbp_r1.ravel(), bins=np.arange(11), density=True)
    for idx in range(10):
        features[f"lbp_r1_{idx}"] = lbp_hist_r1[idx]

    lbp_r2 = local_binary_pattern(gray_uint8, 16, 2, method="uniform")
    lbp_hist_r2, _ = np.histogram(lbp_r2.ravel(), bins=np.arange(19), density=True)
    for idx in range(18):
        features[f"lbp_r2_{idx}"] = lbp_hist_r2[idx]

    for theta in (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4):
        for frequency in (0.1, 0.2):
            real, imag = gabor(gray_uint8, frequency=frequency, theta=theta)
            response = np.sqrt(real**2 + imag**2)
            idx = len([name for name in features if name.startswith("gabor_")])
            features[f"gabor_{idx}"] = np.mean(response)
            features[f"gabor_{idx + 1}"] = np.std(response)

    gray_8 = np.clip(gray / 32, 0, 7).astype(np.uint8)
    glcm = graycomatrix(
        gray_8,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=8,
        symmetric=True,
        normed=True,
    )
    for prop in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation"):
        for idx, value in enumerate(graycoprops(glcm, prop).flatten()):
            features[f"glcm_{prop}_{idx}"] = value

    rows, cols = 4, 2
    h_block = h // rows
    w_block = w // cols
    for row in range(rows):
        for col in range(cols):
            gray_block = gray[row * h_block : (row + 1) * h_block, col * w_block : (col + 1) * w_block]
            hsv_block = hsv[row * h_block : (row + 1) * h_block, col * w_block : (col + 1) * w_block, :]
            if gray_block.size == 0 or hsv_block.size == 0:
                continue
            features[f"block_mean_{row}_{col}"] = np.mean(gray_block)
            features[f"block_std_{row}_{col}"] = np.std(gray_block)
            features[f"block_h_mean_{row}_{col}"] = np.mean(hsv_block[:, :, 0])
            features[f"block_s_mean_{row}_{col}"] = np.mean(hsv_block[:, :, 1])

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx) * 180 / np.pi
    angle[angle < 0] += 180
    grad_hist = np.histogram(angle.flatten(), bins=18, range=(0, 180), weights=magnitude.flatten())[0]
    grad_hist = safe_divide(grad_hist, np.sum(grad_hist))
    for idx, value in enumerate(grad_hist):
        features[f"grad_hist_{idx}"] = value

    frequency = fftshift(fft2(gray))
    magnitude_spectrum = np.log1p(np.abs(frequency) + 1e-8)
    center_y, center_x = h // 2, w // 2
    max_radius = min(center_y, center_x)
    radial = np.zeros(15)
    angular = np.zeros(12)
    for y in range(h):
        for x in range(w):
            radius = int(np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2))
            if radius < max_radius:
                radial[int(radius * len(radial) / max_radius)] += magnitude_spectrum[y, x]
            if x == center_x and y == center_y:
                continue
            theta = np.arctan2(y - center_y, x - center_x) + np.pi
            angular[int(theta * len(angular) / (2 * np.pi)) % len(angular)] += magnitude_spectrum[y, x]

    radial = safe_divide(radial, np.sum(radial))
    angular = safe_divide(angular, np.sum(angular))
    for idx, value in enumerate(radial):
        features[f"fft_radial_{idx}"] = value
    for idx, value in enumerate(angular):
        features[f"fft_angular_{idx}"] = value

    edges = cv2.Canny(gray_uint8, 50, 150)
    edge_y, edge_x = np.where(edges > 0)
    if len(edge_x):
        edge_angles = np.arctan2(gy[edge_y, edge_x], gx[edge_y, edge_x]) * 180 / np.pi
        edge_angles[edge_angles < 0] += 180
        edge_hist = np.histogram(edge_angles, bins=18, range=(0, 180))[0]
        edge_hist = safe_divide(edge_hist, np.sum(edge_hist))
    else:
        edge_hist = np.zeros(18)
    for idx, value in enumerate(edge_hist):
        features[f"edge_orientation_{idx}"] = value

    for distance in (1, 2, 4):
        if distance < w:
            horizontal_diff = np.abs(gray[:, distance:] - gray[:, :-distance])
            features[f"diff_h_mean_{distance}"] = np.mean(horizontal_diff)
            features[f"diff_h_std_{distance}"] = np.std(horizontal_diff)
        if distance < h:
            vertical_diff = np.abs(gray[distance:, :] - gray[:-distance, :])
            features[f"diff_v_mean_{distance}"] = np.mean(vertical_diff)
            features[f"diff_v_std_{distance}"] = np.std(vertical_diff)

    for percentile in (10, 25, 50, 75, 90):
        features[f"gray_percentile_{percentile}"] = np.percentile(gray, percentile)

    _, threshold = cv2.threshold(gray_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    features["otsu_white_ratio"] = np.sum(threshold > 0) / (h * w)
    white_y = np.where(threshold > 0)[0]
    features["white_center_y"] = np.mean(white_y) / h if len(white_y) else 0.5
    return _clean_features(features)


def extract_liquid_features(
    image_bgr: np.ndarray,
    feature_set: str = "full",
    is_already_cropped: bool = True,
) -> dict[str, float] | None:
    if feature_set == "light":
        return extract_light_features(image_bgr, is_already_cropped)
    if feature_set == "full":
        return extract_full_features(image_bgr, is_already_cropped)
    raise ValueError(f"Unsupported feature set: {feature_set}")
