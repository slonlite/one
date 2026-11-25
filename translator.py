"""Локальный переводчик EN→RU на facebook/nllb-200-distilled-600M.

Используется NLLB вместо Helsinki-NLP/opus-mt-en-ru, потому что у opus-mt
только pytorch_model.bin (legacy), который transformers 4.57+ не может
загрузить через torch.load на torch < 2.6 (CVE-2025-32434).
NLLB имеет model.safetensors — загружается безопасно на любом torch.

ВАЖНО: переводчик всегда на CPU. DirectML не работает с meta tensors
(transformers загружает NLLB в meta, потом .to(dml) падает).
Для коротких текстов NLLB на CPU работает быстро (~1 сек на сегмент).
Whisper остаётся на DirectML — он больше выигрывает от GPU.

NEW v2.1: применяется runtime monkey-patch функции isin_mps_friendly
в transformers.generation.utils — обходит баг "Tensor.item() cannot be
called on meta tensors" в transformers 4.5x+ на torch 2.15.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

log = logging.getLogger(__name__)

# NLLB language codes
LANG_EN = "eng_Latn"
LANG_RU = "rus_Cyrl"


def _patch_isin_mps_friendly() -> None:
    """Monkey-patch transformers GenerationMixin._prepare_special_tokens.

    В transformers 4.5x+ функция _prepare_special_tokens делает много операций
    с тензорами (isin, is_floating_point, < 0, .any()) — все они падают на
    torch 2.15 с "Tensor.item() cannot be called on meta tensors" если модель
    загружена с meta tensors (что бывает с NLLB).

    Эта функция ПОЛНОСТЬЮ заменяет _prepare_special_tokens на безопасную версию,
    которая использует только Python int для eos_token_id и pad_token_id —
    без tensor operations, которые могут упасть на meta device.

    Идемпотентная — если уже пропатчено, не делает ничего.
    """
    try:
        from transformers.generation.utils import GenerationMixin

        # Проверяем, не пропатчено ли уже
        if hasattr(GenerationMixin, '_original_prepare_special_tokens'):
            return  # уже пропатчено

        original = GenerationMixin._prepare_special_tokens
        GenerationMixin._original_prepare_special_tokens = original

        def safe_prepare_special_tokens(self, generation_config, kwargs_has_attention_mask=False, device=None):
            """Безопасная версия _prepare_special_tokens — без tensor ops на meta device.

            Создаёт 4 тензорных поля, которые нужны GenerationMixin:
            - _bos_token_tensor
            - _eos_token_tensor
            - _pad_token_tensor
            - _decoder_start_token_tensor

            Принудительно используем CPU device (а не self.device, который
            может быть "meta"). Это обход бага "Tensor.item() cannot be
            called on meta tensors" в transformers 4.5x+ на torch 2.15.
            """
            # Безопасное устройство — всегда CPU (meta tensors ломают tensor ops)
            safe_device = torch.device('cpu')

            def _safe_tensor_or_none(token):
                if token is None:
                    return None
                if isinstance(token, torch.Tensor):
                    # Если токен — тензор, конвертируем в обычный int через .tolist()
                    # (это работает даже на meta tensors, в отличие от .item())
                    try:
                        token_list = token.tolist()
                        if isinstance(token_list, list):
                            return torch.tensor(token_list, device=safe_device, dtype=torch.long)
                        else:
                            return torch.tensor([token_list], device=safe_device, dtype=torch.long)
                    except Exception:
                        return token.to(safe_device)
                # token — int или list[int]
                # Делаем tensor (если scalar — добавляем размерность для eos)
                t = torch.tensor(token, device=safe_device, dtype=torch.long)
                return t

            # Создаём 4 тензорных поля
            bos_token_tensor = _safe_tensor_or_none(generation_config.bos_token_id)
            eos_token_tensor = _safe_tensor_or_none(generation_config.eos_token_id)
            pad_token_tensor = _safe_tensor_or_none(generation_config.pad_token_id)
            decoder_start_token_tensor = _safe_tensor_or_none(generation_config.decoder_start_token_id)

            # for BC: encoder-decoder models используют decoder_start_token_id или bos_token_id
            if getattr(self.config, 'is_encoder_decoder', False):
                if decoder_start_token_tensor is None and bos_token_tensor is not None:
                    decoder_start_token_tensor = bos_token_tensor

            # Если eos — scalar (0-dim), делаем 1D
            if eos_token_tensor is not None and eos_token_tensor.ndim == 0:
                eos_token_tensor = eos_token_tensor.unsqueeze(0)

            # Если pad_token_id не задан, но eos есть — используем eos[0] как pad
            if pad_token_tensor is None and eos_token_tensor is not None:
                pad_token_tensor = eos_token_tensor[0]
                log.info("Setting pad_token_id to eos_token_id: %s (for open-end generation)",
                         int(pad_token_tensor.item()))

            # НЕ делаем проверок isin().any(), (eos < 0).any(), is_floating_point() —
            # они падают на meta tensors. Для NLLB eos=2, pad=1 — конфликта нет.

            # Устанавливаем тензорные поля в generation_config
            generation_config._bos_token_tensor = bos_token_tensor
            generation_config._eos_token_tensor = eos_token_tensor
            generation_config._pad_token_tensor = pad_token_tensor
            generation_config._decoder_start_token_tensor = decoder_start_token_tensor

        GenerationMixin._prepare_special_tokens = safe_prepare_special_tokens
        log.info("Patched GenerationMixin._prepare_special_tokens for meta tensor safety")
    except Exception as e:
        log.warning("Failed to patch _prepare_special_tokens: %s", e)


# NOTE: ранее здесь был monkey-patch _prepare_special_tokens, чтобы обойти
# баг "Tensor.item() cannot be called on meta tensors". Теперь модель
# загружается через to_empty + load_state_dict (см. _ensure_loaded), и meta
# tensors исчезают — патч больше не нужен. Код оставлен для справки, но
# закомментирован, чтобы не ломать Whisper (который работает корректно).
# _patch_isin_mps_friendly()


class EnRuTranslator:
    """Переводчик EN→RU через NLLB-200-distilled-600M (всегда CPU)."""

    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        device: str = "cpu",  # всегда CPU — DirectML не работает с meta tensors
        num_beams: int = 4,
    ):
        self.model_name = model_name
        # Принудительно CPU — DirectML + NLLB meta tensors не работают
        self.device = "cpu"
        self.num_beams = num_beams
        self._tok = None
        self._model = None
        self._forced_bos_token_id: Optional[int] = None
        self._eos_token_id: Optional[int] = None
        self._pad_token_id: Optional[int] = None

    def _resolve_model_path(self) -> str:
        """Возвращает путь к модели: локальная папка (если есть) или HF-имя.

        Если пользователь скачал модель в models/nllb-200-distilled-600M/
        (через snapshot_download с local_dir), используем её напрямую —
        это надёжнее, чем кэш HuggingFace с симлинками.
        """
        from pathlib import Path
        from config import MODELS_DIR

        local_dir = MODELS_DIR / "nllb-200-distilled-600M"
        if (local_dir / "config.json").exists() and (local_dir / "model.safetensors").exists():
            log.info("Использую локальную модель: %s", local_dir)
            return str(local_dir)
        return self.model_name

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        model_path = self._resolve_model_path()
        log.info("Загрузка переводчика %s (CPU)", model_path)
        self._tok = AutoTokenizer.from_pretrained(model_path)
        # NLLB должен знать язык исходного текста до токенизации.
        self._tok.src_lang = LANG_EN
        log.info("NLLB: исходный язык=%s, целевой=%s", LANG_EN, LANG_RU)

        # v2.1: загружаем модель напрямую из safetensors без low_cpu_mem_usage.
        #
        # Проблема: в transformers 4.46.3 параметр low_cpu_mem_usage=True требует
        # пакет accelerate (pip install accelerate). В 4.51.3 он создавал meta
        # tensors даже без accelerate — оба варианта ломают генерацию.
        #
        # Решение: грузим с use_safetensors=True и БЕЗ low_cpu_mem_usage.
        # transformers 4.46.3 при этом читает safetensors напрямую и создаёт
        # реальные CPU-тензоры (не meta). Для модели 600M это ~2.5 ГБ RAM — нормально.
        try:
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                use_safetensors=True,
            )
        except Exception as e:
            log.warning("Загрузка с use_safetensors не удалась (%s), пробую обычную", e)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

        # Проверяем, нет ли meta tensors (бывают в редких случаях)
        has_meta = any(p.is_meta for p in self._model.parameters())
        log.info("Модель загружена, meta tensors: %s", has_meta)

        if has_meta:
            # Fallback: заполняем meta tensors реальными весами из safetensors
            st_path = self._find_safetensors_in_cache()
            if st_path is not None:
                log.info("Загрузка весов из safetensors: %s", st_path)
                from safetensors.torch import load_file
                state_dict = load_file(str(st_path))
                self._model = self._model.to_empty(device="cpu")
                missing, unexpected = self._model.load_state_dict(state_dict, strict=False)
                if missing:
                    log.warning("Missing keys при загрузке: %d", len(missing))
                if unexpected:
                    log.warning("Unexpected keys при загрузке: %d", len(unexpected))
            else:
                log.error("safetensors не найден и модель в meta tensors — генерация упадёт!")

        has_meta_after = any(p.is_meta for p in self._model.parameters())
        log.info("После загрузки весов, meta tensors: %s", has_meta_after)

        # Модель уже на CPU с реальными данными
        self._model.eval()

        # ID форсированного языка (rus_Cyrl) — NLLB требует его в generate
        self._forced_bos_token_id = self._tok.convert_tokens_to_ids(LANG_RU)
        self._eos_token_id = self._tok.eos_token_id
        self._pad_token_id = self._tok.pad_token_id
        log.info("Переводчик готов на CPU, forced_bos=%s, eos=%s, pad=%s",
                 self._forced_bos_token_id, self._eos_token_id, self._pad_token_id)

    def _find_safetensors_in_cache(self):
        """Найти model.safetensors в кэше HuggingFace или локальной папке.

        Ищет в:
        - models/nllb-200-distilled-600M/ (локальная папка Tolmach)
        - ~/.cache/huggingface/hub/models--facebook--nllb-200-distilled-600M/snapshots/<hash>/
        - %USERPROFILE%/.cache/huggingface/hub/models--facebook--nllb-200-distilled-600M/snapshots/<hash>/
        """
        import os
        from pathlib import Path
        from config import MODELS_DIR

        # 1. Сначала ищем в локальной папке Tolmach (если пользователь скачал туда)
        local_dir = MODELS_DIR / "nllb-200-distilled-600M"
        local_st = local_dir / "model.safetensors"
        if local_st.exists() and local_st.stat().st_size > 1024 * 1024:
            return local_st

        # 2. Ищем в кэше HuggingFace
        model_dir_name = "models--" + self.model_name.replace("/", "--")
        home = Path.home()
        possible_caches = [
            home / ".cache" / "huggingface" / "hub" / model_dir_name,
            Path(os.environ.get("HF_HOME", "")) / "hub" / model_dir_name,
            Path(os.environ.get("HUGGINGFACE_HUB_CACHE", "")) / model_dir_name,
        ]

        for cache_path in possible_caches:
            snapshots_dir = cache_path / "snapshots"
            if not snapshots_dir.exists():
                continue
            # Перебираем все snapshots (обычно один main)
            for snapshot in snapshots_dir.iterdir():
                if not snapshot.is_dir():
                    continue
                st_file = snapshot / "model.safetensors"
                if st_file.exists() and st_file.stat().st_size > 1024 * 1024:  # >1 МБ
                    return st_file
        return None

    def translate(self, text: str) -> str:
        if not text or not text.strip():
            return ""
        self._ensure_loaded()
        assert self._tok is not None and self._model is not None

        inputs = self._tok(
            text, return_tensors="pt", truncation=True, max_length=512,
        ).to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                forced_bos_token_id=self._forced_bos_token_id,
                max_length=512,
                num_beams=self.num_beams,
                eos_token_id=self._eos_token_id,
                pad_token_id=self._pad_token_id,
            )
        return self._tok.decode(out[0], skip_special_tokens=True).strip()

    def translate_batch(self, texts: list[str]) -> list[str]:
        if not texts:
            return []
        self._ensure_loaded()
        assert self._tok is not None and self._model is not None
        out_all: list[str] = []
        BATCH = 8  # меньше батч для NLLB — она больше
        for i in range(0, len(texts), BATCH):
            chunk = [t for t in texts[i:i + BATCH] if t and t.strip()]
            if len(chunk) != len(texts[i:i + BATCH]):
                for t in texts[i:i + BATCH]:
                    out_all.append(self.translate(t))
                continue
            inputs = self._tok(
                chunk, return_tensors="pt", truncation=True,
                max_length=512, padding=True,
            ).to(self._model.device)
            with torch.no_grad():
                out = self._model.generate(
                    **inputs,
                    forced_bos_token_id=self._forced_bos_token_id,
                    max_length=512,
                    num_beams=self.num_beams,
                    eos_token_id=self._eos_token_id,
                    pad_token_id=self._pad_token_id,
                )
            decoded = self._tok.batch_decode(out, skip_special_tokens=True)
            out_all.extend(s.strip() for s in decoded)
        return out_all


class SeamlessM4TTranslator:
    """Локальный EN→RU перевод текста через SeamlessM4T v1 или v2."""

    def __init__(self, engine: str = "seamless_v2", device: str = "cpu", num_beams: int = 4):
        self.engine = engine
        self.device = "cpu"
        self.num_beams = max(1, int(num_beams))
        self._runtime = None

    def _ensure_loaded(self):
        if self._runtime is None:
            from .seamless import SeamlessRuntime
            self._runtime = SeamlessRuntime.get(self.engine, self.device)

    def translate(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        self._ensure_loaded()
        rt = self._runtime
        inputs = rt.processor(text=text, src_lang="eng", return_tensors="pt")
        inputs = {k: v.to("cpu") for k, v in inputs.items() if hasattr(v, "to")}
        with rt.torch.inference_mode():
            output = rt.model.generate(
                **inputs,
                tgt_lang="rus",
                generate_speech=False,
                text_num_beams=self.num_beams,
            )
        tokens = output[0] if isinstance(output, (tuple, list)) else output
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()
        # Generation output is normally [batch, sequence].
        while isinstance(tokens, list) and tokens and isinstance(tokens[0], list):
            tokens = tokens[0]
        return rt.processor.decode(tokens, skip_special_tokens=True).strip()

    def translate_batch(self, texts: list[str]) -> list[str]:
        # Keep inference sequential: the large checkpoint is memory intensive.
        return [self.translate(text) for text in texts]


def make_translator(settings):
    """Factory for NLLB-200 and SeamlessM4T v1/v2."""
    engine = (getattr(settings, "translator_engine", "nllb") or "nllb").lower()
    if engine in ("seamless_v1", "seamless_v2"):
        return SeamlessM4TTranslator(
            engine=engine,
            device=getattr(settings, "translator_device", "cpu"),
            num_beams=getattr(settings, "translator_beams", 4),
        )
    return EnRuTranslator(
        model_name=getattr(settings, "translator_model", "facebook/nllb-200-distilled-600M"),
        device=getattr(settings, "translator_device", "cpu"),
        num_beams=getattr(settings, "translator_beams", 4),
    )
