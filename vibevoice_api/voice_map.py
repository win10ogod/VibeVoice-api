from __future__ import annotations

import os
from typing import Dict, Optional
import yaml


class VoiceMapper:
    """Resolve a `voice` name to a reference .wav sample used by VibeVoice.

    This mapper scans `demo/voices` for `.wav` files and maps friendly names to
    file paths. It loosely normalizes names similar to the demo utility so that
    `Andrew`, `andrew`, `voice_andrew`, etc. resolve to the same file when
    possible.
    """

    def __init__(self, root_dir: str) -> None:
        self.root_dir = root_dir
        self._voices_dir = os.path.join(root_dir, "demo", "voices")
        self._name_to_path: Dict[str, str] = {}
        self._scan()
        self._load_yaml_aliases()

    def _scan(self) -> None:
        self._name_to_path.clear()
        if not os.path.isdir(self._voices_dir):
            return
        for entry in os.listdir(self._voices_dir):
            if not entry.lower().endswith(".wav"):
                continue
            name = os.path.splitext(entry)[0]
            path = os.path.join(self._voices_dir, entry)
            # direct
            self._name_to_path[name] = path
            # normalized variants
            simple = name.replace("_", "-")
            self._name_to_path[simple] = path
            if "-" in simple:
                self._name_to_path[simple.split("-")[-1]] = path

        # Map common OpenAI TTS voice names to closest local samples if available
        # This improves out-of-the-box compatibility when users pass e.g. "alloy", "ash", etc.
        alias_map = {
            # male-like
            "alloy": "en-Frank_man",
            "ash": "en-Carter_man",
            "echo": "en-Carter_man",
            "verse": "en-Frank_man",
            "cedar": "en-Carter_man",
            "marin": "en-Frank_man",
            # female-like
            "coral": "en-Maya_woman",
            "sage": "en-Maya_woman",
            "shimmer": "en-Alice_woman",
            "ballad": "en-Alice_woman",
        }
        for alias, target in alias_map.items():
            # assign alias only if target exists in our map
            for key in list(self._name_to_path.keys()):
                if key.lower() == target.lower():
                    self._name_to_path[alias] = self._name_to_path[key]
                    break

    def _load_yaml_aliases(self) -> None:
        """Load optional YAML-defined alias mappings.

        Search order:
        - $VIBEVOICE_VOICE_MAP if set (absolute or relative to root_dir)
        - <root>/voice_map.yaml
        - <root>/config/voice_map.yaml
        YAML structure:
          simple mapping, e.g.
            alloy: en-Alice_woman   # map to scanned name
            promo: demo/voices/custom.wav  # map to explicit path
          or under a top-level key:
            aliases:
              sleek: en-Frank_man
        """
        candidates = []
        env_path = os.environ.get("VIBEVOICE_VOICE_MAP")
        if env_path:
            p = env_path
            if not os.path.isabs(p):
                p = os.path.join(self.root_dir, p)
            candidates.append(p)
        candidates.extend([
            os.path.join(self.root_dir, "voice_map.yaml"),
            os.path.join(self.root_dir, "config", "voice_map.yaml"),
        ])

        path = next((p for p in candidates if os.path.exists(p)), None)
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            return

        if not isinstance(data, dict):
            return
        mapping = data.get("aliases") if isinstance(data.get("aliases"), dict) else data
        if not isinstance(mapping, dict):
            return

        # Build normalized alias map
        audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".aac", ".m4a"}
        for alias, target in mapping.items():
            if not isinstance(alias, str) or not isinstance(target, str):
                continue
            val = target.strip()
            # path mapping
            _, ext = os.path.splitext(val)
            if os.path.sep in val or ext.lower() in audio_exts:
                p = val
                if p.startswith("file://"):
                    p = p[7:]
                if not os.path.isabs(p):
                    p = os.path.join(self.root_dir, p)
                if os.path.exists(p):
                    self._name_to_path[alias] = p
                continue
            # name mapping: direct or relaxed
            # exact
            for key, path in self._name_to_path.items():
                if key.lower() == val.lower():
                    self._name_to_path[alias] = path
                    break
            else:
                # partial contains
                for key, path in self._name_to_path.items():
                    if key.lower() in val.lower() or val.lower() in key.lower():
                        self._name_to_path[alias] = path
                        break

    def resolve(self, voice: Optional[str]) -> Optional[str]:
        if not self._name_to_path:
            return None
        if not voice:
            # default to first one available
            return list(self._name_to_path.values())[0]
        v = voice.strip()
        # exact
        if v in self._name_to_path:
            return self._name_to_path[v]
        # lowercase / relaxed
        v_lower = v.lower()
        for key, path in self._name_to_path.items():
            if key.lower() == v_lower:
                return path
        # partial contains
        for key, path in self._name_to_path.items():
            if key.lower() in v_lower or v_lower in key.lower():
                return path
        # fallback to first
        return list(self._name_to_path.values())[0]

    def available(self) -> Dict[str, str]:
        return dict(self._name_to_path)
