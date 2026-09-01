#!/usr/bin/env python3
"""
Video Audio Extractor for CohereX
Extracts high-quality audio tracks from video and movie files for speech recognition.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Union, Tuple


def find_ffmpeg() -> str:
    """Finds the ffmpeg executable from PATH or common conda paths."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path
    
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        potential = os.path.join(conda_prefix, "Scripts", "ffmpeg.exe")
        if os.path.isfile(potential):
            return potential
    
    user_home = os.path.expanduser("~")
    common_locations = [
        os.path.join(user_home, "miniconda3", "Scripts", "ffmpeg.exe"),
        os.path.join(user_home, "anaconda3", "Scripts", "ffmpeg.exe"),
        "C:\\ffmpeg\\bin\\ffmpeg.exe",
    ]
    for loc in common_locations:
        if os.path.isfile(loc):
            return loc
            
    raise RuntimeError(
        "FFmpeg executable not found. Please ensure FFmpeg is installed and available on PATH."
    )


def extract_audio_from_video(
    video_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    sample_rate: int = 16000,
    channels: int = 1,
    audio_format: str = "wav",
    overwrite: bool = True,
    start_time: Optional[str] = None,
    duration: Optional[str] = None
) -> str:
    """
    Extracts an audio track from a video/movie file using FFmpeg.
    """
    video_path = Path(video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    if output_path is None:
        output_dir = video_path.parent
        output_filename = f"{video_path.stem}_extracted_audio.{audio_format}"
        output_path = output_dir / output_filename
    else:
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_bin = find_ffmpeg()

    cmd = [
        ffmpeg_bin,
        "-y" if overwrite else "-n",
    ]

    if start_time:
        cmd.extend(["-ss", str(start_time)])
    if duration:
        cmd.extend(["-t", str(duration)])

    cmd.extend([
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le" if audio_format == "wav" else "libmp3lame",
        "-ar", str(sample_rate),
        "-ac", str(channels),
        str(output_path)
    ])

    process = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False
    )

    if process.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed with return code {process.returncode}:\n{process.stderr}"
        )

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"FFmpeg finished but output file is missing or empty at: {output_path}")

    return str(output_path)


def get_media_info(media_path: Union[str, Path]) -> dict:
    """Retrieves duration and file size of a media file."""
    path = Path(media_path)
    if not path.exists():
        return {}
    
    size_mb = path.stat().st_size / (1024 * 1024)
    return {
        "filename": path.name,
        "extension": path.suffix.lower(),
        "size_mb": round(size_mb, 2),
        "is_video": path.suffix.lower() in [".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v"]
    }
