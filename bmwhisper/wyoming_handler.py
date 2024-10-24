"""Event handler for clients of the server."""
import argparse
import asyncio
import logging
import os
import tempfile
import wave
from typing import TYPE_CHECKING, Optional

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

from .transcribe import transcribe
from .model import Whisper

_LOGGER = logging.getLogger(__name__)


class FasterWhisperEventHandler(AsyncEventHandler):
    """Event handler for clients."""
  
    def __init__(
        self,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        model: Whisper,
        model_lock: asyncio.Lock,
        temperature: float,
        language: str = "en",
        *args,
        initial_prompt: Optional[str] = None,
        **kwargs,
    ) -> None:
        # # Weird problem reader and writer, had to extract separately
        reader_writer, *_ = args
        writer = reader_writer
        reader = reader_writer._reader

        super().__init__(reader, writer)
        _LOGGER.info("Starting Handler")

        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.model = model
        self.model_lock = model_lock
        self.temperature = temperature
        self.initial_prompt = initial_prompt
        self._language = language
        self._wav_dir = tempfile.TemporaryDirectory()
        self._wav_path = os.path.join(self._wav_dir.name, "speech.wav")
        self._wav_file: Optional[wave.Wave_write] = None
        

    async def handle_event(self, event: Event) -> bool:
        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)

            if self._wav_file is None:
                self._wav_file = wave.open(self._wav_path, "wb")
                self._wav_file.setframerate(chunk.rate)
                self._wav_file.setsampwidth(chunk.width)
                self._wav_file.setnchannels(chunk.channels)

            self._wav_file.writeframes(chunk.audio)
            return True

        if AudioStop.is_type(event.type):
            _LOGGER.debug(
                "Audio stopped. Transcribing with initial prompt=%s",
                self.initial_prompt,
            )
            assert self._wav_file is not None

            self._wav_file.close()
            self._wav_file = None

            async with self.model_lock:
                # Initialise variables
                self.model.init_cnt()
                # Run the transcriber
                segments = transcribe(self.model, self._wav_path, **self.cli_args)
                print(segments)

                # Extract text from dict
                text = segments["text"]
                _LOGGER.info(text)

                # Write to Wyoming protocol
                await self.write_event(Transcript(text=text).event())
                _LOGGER.debug("Completed request")

                # Reset
                self._language = self.cli_args["language"]
                return False

        if Transcribe.is_type(event.type):
            transcribe_event = Transcribe.from_event(event)
            if transcribe_event.language:
                self._language = transcribe_event.language
                _LOGGER.debug("Language set to %s", transcribe_event.language)
            return True

        if Describe.is_type(event.type):
            print(self.wyoming_info_event)
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        return True