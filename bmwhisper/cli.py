import argparse
import asyncio
import warnings
import time

from functools import partial
import numpy as np
import torch
from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer

from . import load_model
from .wyoming_handler import FasterWhisperEventHandler
from .tokenizer import LANGUAGES, TO_LANGUAGE_CODE
from .utils import (
    optional_float,
    optional_int,
    str2bool,
)
from .version import __version__

async def run_cli() -> None:
    start_time = time.time()
    from . import available_models

    # fmt: off
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", default="small", choices=available_models(), help="name of the Whisper model to use")
    parser.add_argument("--model_dir", type=str, default=None, help="the path to save model files; uses ~/.cache/whisper by default")
    parser.add_argument("--bmodel_dir", type=str, default="./bmodel", help="the path to save model files; uses ./bmodel by default")
    # parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="device to use for PyTorch inference")
    parser.add_argument("--verbose", type=str2bool, default=True, help="whether to print out the progress and debug messages")

    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"], help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
    parser.add_argument("--language", type=str, default="en", choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]), help="language spoken in the audio, specify None to perform language detection")

    parser.add_argument("--temperature", type=float, default=0, help="temperature to use for sampling")
    parser.add_argument("--best_of", type=optional_int, default=5, help="number of candidates when sampling with non-zero temperature")
    parser.add_argument("--beam_size", type=optional_int, default=5, help="number of beams in beam search, only applicable when temperature is zero")
    parser.add_argument("--patience", type=float, default=None, help="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search")
    parser.add_argument("--length_penalty", type=float, default=None, help="optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple length normalization by default")

    parser.add_argument("--suppress_tokens", type=str, default="-1", help="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations")
    parser.add_argument("--initial_prompt", type=str, default=None, help="optional text to provide as a prompt for the first window.")
    parser.add_argument("--condition_on_previous_text", type=str2bool, default=True, help="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop")

    parser.add_argument("--temperature_increment_on_fallback", type=optional_float, default=0.2, help="temperature to increase when falling back when the decoding fails to meet either of the thresholds below")
    parser.add_argument("--compression_ratio_threshold", type=optional_float, default=2.4, help="if the gzip compression ratio is higher than this value, treat the decoding as failed")
    parser.add_argument("--logprob_threshold", type=optional_float, default=-1.0, help="if the average log probability is lower than this value, treat the decoding as failed")
    parser.add_argument("--no_speech_threshold", type=optional_float, default=0.6, help="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence")
    parser.add_argument("--word_timestamps", type=str2bool, default=False, help="(experimental) extract word-level timestamps and refine the results based on them")
    parser.add_argument("--prepend_punctuations", type=str, default="\"\'“¿([{-", help="if word_timestamps is True, merge these punctuation symbols with the next word")
    parser.add_argument("--append_punctuations", type=str, default="\"\'.。,，!！?？:：”)]}、", help="if word_timestamps is True, merge these punctuation symbols with the previous word")
    parser.add_argument("--highlight_words", type=str2bool, default=False, help="(requires --word_timestamps True) underline each word as it is spoken in srt and vtt")
    parser.add_argument("--max_line_width", type=optional_int, default=None, help="(requires --word_timestamps True) the maximum number of characters in a line before breaking the line")
    parser.add_argument("--max_line_count", type=optional_int, default=None, help="(requires --word_timestamps True) the maximum number of lines in a segment")
    parser.add_argument("--threads", type=optional_int, default=0, help="number of threads used by torch for CPU inference; supercedes MKL_NUM_THREADS/OMP_NUM_THREADS")
    parser.add_argument("--padding_size", type=optional_int, default=448, help="max pre-allocation size for the key-value cache")
    parser.add_argument("--chip_mode", default="pcie", choices=["pcie", "soc"], help="name of the Whisper model to use")
    # fmt: on

    args = parser.parse_args()
    args_dict = parser.parse_args().__dict__
    args.model_name = args.model

    model_name = args.model_name
    if model_name.endswith(".en") and args.language not in {"en", "English"}:
        if args.language is not None:
            warnings.warn(
                f"{model_name} is an English-only model but receipted '{args.language}'; using English instead."
            )
        args.language = "en"

    temperature = args.temperature
    if (increment := args.temperature_increment_on_fallback) is not None:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, increment))
    else:
        temperature = [temperature]

    if (threads := args.threads) > 0:
        torch.set_num_threads(threads)

    model = load_model(args) 

    # We need to insert Wyoming protocol similar to wyoming-faster-whisper    
    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="bmwhisper",
                description="Whisper transcription using Sophon TPU",
                attribution=Attribution(
                    name="JKay0327",
                    url="https://github.com/JKay0327/whisper-TPU_py",
                ),
                installed=True,
                version=__version__,
                models=[
                    AsrModel(
                        name=model_name,
                        description=model_name,
                        attribution=Attribution(
                            name="Systran",
                            url="https://huggingface.co/Systran",
                        ),
                        installed=True,
                        languages=tuple(LANGUAGES.keys()),
                        version=__version__,
                    )
                ],
            )
        ],
    )
    
    pop_list = ["model", "model_dir", "bmodel_dir", "chip_mode", "temperature_increment_on_fallback", "threads", "max_line_width", "max_line_count", "highlight_words"]
    for arg in pop_list:
        args_dict.pop(arg)
    
    # Load model
    print("Starting Wyoming Server")
    server = AsyncServer.from_uri('tcp://0.0.0.0:10300')
    model_lock = asyncio.Lock()
    
    await server.run(
        partial(
            FasterWhisperEventHandler,
            wyoming_info,
            args_dict,
            model,
            model_lock,
            temperature,
            initial_prompt=args.initial_prompt,
        )
    )

# -----------------------------------------------------------------------------

def run() -> None:
    asyncio.run(run_cli())

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass